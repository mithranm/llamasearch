# src/llamasearch/core/query_processor.py
import difflib
import html
import logging
import threading
import time
from typing import Any, Dict, List, Optional, cast

import chromadb
import numpy as np
from chromadb.api.types import QueryResult, Metadata, Document

from .bm25 import WhooshBM25Retriever
from .embedder import EnhancedEmbedder

# --- Import GenericONNXLLM ---
from .onnx_model import GenericONNXLLM

from ..protocols import LLM
from ..utils import log_query

logger = logging.getLogger(__name__)

CHUNK_SIMILARITY_THRESHOLD = 0.85
THINK_END_TAG = "</think>"  # Define the tag for splitting


def _are_chunks_too_similar(text1: str, text2: str, threshold: float) -> bool:
    """Checks if two text chunks are too similar using SequenceMatcher."""
    if not text1 or not text2:
        return False
    return difflib.SequenceMatcher(None, text1, text2).ratio() > threshold


class _QueryProcessingMixin:
    model: Optional[LLM]
    embedder: Optional[EnhancedEmbedder]
    chroma_collection: Optional[chromadb.Collection]
    bm25: Optional[WhooshBM25Retriever]
    max_results: int
    bm25_weight: float
    vector_weight: float
    context_length: int
    debug: bool
    _shutdown_event: Optional[threading.Event]

    def _get_token_count(self, text: str) -> int:
        """Estimates token count for given text."""
        if not text:
            return 0
        # --- Check instance type (Only GenericONNXLLM now) ---
        if (
            self.model
            and isinstance(self.model, GenericONNXLLM)
            and hasattr(self.model, "_tokenizer")
            and self.model._tokenizer is not None
        ):
            try:
                tokenizer = getattr(self.model, "_tokenizer", None)
                if tokenizer and hasattr(tokenizer, "__call__"):
                    # Use truncation=False to get accurate count, though context building handles limits
                    token_output = tokenizer(
                        text, add_special_tokens=False, truncation=False
                    )
                    input_ids = token_output.get("input_ids")
                    if isinstance(input_ids, list):
                        return len(input_ids)
            except Exception as tok_err:
                logger.warning(f"Tokenizer error: {tok_err}", exc_info=self.debug)
                return max(1, len(text) // 4)  # Fallback

        return max(1, len(text) // 4)  # Fallback estimation

    def llm_query(self, query_text: str, debug_mode: bool = False) -> Dict[str, Any]:
        """Processes a query: retrieves context, generates response, returns structured dict."""
        start_time_total = time.time()
        assert self.model is not None, "LLM not initialized"
        assert self.embedder is not None, "Embedder not initialized"
        assert self.chroma_collection is not None, "Chroma collection not initialized"
        assert self.bm25 is not None, "BM25 retriever not initialized"

        # --- Modified System Instruction ---
        # System prompt is handled by the chat template application now.
        # We only need the core user query instruction here.
        user_instruction = (
            f"Answer the query using *only* the provided Context. "
            f"If the answer isn't in the Context, state that clearly. Query: {query_text}"
        )
        # ---------------------------------
        debug_info: Dict[str, Any] = {}
        retrieval_time, gen_time, query_embedding_time = -1.0, -1.0, 0.0
        final_context, retrieved_display_html = "", ""
        context_chunks_details: List[Dict[str, Any]] = []
        query_embedding: Optional[List[float]] = None
        vector_results: Optional[QueryResult] = None
        bm25_results: Dict[str, List[Any]] = {}
        combined_scores: Dict[str, float] = {}
        doc_lookup: Dict[str, Dict[str, Any]] = {}
        vec_results_count, bm25_results_count = 0, 0
        retrieval_start = time.time()
        num_candidates = max(self.max_results * 5, 25)

        # --- Retrieval Phase (No changes needed here) ---
        try:
            embed_start = time.time()
            embeddings_array = self.embedder.embed_strings(
                [query_text], input_type="query"
            )
            query_embedding_time = time.time() - embed_start
            debug_info["query_embedding_time"] = f"{query_embedding_time:.3f}s"

            if (
                embeddings_array is None
                or not isinstance(embeddings_array, np.ndarray)
                or embeddings_array.ndim != 2
                or embeddings_array.shape[0] == 0
            ):
                raise ValueError("Failed to generate valid query embedding.")
            query_embedding = embeddings_array[0].tolist()

            vector_weight_ratio = self.vector_weight / (
                self.vector_weight + self.bm25_weight + 1e-9
            )
            num_vector_results = max(1, int(num_candidates * vector_weight_ratio))
            num_bm25_results = max(1, int(num_candidates * (1.0 - vector_weight_ratio)))
            logger.debug(
                f"Candidates: Vector={num_vector_results}, BM25={num_bm25_results}"
            )

            logger.debug(f"Querying ChromaDB (n={num_vector_results})...")
            assert query_embedding is not None
            vector_results = self.chroma_collection.query(
                query_embeddings=[query_embedding],
                n_results=num_vector_results,
                include=["metadatas", "documents", "distances"],
            )
            vec_results_count = (
                len(vector_results.get("ids", [[]])[0]) if vector_results else 0
            )
            logger.debug(f"ChromaDB returned {vec_results_count} results.")

            logger.debug(f"Querying Whoosh BM25 (n={num_bm25_results})...")
            bm25_results = self.bm25.query(query_text, n_results=num_bm25_results)
            bm25_results_count = len(bm25_results.get("ids", []))
            logger.debug(f"Whoosh BM25 returned {bm25_results_count} results.")

        except Exception as e:
            logger.error(f"Retrieval phase failed: {e}", exc_info=debug_mode)
            vector_results, bm25_results = None, {}
            vec_results_count, bm25_results_count = 0, 0
            combined_scores, doc_lookup = {}, {}

        retrieval_time = time.time() - retrieval_start
        debug_info["retrieval_time"] = f"{retrieval_time:.3f}s"
        debug_info["vector_initial_results"] = vec_results_count
        debug_info["bm25_initial_results"] = bm25_results_count
        k_rrf = 60.0

        # --- Combine results using RRF (No changes needed here) ---
        if vector_results:
            ids_list: Optional[List[List[str]]] = vector_results.get("ids")
            dist_list: Optional[List[List[float]]] = vector_results.get("distances")
            meta_list: Optional[List[List[Metadata]]] = vector_results.get("metadatas")
            doc_list: Optional[List[List[Document]]] = vector_results.get("documents")

            if ids_list and len(ids_list) > 0 and isinstance(ids_list[0], list):
                ids = ids_list[0]
                list_len = len(ids)
                dists = (
                    dist_list[0]
                    if dist_list
                    and len(dist_list) > 0
                    and len(dist_list[0]) == list_len
                    else [0.0] * list_len
                )
                metas = (
                    meta_list[0]
                    if meta_list
                    and len(meta_list) > 0
                    and len(meta_list[0]) == list_len
                    else [{}] * list_len
                )
                docs = (
                    doc_list[0]
                    if doc_list and len(doc_list) > 0 and len(doc_list[0]) == list_len
                    else [""] * list_len
                )

                for i, chunk_id in enumerate(ids):
                    rank_vec = i + 1
                    rrf_score_vec = 1.0 / (k_rrf + rank_vec)
                    combined_scores[chunk_id] = (
                        combined_scores.get(chunk_id, 0.0)
                        + rrf_score_vec * self.vector_weight
                    )
                    if chunk_id not in doc_lookup:
                        doc_lookup[chunk_id] = {
                            "document": str(docs[i]),
                            "metadata": dict(metas[i]),
                        }
                    if debug_mode:
                        dist = dists[i]
                        score = 1.0 - (dist / 2.0)
                        debug_info.setdefault("chunk_details", {}).setdefault(
                            chunk_id, {}
                        ).update(
                            {
                                "vector_rank": rank_vec,
                                "vector_dist": f"{dist:.4f}",
                                "vector_score": f"{score:.4f}",
                                "vector_rrf": f"{rrf_score_vec * self.vector_weight:.4f}",
                            }
                        )

        if bm25_results and "ids" in bm25_results:
            bm25_ids = bm25_results.get("ids", [])
            bm25_scores = bm25_results.get("scores", [])
            if len(bm25_scores) != len(bm25_ids):
                bm25_scores = [0.0] * len(bm25_ids)
            for i, chunk_id in enumerate(bm25_ids):
                rank_bm25 = i + 1
                score_val = bm25_scores[i]
                rrf_score_bm25 = 1.0 / (k_rrf + rank_bm25)
                combined_scores[chunk_id] = (
                    combined_scores.get(chunk_id, 0.0)
                    + rrf_score_bm25 * self.bm25_weight
                )
                doc_lookup.setdefault(chunk_id, {"document": None, "metadata": None})
                if debug_mode:
                    debug_info.setdefault("chunk_details", {}).setdefault(
                        chunk_id, {}
                    ).update(
                        {
                            "bm25_rank": rank_bm25,
                            "bm25_raw_score": f"{score_val:.4f}",
                            "bm25_rrf": f"{rrf_score_bm25 * self.bm25_weight:.4f}",
                        }
                    )

        debug_info["combined_unique_chunks"] = len(combined_scores)
        sorted_chunks = sorted(
            combined_scores.items(), key=lambda item: item[1], reverse=True
        )

        # --- Candidate Filtering & Context Building (No changes needed here) ---
        num_candidates_to_filter = min(len(sorted_chunks), self.max_results * 3)
        top_candidates_with_scores = sorted_chunks[:num_candidates_to_filter]
        filtered_top_chunks_with_scores = []
        added_chunk_texts = set()
        skipped_due_similarity = 0
        logger.debug(
            f"Processing {len(top_candidates_with_scores)} candidates for context (Max Results: {self.max_results})..."
        )

        for chunk_id, score in top_candidates_with_scores:
            if len(filtered_top_chunks_with_scores) >= self.max_results:
                break
            if (
                chunk_id not in doc_lookup
                or not doc_lookup[chunk_id].get("document")
                or not doc_lookup[chunk_id].get("metadata")
            ):
                logger.debug(f"Fetching missing doc/meta for chunk {chunk_id}...")
                try:
                    chroma_get = self.chroma_collection.get(
                        ids=[chunk_id], include=["documents", "metadatas"]
                    )
                    c_docs = chroma_get.get("documents", [])
                    c_metas = chroma_get.get("metadatas", [])
                    if (
                        c_docs
                        and isinstance(c_docs[0], str)
                        and c_metas
                        and isinstance(c_metas[0], dict)
                    ):
                        doc_lookup[chunk_id] = {
                            "document": c_docs[0],
                            "metadata": cast(Dict[str, Any], c_metas[0]),
                        }
                    else:
                        logger.warning(
                            f"Chroma fetch for {chunk_id} failed/incomplete. Skipping."
                        )
                        continue
                except Exception as fetch_err:
                    logger.warning(f"Error fetching {chunk_id}: {fetch_err}. Skipping.")
                    continue

            current_text = doc_lookup[chunk_id].get("document", "")
            if not current_text or not current_text.strip():
                logger.debug(f"Skipping chunk {chunk_id}: Empty text.")
                continue

            is_too_similar = any(
                _are_chunks_too_similar(
                    current_text, existing, CHUNK_SIMILARITY_THRESHOLD
                )
                for existing in added_chunk_texts
            )
            if is_too_similar:
                skipped_due_similarity += 1
                logger.debug(f"Skipping chunk {chunk_id}: Too similar.")
            else:
                filtered_top_chunks_with_scores.append((chunk_id, score))
                added_chunk_texts.add(current_text)

        top_chunk_ids_with_scores = filtered_top_chunks_with_scores
        top_chunk_ids = [cid for cid, _ in top_chunk_ids_with_scores]
        debug_info["skipped_similar_chunk_count"] = skipped_due_similarity
        debug_info["final_selected_chunk_count"] = len(top_chunk_ids)
        logger.info(f"Selected {len(top_chunk_ids)} final unique chunks for context.")

        # --- Context Construction & LLM Call ---
        if not top_chunk_ids:
            response = "Could not find relevant information."
            query_time = retrieval_time if retrieval_time > 0 else query_embedding_time
            return {
                "response": response,
                "debug_info": debug_info if debug_mode else {},
                "retrieved_context": "",
                "formatted_response": "...",
                "query_time_seconds": query_time,
                "generation_time_seconds": 0,
            }

        # --- Build Context for LLM ---
        temp_context_parts, temp_display_parts_html = [], []
        # The base prompt is now constructed within the LLM's generate method using chat template
        # We estimate token count based on the *user instruction* and the *context content* only
        instruction_len = self._get_token_count(user_instruction)
        # Reserve tokens for system prompt, chat markers, and generation buffer
        generation_reservation = max(
            300, self.context_length // 4
        )  # Increased reservation
        template_overhead_estimate = 50  # Estimate for system prompt + chat markers
        available_context_tokens = max(
            0,
            self.context_length
            - instruction_len
            - template_overhead_estimate
            - generation_reservation,
        )
        current_context_len = 0

        for i, (chunk_id, score) in enumerate(top_chunk_ids_with_scores):
            doc_data = doc_lookup.get(chunk_id)
            if not doc_data:
                continue
            doc_text = doc_data.get("document", "")
            metadata = doc_data.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}
            if not doc_text or not doc_text.strip():
                continue

            original_url = metadata.get("original_url")
            source_path = metadata.get("source_path", "Unknown")
            display_source = (
                str(original_url).strip()
                if isinstance(original_url, str) and original_url.strip()
                else str(source_path).strip()
            )
            # Context for LLM only needs the text content now
            doc_chunk_full_text_llm = f"{doc_text}\n\n"  # Simple text content

            # Display HTML remains the same
            filename_meta = metadata.get("filename", "N/A")
            original_chunk_index = metadata.get("original_chunk_index", "N/A")
            filename_display_part = (
                f" | File: {filename_meta}"
                if filename_meta != "N/A" and display_source != source_path
                else ""
            )
            header_html = f'<div style="border-top: 1px solid #eee; margin-top: 10px; padding-top: 5px; font-size: 0.9em; color: #333;"><b>Rank {i + 1}</b> (Score: {score:.4f}) | Source: {html.escape(display_source)}{html.escape(filename_display_part)} | Chunk: {original_chunk_index}</div>'
            doc_chunk_display_html = (
                f"<p>{html.escape(doc_text).replace(chr(10), '<br>')}</p>"
            )

            doc_chunk_len = self._get_token_count(
                doc_chunk_full_text_llm
            )  # Count only the content
            if current_context_len + doc_chunk_len <= available_context_tokens:
                temp_context_parts.append(doc_chunk_full_text_llm)
                temp_display_parts_html.append(f"{header_html}{doc_chunk_display_html}")
                current_context_len += doc_chunk_len
                context_chunks_details.append(
                    {
                        "id": chunk_id,
                        "rank": i + 1,
                        "score": score,
                        "token_count": doc_chunk_len,
                        "source": display_source,
                        "original_chunk_index": original_chunk_index,
                        "original_url": original_url,
                        "source_path": source_path,
                        "filename": filename_meta,
                    }
                )
            else:
                logger.warning(
                    f"Stopping context build at rank {i + 1}. Limit reached ({current_context_len + doc_chunk_len} > {available_context_tokens} context tokens)."
                )
                debug_info["context_truncated_at_chunk_rank"] = i + 1
                break

        final_context = "".join(
            temp_context_parts
        ).strip()  # This is the context to pass to the LLM prompt wrapper
        retrieved_display_html = (
            "".join(temp_display_parts_html).strip()
            or "<p><i>No relevant chunks selected.</i></p>"
        )
        debug_info["final_context_content_token_count"] = current_context_len
        debug_info["final_context_content_chars"] = len(final_context)

        if not final_context and top_chunk_ids:
            response = "Error: Could not build context from selected chunks."
            logger.error(response)
            # ... (return error dictionary as before) ...
            query_time = retrieval_time if retrieval_time > 0 else query_embedding_time
            return {
                "response": response,
                "debug_info": debug_info if debug_mode else {},
                "retrieved_context": retrieved_display_html,
                "formatted_response": "...",
                "query_time_seconds": query_time,
                "generation_time_seconds": 0,
            }

        # --- Modified Prompt Construction ---
        # The full prompt is now built inside model.generate using the chat template.
        # We pass the user instruction which *includes* the context.
        prompt_for_llm = f"{user_instruction}\n\nContext:\n{final_context}"
        # Estimate total tokens (rough check for sanity)
        prompt_token_count = (
            self._get_token_count(prompt_for_llm) + template_overhead_estimate
        )
        debug_info["estimated_full_prompt_tokens"] = prompt_token_count
        if debug_mode:
            logger.debug(
                f"--- LLM User Instruction + Context ({prompt_token_count} est. tokens) ---\n{prompt_for_llm[:1000]}...\n--- LLM Prompt End ---"
            )
        # ----------------------------------

        # Sanity check token count before calling LLM
        if prompt_token_count >= self.context_length:
            response = f"Error: Estimated prompt too long ({prompt_token_count} >= {self.context_length})."
            logger.error(response)
            query_time = retrieval_time if retrieval_time > 0 else query_embedding_time
            return {
                "response": response,
                "debug_info": debug_info if debug_mode else {},
                "retrieved_context": retrieved_display_html,
                "formatted_response": "...",
                "query_time_seconds": query_time,
                "generation_time_seconds": 0,
            }

        logger.info("Generating response with LLM...")
        gen_start = time.time()
        llm_full_output = "LLM Error."  # Stores the potentially combined think+answer
        raw_llm_output = None
        assert self.model is not None

        try:
            if self._shutdown_event and self._shutdown_event.is_set():
                raise InterruptedError("Shutdown before LLM generation.")
            max_gen_tokens = max(50, self.context_length - prompt_token_count - 15)
            debug_info["llm_max_gen_tokens"] = max_gen_tokens

            # Call generate with the *instruction + context* prompt
            # Model's generate method handles the chat template wrapping
            llm_full_output, raw_llm_output = self.model.generate(
                prompt=prompt_for_llm,  # Pass instruction + context
                max_tokens=max_gen_tokens,  # Use calculated max tokens
            )

        except InterruptedError:
            llm_full_output = "LLM generation cancelled during shutdown."
            logger.warning(llm_full_output)
        except Exception as e:
            logger.error(f"LLM generation error: {e}", exc_info=debug_mode)
            raw_llm_output = {"error": str(e)}
            llm_full_output = f"LLM Error: {e}"

        gen_time = time.time() - gen_start
        logger.info(f"LLM generation took {gen_time:.2f}s.")
        debug_info["llm_generation_time"] = f"{gen_time:.3f}s"
        debug_info["llm_raw_output_metadata"] = (
            raw_llm_output  # Store metadata like token counts
        )

        # --- Split thinking vs response ---
        response = ""  # Final answer for user
        if isinstance(llm_full_output, str):
            response = llm_full_output.strip()  # Use the whole output as the answer
        else:
            # Handle case where LLM output wasn't a string (e.g., error occurred)
            response = str(llm_full_output)  # Use the error message as response
            logger.error(f"LLM output was not a string: {response}")

        # Check for errors reported by the LLM itself (e.g., via metadata)
        if (
            raw_llm_output
            and isinstance(raw_llm_output, dict)
            and raw_llm_output.get("error")
        ):
            response = (
                f"LLM Error: {raw_llm_output['error']}"  # Overwrite response with error
            )
            logger.error(f"LLM reported error: {response}")
        elif not response or response.isspace():
            response = (
                "(LLM returned empty response after splitting)"  # More specific message
            )
            logger.warning(response)

        total_time = time.time() - start_time_total
        debug_info["total_query_processing_time"] = f"{total_time:.3f}s"
        formatted_response_html = f"<h2>AI Answer</h2><p>{html.escape(response)}</p><h2>Retrieved Context</h2>{retrieved_display_html}"

        try:
            # Log the *final user-facing response* and potentially the thinking part in debug
            log_query(
                query=query_text,
                chunks=context_chunks_details,
                response=response,  # Log the cleaned answer
                debug_info=debug_info,  # Contains thinking_content if debug_mode
                full_logging=debug_mode,
            )
        except Exception as log_e:
            logger.warning(f"Failed to log query details: {log_e}")

        query_time_final = (
            retrieval_time if retrieval_time > 0 else query_embedding_time
        )
        logger.info(f"Search Result:\n{response}")

        return {
            "response": response,  # Return the cleaned answer
            "debug_info": debug_info if debug_mode else {},
            "retrieved_context": retrieved_display_html,
            "formatted_response": formatted_response_html,
            "query_time_seconds": query_time_final,
            "generation_time_seconds": gen_time if gen_time > 0 else 0,
        }
