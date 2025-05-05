# src/llamasearch/core/llmsearch.py

import os
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional

from llamasearch.utils import setup_logging
from llamasearch.core.vectordb import VectorDB
from llamasearch.core.embedder import EnhancedEmbedder, DEFAULT_MODEL_NAME
from llamasearch.core.teapot import load_teapot_onnx_llm, TeapotONNXLLM
from llamasearch.protocols import LLM, ModelInfo  # Added ModelInfo import
from llamasearch.exceptions import ModelNotFoundError

logger = setup_logging(__name__)


class LLMSearch:
    """
    A RAG-based search class using the Teapot ONNX model.
    Manages LLM, Embedder, and VectorDB instances. Checks for model availability.
    """

    def __init__(
        self,
        storage_dir: Path,
        teapot_onnx_quant: str = "auto",
        teapot_provider: Optional[str] = None,
        teapot_provider_opts: Optional[Dict[str, Any]] = None,
        verbose: bool = True,
        max_results: int = 3,
        embedder_model: Optional[str] = None,
        embedder_batch_size: int = 32,
        embedder_device: Optional[str] = None,
        vectordb_similarity_threshold: float = 0.25,
        vectordb_max_chunk_size: int = 512,
        vectordb_chunk_overlap: int = 64,
        vectordb_min_chunk_size: int = 128,
        vectordb_collection_name: str = "default",
        max_workers: int = 1,
        debug: bool = False,
    ):
        self.verbose = verbose
        self.max_results = max_results
        self.debug = debug
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger
        self.model: Optional[LLM] = None
        self.embedder: Optional[EnhancedEmbedder] = None
        self.vectordb: Optional[VectorDB] = None
        self.context_length: int = 0  # Initialize context length
        self.llm_device_type: str = "cpu"  # Initialize device type

        # --- Initialize components with error handling for missing models ---
        try:
            # --- Load Teapot ONNX LLM ---
            self.logger.info("Initializing Teapot ONNX LLM for LLMSearch...")
            loaded_model: Optional[LLM] = load_teapot_onnx_llm(
                onnx_quantization=teapot_onnx_quant,
                preferred_provider=teapot_provider,
                preferred_options=teapot_provider_opts,
            )
            # Check if model loading was successful before assigning and accessing
            if loaded_model is None:
                # This case should ideally be caught by exceptions inside load_teapot...
                # but handle defensively.
                raise RuntimeError("load_teapot_onnx_llm returned None unexpectedly.")

            self.model = loaded_model  # Assign only after successful load

            # --- Access model attributes safely *after* assignment ---
            model_info: ModelInfo = self.model.model_info  # Get info object
            self.context_length = model_info.context_length
            if hasattr(self.model, "device"):
                self.llm_device_type = self.model.device.type
                self.logger.info(
                    f"LLMSearch using {model_info.model_id} on device {self.model.device}. Context: {self.context_length}"
                )
            else:
                self.llm_device_type = "cpu"
                self.logger.warning("LLM assuming CPU.")
                self.logger.info(
                    f"LLMSearch using {model_info.model_id}. Context: {self.context_length}"
                )

            # --- Initialize Embedder ---
            compute_device = embedder_device or (
                "cpu" if self.llm_device_type == "cpu" else "cuda"
            )
            self.logger.info(f"Configuring Embedder on device: {compute_device}")
            # EnhancedEmbedder.__init__ handles its own model checks
            self.embedder = EnhancedEmbedder(
                model_name=embedder_model or DEFAULT_MODEL_NAME,
                batch_size=embedder_batch_size,
                num_workers=max_workers,
            )

            # --- Initialize VectorDB ---
            self.logger.info(f"Initializing VectorDB (storage: {self.storage_dir})")
            # VectorDB.__init__ -> BM25Retriever.__init__ -> load_nlp_model checks spaCy
            self.vectordb = VectorDB(
                embedder=self.embedder,  # Pass the successfully initialized embedder
                storage_dir=self.storage_dir,
                collection_name=vectordb_collection_name,
                max_chunk_size=vectordb_max_chunk_size,
                chunk_overlap=vectordb_chunk_overlap,
                min_chunk_size=vectordb_min_chunk_size,
                embedder_batch_size=embedder_batch_size,
                similarity_threshold=vectordb_similarity_threshold,
                max_results=self.max_results,
                device=compute_device,
            )
            self.logger.info("LLMSearch components initialized successfully.")

        except ModelNotFoundError as e:
            self.logger.error(f"LLMSearch initialization failed: {e}")
            self.close()  # Attempt cleanup
            raise  # Re-raise the specific error
        except Exception as e:
            self.logger.error(
                f"Unexpected error during LLMSearch initialization: {e}", exc_info=True
            )
            self.close()  # Attempt cleanup
            raise RuntimeError(
                "LLMSearch failed to initialize due to an unexpected error."
            ) from e

    def add_document(self, file_path: Path) -> int:
        """Adds a source (file/dir) to VectorDB, delegating processing."""
        if not self.vectordb:
            self.logger.error("VectorDB is not initialized.")
            return 0
        file_path = Path(file_path)
        if not file_path.exists():
            self.logger.error(f"Source path not found: {file_path}")
            return 0

        self.logger.info(f"Requesting VectorDB to add source: {file_path}")
        try:
            added_count = self.vectordb.add_source(file_path)
            if added_count > 0:
                logger.info(
                    f"VectorDB added {added_count} chunks from: {file_path.name}"
                )
            else:
                logger.info(
                    f"VectorDB added 0 new chunks for source: {file_path.name}."
                )
            return added_count
        except Exception as e:
            logger.error(
                f"Failed add source {file_path} via VectorDB: {e}", exc_info=self.debug
            )
            return 0

    def add_documents_from_directory(
        self, directory_path: Path, recursive: bool = True
    ) -> int:
        """Adds all processable files from a directory using VectorDB."""
        if not self.vectordb:
            self.logger.error("VectorDB is not initialized.")
            return 0
        directory_path = Path(directory_path)
        if not directory_path.is_dir():
            self.logger.error(f"{directory_path} is not a valid directory.")
            return 0

        total_chunks_added, files_processed, process_count = 0, 0, 0
        files_failed = []
        glob_pattern = "**/*" if recursive else "*"
        logger.info(
            f"Scanning directory {'recursively' if recursive else ''}: {directory_path}"
        )

        for p in directory_path.glob(glob_pattern):
            if p.is_file():
                process_count += 1
                logger.debug(f"Requesting VectorDB to add source: {p}")
                try:
                    added = self.add_document(p)  # Use the single doc method
                    if added > 0:
                        total_chunks_added += added
                        files_processed += 1
                except Exception as e:
                    logger.error(
                        f"Error requesting add for {p.name}: {e}", exc_info=self.debug
                    )
                    files_failed.append(p.name)

        logger.info(
            f"Directory scan complete. Processed {process_count} files. VectorDB added {total_chunks_added} new chunks from {files_processed} files."
        )
        if files_failed:
            logger.warning(
                f"Errors requesting processing for {len(files_failed)} files: {', '.join(files_failed)}"
            )
        return total_chunks_added

    def _get_token_count(self, text: str) -> int:
        """Calculates token count using Teapot's tokenizer or estimation."""
        # Check if model and tokenizer exist before using them
        if (
            self.model
            and isinstance(self.model, TeapotONNXLLM)
            and hasattr(self.model, "_tokenizer")
            and self.model._tokenizer
        ):
            try:
                return len(self.model._tokenizer.encode(text, add_special_tokens=False))
            except Exception as e:
                logger.warning(
                    f"Could not use Teapot tokenizer for count: {e}. Estimating."
                )
        return max(1, len(text) // 4)  # Fallback estimate

    def llm_query(self, query_text: str, debug_mode: bool = False) -> Dict[str, Any]:
        """RAG-based retrieval + LLM generation using Teapot ONNX"""
        # Check initialization before proceeding
        if self.model is None or not hasattr(self.model, "generate"):
            return {
                "response": "Error: LLM not initialized.",
                "formatted_response": "Error: LLM not initialized.",
            }
        if self.vectordb is None:
            return {
                "response": "Error: VectorDB not initialized.",
                "formatted_response": "Error: VectorDB not initialized.",
            }

        debug_info: Dict[str, Any] = {}
        final_context, retrieved_display, query_time, gen_time = "", "", -1.0, -1.0

        try:  # Retrieve Context
            logger.debug("Performing vector search for query: '%s...'", query_text[:50])
            query_start_time = time.time()
            results = self.vectordb.vectordb_query(query_text, max_out=self.max_results)
            query_time = time.time() - query_start_time
            debug_info["vector_query_time"] = f"{query_time:.3f}s"
            debug_info["vector_results_count"] = len(results.get("documents", []))

            docs, metas, scores = (
                results.get("documents", []),
                results.get("metadatas", []),
                results.get("scores", []),
            )
            if not docs:
                final_context = "[No relevant context found in documents]"
                retrieved_display = "No relevant chunks retrieved."
                logger.warning("No relevant context found for query.")
            else:
                logger.info(f"Retrieved {len(docs)} chunks.")
                temp_context, temp_display = "", ""
                # Use self.context_length which was set safely in __init__
                prompt_base_len = self._get_token_count(
                    f"Context:\n\n\nQuery: {query_text}\n\nAnswer:"
                )
                available_context_tokens = (
                    self.context_length - prompt_base_len - 200
                )  # Safety margin
                logger.debug(
                    f"Context limit: {self.context_length}, Base prompt: {prompt_base_len}, Available: {available_context_tokens}"
                )

                for i, doc_text in enumerate(docs):
                    score = scores[i] if i < len(scores) else 0.0
                    source = metas[i].get("source", "N/A") if i < len(metas) else "N/A"
                    chunk_id_meta = (
                        metas[i].get("chunk_id", "N/A") if i < len(metas) else "N/A"
                    )
                    chunk_identifier = (
                        chunk_id_meta
                        if chunk_id_meta != "N/A"
                        else metas[i].get("original_chunk_index", f"docidx_{i}")
                    )

                    header = f"[Doc {i + 1} | Source: {os.path.basename(source)} | Score: {score:.2f}]\n"
                    doc_chunk = f"{header}{doc_text}\n\n"
                    display_chunk = f"--- Chunk {i + 1} (Score: {score:.2f}) ---\nSource: {source}\nChunk ID: {chunk_identifier}\n{doc_text}\n\n"

                    current_context_len = self._get_token_count(temp_context)
                    doc_chunk_len = self._get_token_count(doc_chunk)

                    if current_context_len + doc_chunk_len <= available_context_tokens:
                        temp_context += doc_chunk
                        temp_display += display_chunk
                    else:
                        logger.warning(
                            f"Stopping context inclusion at chunk {i + 1}/{len(docs)} due to limit."
                        )
                        debug_info["context_truncated_at_chunk"] = i + 1
                        break
                final_context, retrieved_display = (
                    temp_context.strip(),
                    temp_display.strip(),
                )
        except Exception as e:
            logger.error(f"Error during vector DB query: {e}", exc_info=self.debug)
            return {
                "response": f"Error during context retrieval: {e}",
                "formatted_response": f"Error: {e}",
            }

        # Construct Prompt
        system_instruction = "Answer the query using *only* the provided Context. If the answer isn't in the Context, say so."
        prompt = f"{system_instruction}\n\nContext:\n{final_context}\n\nQuery: {query_text}\n\nAnswer:"
        debug_info["final_prompt_len_chars"] = len(prompt)
        debug_info["final_prompt_len_tokens"] = self._get_token_count(prompt)
        if self.debug:
            logger.debug(f"--- LLM Prompt Start ---\n{prompt}\n--- LLM Prompt End ---")

        # Generate Response
        logger.info("Generating response with LLM...")
        gen_start = time.time()
        text_response, raw_llm_output = "Error: LLM generation failed.", None
        try:
            # self.model is guaranteed non-None here due to check at start of method
            text_response, raw_llm_output = self.model.generate(
                prompt=prompt,
                max_tokens=max(150, self.context_length // 3),
                temperature=0.1,
                top_p=0.9,
                repeat_penalty=1.15,
                do_sample=True,
            )
        except Exception as e:
            logger.error(f"Error during LLM generation: {e}", exc_info=self.debug)
            raw_llm_output = {"error": str(e)}
        gen_time = time.time() - gen_start
        logger.info(
            f"LLM generation took {gen_time:.2f}s. Len: {len(text_response)} chars."
        )
        debug_info["llm_generation_time"] = f"{gen_time:.3f}s"
        try:
            debug_info["raw_llm_output"] = (
                json.dumps(raw_llm_output) if self.debug else "Disabled"
            )
        except TypeError:
            debug_info["raw_llm_output"] = (
                str(raw_llm_output) if self.debug else "Disabled"
            )

        # Format and Return
        formatted_response = f"## AI Answer\n{text_response}\n\n## Retrieved Context\n{retrieved_display}"
        return {
            "response": text_response,
            "debug_info": debug_info if debug_mode else {},
            "retrieved_context": retrieved_display,
            "formatted_response": formatted_response,
            "query_time_seconds": query_time,
            "generation_time_seconds": gen_time,
        }

    def close(self) -> None:
        """Unload models and release resources."""
        # Use temporary variables to avoid accessing potentially non-existent attributes
        model_to_close = getattr(self, "model", None)
        embedder_to_close = getattr(self, "embedder", None)
        vectordb_to_close = getattr(self, "vectordb", None)

        self.logger.info("Closing LLMSearch and its components...")
        if model_to_close:
            try:
                if hasattr(model_to_close, "unload"):
                    model_to_close.unload()
                del self.model
                self.model = None
                logger.debug("LLM closed.")
            except Exception as e:
                logger.error(f"Error closing LLM: {e}", exc_info=self.debug)
        if embedder_to_close:
            try:
                if hasattr(embedder_to_close, "close"):
                    embedder_to_close.close()
                del self.embedder
                self.embedder = None
                logger.debug("Embedder closed.")
            except Exception as e:
                logger.error(f"Error closing Embedder: {e}", exc_info=self.debug)
        if vectordb_to_close:
            try:
                if hasattr(vectordb_to_close, "close"):
                    vectordb_to_close.close()
                del self.vectordb
                self.vectordb = None
                logger.debug("VectorDB closed.")
            except Exception as e:
                logger.error(f"Error closing VectorDB: {e}", exc_info=self.debug)
        self.logger.info("LLMSearch closed.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
