import os
import re
import json
import shutil
import numpy as np
from typing import Dict, Any, Optional, List, Set
from pathlib import Path

from llamasearch.utils import setup_logging
from llamasearch.core.embedder import EnhancedEmbedder
from llamasearch.core.chunker import MarkdownChunker, HtmlChunker
from llamasearch.core.bm25 import BM25Retriever
from llamasearch.core.grapher import KnowledgeGraph

from sklearn.metrics.pairwise import cosine_similarity

logger = setup_logging(__name__)

class VectorDB:
    """
    VectorDB integrates:
      - Vector-based similarity (via embeddings)
      - BM25-based keyword retrieval
      - Knowledge graph expansions
    """

    def __init__(
        self,
        storage_dir: Path,
        collection_name: str,
        embedder: Optional[EnhancedEmbedder] = None,
        max_chunk_size: int = 512,
        text_embedding_size: int = 768,
        chunk_overlap: int = 50,
        min_chunk_size: int = 100,
        chunker_batch_size: int = 32,
        embedder_batch_size: int = 32,
        similarity_threshold: float = 0.2,  # not used for filtering vector results now
        max_results: int = 3,
        graph_weight: float = 0.3,
        bm25_weight: float = 0.3,
        vector_weight: float = 0.4,
        device: str = "cpu",
        enable_deduplication: bool = True,
        dedup_similarity_threshold: float = 0.8
    ):
        self.collection_name = collection_name
        self.max_chunk_size = max_chunk_size
        self.text_embedding_size = text_embedding_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.embedder_batch_size = embedder_batch_size
        self.chunker_batch_size = chunker_batch_size
        self.similarity_threshold = similarity_threshold
        self.max_results = max_results
        self.graph_weight = graph_weight
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        self.storage_dir = storage_dir
        self.device = device
        self.enable_deduplication = enable_deduplication
        self.dedup_similarity_threshold = dedup_similarity_threshold

        # Prepare directories
        collection_dir = os.path.join(str(storage_dir), collection_name)
        os.makedirs(collection_dir, exist_ok=True)
        self.vector_dir = os.path.join(collection_dir, "vector")
        self.bm25_dir = os.path.join(collection_dir, "bm25")
        self.kg_dir = os.path.join(collection_dir, "kg")
        os.makedirs(self.vector_dir, exist_ok=True)
        os.makedirs(self.bm25_dir, exist_ok=True)
        os.makedirs(self.kg_dir, exist_ok=True)

        # Metadata paths
        self.metadata_path = os.path.join(self.vector_dir, "meta.json")
        self.embeddings_path = os.path.join(self.vector_dir, "embeddings.npy")

        # Embedder
        self.embedder = embedder or EnhancedEmbedder(
            batch_size=self.embedder_batch_size,
            device=self.device
        )

        # Chunkers
        chunk_params = {
            "max_chunk_size": self.max_chunk_size,
            "min_chunk_size": self.min_chunk_size,
            "batch_size": self.chunker_batch_size,
            "overlap_size": self.chunk_overlap,
            "debug_output": False
        }
        self.markdown_chunker = MarkdownChunker(**chunk_params)
        self.html_chunker = HtmlChunker(**chunk_params)

        # BM25
        self.bm25 = BM25Retriever(storage_dir=self.bm25_dir)

        # Knowledge graph
        self.kg = KnowledgeGraph(storage_dir=self.kg_dir)

        # Internal lists for document chunks and metadata
        self.documents: List[str] = []
        self.document_metadata: List[Dict[str, Any]] = []
        self.processed_chunks: List[Set[str]] = []
        self._data_modified = False

        self._load_metadata()

    def _load_metadata(self) -> bool:
        """
        Load document metadata if it exists.
        """
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.documents = data.get("documents", [])
                self.document_metadata = data.get("metadata", [])
                self.processed_chunks = []
                for doc in self.documents:
                    self.processed_chunks.append(self._get_tokens(doc))
                if len(self.documents) != len(self.document_metadata):
                    logger.error("Mismatch in loaded docs vs metadata length.")
                    self.documents = []
                    self.document_metadata = []
                    self.processed_chunks = []
                    return False
                logger.info(f"Loaded {len(self.documents)} documents from metadata.")
                return True
            except Exception as e:
                logger.error(f"Error loading metadata: {e}")
                self.documents = []
                self.document_metadata = []
                self.processed_chunks = []
                return False
        else:
            logger.info(f"No metadata file at {self.metadata_path}. Starting fresh.")
            return False

    def _save_metadata(self) -> None:
        """
        Save document text and metadata.
        """
        logger.info(f"Saving metadata for {len(self.documents)} docs.")
        os.makedirs(os.path.dirname(self.metadata_path), exist_ok=True)
        temp_fn = "tmp_meta.json"
        temp_path = os.path.join(os.path.dirname(self.metadata_path), temp_fn)
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump({
                    "documents": self.documents,
                    "metadata": self.document_metadata
                }, f, ensure_ascii=False, indent=2)
            os.replace(temp_path, self.metadata_path)
            logger.info("Metadata saved.")
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def _get_tokens(self, text: str) -> Set[str]:
        return set(re.findall(r"\b\w+\b", text.lower()))

    def is_document_processed(self, file_path: str) -> bool:
        """
        Returns True if we already processed this file (matching mtime).
        """
        if not self.document_metadata:
            return False
        if not os.path.exists(file_path):
            return False
        mtime = os.path.getmtime(file_path)
        for md in self.document_metadata:
            if md.get("source") == file_path:
                if abs(md.get("mtime", 0) - mtime) < 1e-4:
                    return True
                else:
                    # remove old references if file has changed
                    self._remove_document(file_path)
                    return False
        return False

    def _remove_document(self, file_path: str) -> None:
        """
        Remove all chunks and metadata for the specified document file.
        """
        old_count = len(self.documents)
        indices = [i for i, m in enumerate(self.document_metadata) if m.get("source") == file_path]
        for idx in indices:
            if idx < len(self.documents):
                doc_id = f"doc_{idx}"
                try:
                    self.bm25.remove_document(doc_id)
                except Exception:
                    pass
        new_docs = []
        new_meta = []
        new_tokens = []
        for i in range(len(self.documents)):
            if i not in indices:
                new_docs.append(self.documents[i])
                new_meta.append(self.document_metadata[i])
                new_tokens.append(self.processed_chunks[i])
        self.documents = new_docs
        self.document_metadata = new_meta
        self.processed_chunks = new_tokens
        logger.info(f"Removed {old_count - len(self.documents)} chunks from {file_path}.")

    def add_document_chunks(self, file_path: str, chunks: List[Dict[str, Any]]) -> int:
        """
        Add all chunks for a single document, compute embeddings, update BM25,
        and store in the knowledge graph.
        """
        if not chunks:
            logger.warning(f"No chunks to add for {file_path}")
            return 0
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        if self.is_document_processed(file_path):
            logger.info(f"Document already processed: {file_path}")
            return sum(1 for m in self.document_metadata if m.get("source") == file_path)

        doc_mtime = os.path.getmtime(file_path)
        full_text = "\n\n".join(c["chunk"] for c in chunks if "chunk" in c)
        self.kg.add_document(file_path, full_text)

        filtered_chunks = []
        have_kept_one = False
        if self.enable_deduplication:
            for ch in chunks:
                text = ch.get("chunk", "")
                if not have_kept_one:
                    filtered_chunks.append(ch)
                    self.processed_chunks.append(self._get_tokens(text))
                    have_kept_one = True
                    continue
                if self._is_duplicate_chunk(text):
                    continue
                filtered_chunks.append(ch)
                self.processed_chunks.append(self._get_tokens(text))
        else:
            filtered_chunks = chunks

        doc_start_idx = len(self.documents)
        batch_size = self.chunker_batch_size or 32
        all_batches = [filtered_chunks[i : i + batch_size] for i in range(0, len(filtered_chunks), batch_size)]
        dest_dir = os.path.dirname(self.embeddings_path)
        os.makedirs(dest_dir, exist_ok=True)
        tmp_fn = "tmp_embed.npy"
        tmp_path = os.path.join(dest_dir, tmp_fn)
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

        added_chunks = 0
        base_meta = {
            "source": file_path,
            "filename": os.path.basename(file_path),
            "mtime": doc_mtime
        }
        try:
            for batch_idx, batch in enumerate(all_batches):
                texts = []
                metas = []
                embed_input = []
                for i, c in enumerate(batch):
                    ch_text = c.get("chunk", "")
                    texts.append(ch_text)
                    meta = dict(base_meta)
                    meta["batch_idx"] = batch_idx
                    meta["chunk_idx"] = i
                    if "metadata" in c:
                        for k, v in c["metadata"].items():
                            meta[k] = v
                    metas.append(meta)
                    embed_input.append(ch_text)
                emb_array = self.embedder.embed_batch(embed_input)
                emb_array = np.array(emb_array, dtype=np.float32)
                if os.path.exists(tmp_path):
                    prev_arr = np.load(tmp_path, allow_pickle=False)
                    combined = np.vstack([prev_arr, emb_array])
                    np.save(tmp_path, combined, allow_pickle=False)
                else:
                    np.save(tmp_path, emb_array, allow_pickle=False)
                self.documents.extend(texts)
                self.document_metadata.extend(metas)
                for idx_local, text_val in enumerate(texts):
                    doc_idx = doc_start_idx + added_chunks + idx_local
                    doc_id = f"doc_{doc_idx}"
                    self.bm25.add_document(text_val, doc_id)
                added_chunks += len(batch)
            if added_chunks > 0:
                self._merge_embeddings(tmp_path)
                self._save_metadata()
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            logger.info(f"Added {added_chunks} chunks from {file_path}")
            return added_chunks
        except Exception as e:
            logger.error(f"Error adding chunks from {file_path}: {e}")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

    def _merge_embeddings(self, temp_path: str) -> None:
        """
        Merge newly computed embeddings with existing embeddings file.
        """
        logger.info("Merging new embeddings with existing.")
        if os.path.exists(self.embeddings_path) and os.path.getsize(self.embeddings_path) > 0:
            try:
                old_arr = np.load(self.embeddings_path, allow_pickle=False)
                new_arr = np.load(temp_path, allow_pickle=False)
                merged = np.vstack([old_arr, new_arr])
                np.save(self.embeddings_path, merged, allow_pickle=False)
                logger.info("Embeddings merged.")
            except Exception as e:
                logger.error(f"Error merging embeddings: {e}")
                raise
        else:
            shutil.copy2(temp_path, self.embeddings_path)
            logger.info("Created new embeddings file.")

    def _is_duplicate_chunk(self, text: str) -> bool:
        """
        Deduplicate similar chunks using Jaccard similarity over recent chunks.
        """
        chunk_tokens = self._get_tokens(text)
        if not chunk_tokens:
            return False
        recent_limit = 100
        for existing_tokens in self.processed_chunks[-recent_limit:]:
            intersection = len(chunk_tokens & existing_tokens)
            union = len(chunk_tokens | existing_tokens)
            if union > 0:
                sim = intersection / union
                if sim > self.dedup_similarity_threshold:
                    return True
        return False

    def _vector_search(self, query_text: str, n_ret: int) -> Dict[str, Any]:
        """
        Vector-based search using cosine similarity to embeddings.
        """
        empty_res = {"query": query_text, "ids": [], "scores": [], "score_details": []}
        query_text = query_text.strip()
        if not query_text:
            return empty_res
        if not os.path.exists(self.embeddings_path):
            logger.warning(f"No embeddings found at {self.embeddings_path}")
            return empty_res
        try:
            all_embeddings = np.load(self.embeddings_path, allow_pickle=False)
            all_embeddings = np.array(all_embeddings, dtype=np.float32)
            q_emb = self.embedder.embed_string(query_text)
            scores = cosine_similarity(q_emb.reshape(1, -1), all_embeddings)[0]
            top_idx = np.argsort(scores)[::-1][:n_ret]
            doc_ids = [f"doc_{i}" for i in top_idx]
            doc_scores = [float(scores[i]) for i in top_idx]
            details = [{"index": i, "cos_sim": float(scores[i])} for i in top_idx]
            return {"query": query_text, "ids": doc_ids, "scores": doc_scores, "score_details": details}
        except Exception as e:
            logger.error(f"Vector search error: {e}")
            return empty_res

    def _bm25_search(self, query_text: str, n_ret: int) -> Dict[str, Any]:
        """
        BM25-based search.
        """
        empty_res = {"query": query_text, "ids": [], "scores": [], "score_details": []}
        if not query_text.strip():
            return empty_res
        try:
            results = self.bm25.query(query_text, n_results=n_ret)
            doc_ids = results.get("ids", [])
            doc_scr = results.get("scores", [])
            if not doc_ids:
                return empty_res
            details = []
            for d_id, sc in zip(doc_ids, doc_scr):
                try:
                    idx = int(d_id.split("_")[1])
                except Exception:
                    idx = -1
                if idx >= 0:
                    details.append({"index": idx, "bm25": sc})
            return {"query": query_text, "ids": doc_ids, "scores": doc_scr, "score_details": details}
        except Exception as e:
            logger.error(f"BM25 search error: {e}")
            return empty_res

    def _kg_search(self, query_text: str, n_ret: int) -> Dict[str, Any]:
        """
        Use the knowledge graph 'search' method to find relevant docs.
        """
        empty_res = {"query": query_text, "ids": [], "scores": [], "score_details": []}
        if not query_text.strip():
            return empty_res
        query_entities = [query_text]
        kg_results = self.kg.search(query_entities, limit=n_ret)
        if not kg_results:
            return empty_res
        doc_scores_map = {}
        for ent_base, doc_dict in kg_results.items():
            for doc_src, sc in doc_dict.items():
                if doc_src not in doc_scores_map:
                    doc_scores_map[doc_src] = sc
                else:
                    doc_scores_map[doc_src] = max(doc_scores_map[doc_src], sc)
        src_to_indices = {}
        for i, md in enumerate(self.document_metadata):
            source = md.get("source", "")
            src_to_indices.setdefault(source, []).append(i)
        doc_ids = []
        doc_scores = []
        details = []
        sorted_items = sorted(doc_scores_map.items(), key=lambda x: x[1], reverse=True)
        for (doc_src, sc) in sorted_items:
            if doc_src not in src_to_indices:
                continue
            for idx in src_to_indices[doc_src]:
                doc_ids.append(f"doc_{idx}")
                doc_scores.append(sc)
                details.append({"index": idx, "kg_score": sc})
        return {"query": query_text, "ids": doc_ids[:n_ret], "scores": doc_scores[:n_ret], "score_details": details[:n_ret]}

    def vectordb_query(self, query_text: str, max_out: int = 3) -> Dict[str, Any]:
        """
        Combines vector, BM25, and knowledge graph search results.
        Always incorporates KG search results to expand context so that up to max_out chunks are returned.
        """
        over = max_out * 2
        vec_res = self._vector_search(query_text, over)
        bm_res = self._bm25_search(query_text, over)
        combined_scores = {}

        def get_idx(doc_id: str) -> int:
            try:
                return int(doc_id.split("_")[1])
            except Exception:
                return -1

        # Incorporate vector search scores.
        for d_id, sc in zip(vec_res["ids"], vec_res["scores"]):
            idx = get_idx(d_id)
            if idx < 0:
                continue
            source = self.document_metadata[idx].get("source", "") if idx < len(self.document_metadata) else ""
            if idx not in combined_scores:
                combined_scores[idx] = {"vscore": 0.0, "bm25": 0.0, "source": source}
            combined_scores[idx]["vscore"] = max(combined_scores[idx]["vscore"], sc)

        # Incorporate BM25 search scores.
        for d_id, sc in zip(bm_res["ids"], bm_res["scores"]):
            idx = get_idx(d_id)
            if idx < 0:
                continue
            source = self.document_metadata[idx].get("source", "") if idx < len(self.document_metadata) else ""
            if idx not in combined_scores:
                combined_scores[idx] = {"vscore": 0.0, "bm25": 0.0, "source": source}
            combined_scores[idx]["bm25"] = max(combined_scores[idx]["bm25"], sc)

        # Compute combined score using weighted vector and BM25 scores.
        for idx, data in combined_scores.items():
            data["combined"] = data["vscore"] * self.vector_weight + data["bm25"] * self.bm25_weight

        # Boost for exact keyword matches: if a document's text contains the query exactly (case-insensitively),
        # set its combined score to at least 1.0.
        exact_pattern = re.compile(r'\b' + re.escape(query_text.lower()) + r'\b')
        for idx, doc_text in enumerate(self.documents):
            if exact_pattern.search(doc_text.lower()):
                if idx in combined_scores:
                    combined_scores[idx]["combined"] = max(combined_scores[idx]["combined"], 1.0)
                else:
                    source = self.document_metadata[idx].get("source", "") if idx < len(self.document_metadata) else ""
                    combined_scores[idx] = {"vscore": 0.0, "bm25": 0.0, "source": source, "combined": 1.0}

        # Incorporate knowledge graph search results for context expansion.
        # This "double heuristic" uses both entity name/context similarity (done inside self.kg.search)
        # and our configured graph_weight.
        kg_res = self._kg_search(query_text, over)
        for d_id, sc in zip(kg_res["ids"], kg_res["scores"]):
            idx = get_idx(d_id)
            if idx < 0:
                continue
            source = self.document_metadata[idx].get("source", "") if idx < len(self.document_metadata) else ""
            kg_score = sc * self.graph_weight
            if idx in combined_scores:
                combined_scores[idx]["combined"] = max(combined_scores[idx]["combined"], kg_score)
            else:
                combined_scores[idx] = {"vscore": 0.0, "bm25": 0.0, "source": source, "combined": kg_score}

        # Now sort all candidates by their combined score in descending order.
        sorted_candidates = sorted(combined_scores.items(), key=lambda x: x[1]["combined"], reverse=True)
        # Select the top max_out candidates.
        selected = sorted_candidates[:max_out]

        final_ids = []
        final_scores = []
        final_details = []
        for idx, scd in selected:
            final_ids.append(f"doc_{idx}")
            final_scores.append(scd["combined"])
            detail = {
                "index": idx,
                "vector_score": scd.get("vscore", 0.0),
                "bm25_score": scd.get("bm25", 0.0),
                "kg_score": scd.get("kg", 0.0),
                "combined_score": scd["combined"]
            }
            final_details.append(detail)

        # Gather the actual chunk texts and their metadata.
        retrieved_docs = [self.documents[int(doc_id.split('_')[1])] for doc_id in final_ids]
        retrieved_metadata = [self.document_metadata[int(doc_id.split('_')[1])] for doc_id in final_ids]

        return {
            "query": query_text,
            "ids": final_ids,
            "scores": final_scores,
            "score_details": final_details,
            "documents": retrieved_docs,
            "metadatas": retrieved_metadata
        }



    def close(self) -> None:
        """
        Close any resources if necessary.
        """
        if hasattr(self.embedder, "close"):
            self.embedder.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
