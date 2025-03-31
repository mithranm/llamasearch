# llamasearch/core/vectordb.py

import os
import numpy as np
import logging
import gc
import tempfile
import shutil
from typing import Dict, Any, Optional, Union, Tuple, List
from sklearn.metrics.pairwise import cosine_similarity
import json
import re
import spacy

from .embedding import Embedder
from .embedding_pca import PCAReducer
from .bm25 import BM25Retriever
from ..utils import find_project_root
from .chunker import MarkdownChunker

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

class VectorDB:
    """
    Vector + BM25 for RAG.
    - batch_size=4 for the embedder.
    - For name queries, show all matches, boosting chunks that contain the exact name text.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedder: Optional[Embedder] = None,
        chunk_size: int = 150,
        text_embedding_size: int = 384,
        chunk_overlap: int = 125,
        min_chunk_size: int = 50,
        batch_size: int = 8,
        similarity_threshold: float = 0.15,
        max_chunks: int = 5000,
        persist: bool = True,
        storage_dir: Optional[str] = None,
        use_pca: bool = False,
        pca_components: int = 128,
        pca_training_size: int = 1000,
    ):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.text_embedding_size = text_embedding_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.batch_size = batch_size
        self.similarity_threshold = similarity_threshold
        self.max_chunks = max_chunks
        self.persist = persist
        self.use_pca = use_pca
        self.pca_components = pca_components
        self.pca_training_size = pca_training_size

        if storage_dir:
            self.storage_dir = storage_dir
        else:
            project_root = find_project_root()
            suffix = "pca" if use_pca else "no_pca"
            self.storage_dir = os.path.join(project_root, f"vector_db_{suffix}")

        if not self.persist and os.path.exists(self.storage_dir):
            self.logger.info(f"Clearing vector database directory at {self.storage_dir}")
            try:
                for filename in os.listdir(self.storage_dir):
                    filepath = os.path.join(self.storage_dir, filename)
                    if os.path.isfile(filepath):
                        os.unlink(filepath)
                    elif os.path.isdir(filepath):
                        shutil.rmtree(filepath)
                self.logger.info("Vector database directory cleared successfully")
            except Exception as e:
                self.logger.error(f"Error clearing vector database: {e}")

        os.makedirs(self.storage_dir, exist_ok=True)

        self.metadata_path = os.path.join(self.storage_dir, f"{collection_name}_meta.json")
        self.embeddings_path = os.path.join(self.storage_dir, f"{collection_name}_embeddings.npy")

        # Use batch_size=4 for embedding
        self.embedder = embedder or Embedder(batch_size=4, max_length=text_embedding_size)

        self.pca_reducer = None
        if self.use_pca:
            self.pca_reducer = PCAReducer(
                n_components=self.pca_components,
                storage_dir=self.storage_dir,
                model_name=f"{collection_name}_pca",
            )
            if self.pca_reducer.load_model():
                self.logger.info(f"Loaded existing PCA model with {self.pca_components} components")

        # chunker
        self.chunker = MarkdownChunker(
            chunk_size=150,
            text_embedding_size=384,
            min_chunk_size=50,
            max_chunks=self.max_chunks,
            batch_size=1,
            code_context_window=2,
            include_section_headers=True,
            always_create_chunks=True,
        )

        # BM25
        self.bm25_retriever = BM25Retriever()

        # Load existing metadata
        self.documents = []
        self.document_metadata = []
        self._load_metadata()

        self.has_pca_training_data = (self._get_embedding_count() >= self.pca_training_size)
        self.document_chunk_map = {}
        self._build_document_chunk_map()

        self.logger.info(f"Initialized VectorDB with {len(self.documents)} documents")
        self.logger.info(f"Storage directory: {self.storage_dir}")
        self.logger.info(f"Embeddings stored at: {self.embeddings_path}")
        self.logger.info(f"Metadata stored at: {self.metadata_path}")
        self.logger.info(f"Persistence: {self.persist}")
        self.logger.info(f"PCA mode: {self.use_pca}")
        self.logger.info(f"Text embedding size: {self.text_embedding_size} tokens")

        try:
            self.nlp = spacy.load("en_core_web_sm")
            self.logger.info("Loaded spacy NER model")
        except OSError:
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
            self.nlp = spacy.load("en_core_web_sm")
            self.logger.info("Downloaded and loaded spacy NER model")

    def _build_document_chunk_map(self):
        self.document_chunk_map = {}
        for i, meta in enumerate(self.document_metadata):
            src = meta.get("source", "")
            if not src:
                continue
            if src not in self.document_chunk_map:
                self.document_chunk_map[src] = []
            info = {
                "index": i,
                "chunk_idx": meta.get("chunk_idx", -1),
                "batch_idx": meta.get("batch_idx", -1),
            }
            self.document_chunk_map[src].append(info)

        for src in self.document_chunk_map:
            self.document_chunk_map[src].sort(
                key=lambda x: (x["batch_idx"], x["chunk_idx"])
            )
        self.logger.info(f"Built document-chunk map for {len(self.document_chunk_map)} sources")

    def _load_metadata(self):
        if os.path.exists(self.metadata_path):
            self.logger.info(f"Loading metadata from {self.metadata_path}")
            try:
                with open(self.metadata_path, "r") as f:
                    data = json.load(f)
                self.documents = data.get("documents", [])
                self.document_metadata = data.get("metadata", [])
                self.logger.info(f"Loaded {len(self.documents)} document entries")
            except Exception as e:
                self.logger.error(f"Error loading metadata: {e}")
                self.documents = []
                self.document_metadata = []

    def _save_metadata(self):
        self.logger.info(f"Saving metadata ({len(self.documents)} documents)")
        tmp_path = None
        try:
            import tempfile
            with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as tf:
                tmp_path = tf.name
                meta = {
                    "documents": self.documents,
                    "metadata": self.document_metadata,
                }
                json.dump(meta, tf)
            os.replace(tmp_path, self.metadata_path)
            self.logger.info(f"Metadata saved to {self.metadata_path}")
        except Exception as e:
            self.logger.error(f"Error saving metadata: {e}")
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def _get_embedding_count(self) -> int:
        if os.path.exists(self.embeddings_path):
            try:
                arr = np.load(self.embeddings_path, mmap_mode="r")
                c = arr.shape[0]
                del arr
                return c
            except Exception as e:
                self.logger.error(f"Error checking embedding count: {e}")
        return 0

    def _is_document_processed(self, file_path: str) -> bool:
        if not self.document_metadata:
            return False
        mtime_disk = os.path.getmtime(file_path)
        for m in self.document_metadata:
            if m.get("source") == file_path:
                if abs(m.get("mtime", 0) - mtime_disk) < 0.0001:
                    return True
                else:
                    old_len = len(self.documents)
                    self.documents = [
                        d
                        for i, d in enumerate(self.documents)
                        if self.document_metadata[i].get("source") != file_path
                    ]
                    self.document_metadata = [
                        x for x in self.document_metadata
                        if x.get("source") != file_path
                    ]
                    self.logger.info(
                        f"File changed, removed {old_len - len(self.documents)} old chunks"
                    )
                    return False
        return False

    def add_document(self, file_path: str, metadata: Optional[Dict[str, Any]] = None) -> int:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        if not file_path.lower().endswith(".md"):
            raise ValueError("Only markdown (.md) files are supported")

        if self.persist and self._is_document_processed(file_path):
            self.logger.info(f"Document already processed: {file_path}")
            c = sum(m.get("source") == file_path for m in self.document_metadata)
            return c

        doc_mtime = os.path.getmtime(file_path)
        base_meta = {
            "source": file_path,
            "filename": os.path.basename(file_path),
            "mtime": doc_mtime,
            **(metadata or {}),
        }

        added_chunks = 0
        temp_path = None

        try:
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=".npy") as tf:
                temp_path = tf.name
                self.logger.info(f"Created temporary embeddings file: {temp_path}")

            batch_idx = 0
            for batch in self.chunker.process_file_in_batches(file_path, self.batch_size):
                if not batch:
                    continue
                emb_texts = [x["embedding_text"] for x in batch]
                full_chunks = [x["chunk"] for x in batch]
                metas = []
                for i, x in enumerate(batch):
                    meta_now = {
                        **base_meta,
                        **(x.get("metadata", {})),
                        "batch_idx": batch_idx,
                        "chunk_idx": i,
                    }
                    metas.append(meta_now)

                self.logger.info(f"Generating embeddings for batch {batch_idx+1} ({len(batch)} chunks)")
                embeddings = self.embedder.embed_batch(emb_texts)
                embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

                if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                    old = np.load(temp_path)
                    old = np.ascontiguousarray(old, dtype=np.float32)
                    combined = np.vstack([old, embeddings])
                    combined = np.ascontiguousarray(combined, dtype=np.float32)
                    np.save(temp_path, combined, allow_pickle=False)
                    del old, combined
                else:
                    np.save(temp_path, embeddings, allow_pickle=False)

                self.documents.extend(full_chunks)
                self.document_metadata.extend(metas)

                # Also add to BM25
                for doc_text, m in zip(full_chunks, metas):
                    self.bm25_retriever.add_document(doc_text, m)

                added_chunks += len(batch)
                batch_idx += 1
                del emb_texts, full_chunks, metas, embeddings
                gc.collect()

                self.logger.info(
                    f"Processed batch {batch_idx}: {len(batch)} chunks (total: {added_chunks})"
                )

            if added_chunks > 0:
                if self.use_pca:
                    self._train_or_update_pca(temp_path)
                else:
                    self._merge_embeddings_without_pca(temp_path)

            self._build_document_chunk_map()
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
                self.logger.info("Cleaned up temp file")

            return added_chunks

        except Exception as e:
            self.logger.error(f"Error adding document {file_path}: {e}")
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
            raise

    def _train_or_update_pca(self, temp_embeddings_path=None):
        if not self.use_pca or not self.pca_reducer:
            return
        existing = (os.path.exists(self.embeddings_path) and os.path.getsize(self.embeddings_path) > 0)
        if not self.has_pca_training_data and temp_embeddings_path and os.path.exists(temp_embeddings_path):
            if existing:
                old = np.load(self.embeddings_path)
                new = np.load(temp_embeddings_path)
                combo = np.vstack([old, new])
                combo = np.ascontiguousarray(combo, dtype=np.float32)
                if combo.shape[0] >= self.pca_training_size:
                    self.logger.info(f"Training PCA with {combo.shape[0]} embeddings")
                    self.pca_reducer.fit(combo)
                    self.has_pca_training_data = True
                    reduced = self.pca_reducer.transform(combo)
                    np.save(self.embeddings_path, reduced, allow_pickle=False)
                    del old, new, combo, reduced
                else:
                    np.save(self.embeddings_path, combo, allow_pickle=False)
                    del old, new, combo
                    self.logger.info("Combined embeddings saved, not enough for PCA yet")
            else:
                arr_new = np.load(temp_embeddings_path)
                if arr_new.shape[0] >= self.pca_training_size:
                    self.logger.info(f"Training PCA with {arr_new.shape[0]} embeddings")
                    self.pca_reducer.fit(arr_new)
                    self.has_pca_training_data = True
                    reduced = self.pca_reducer.transform(arr_new)
                    np.save(self.embeddings_path, reduced, allow_pickle=False)
                    del arr_new, reduced
                else:
                    np.save(self.embeddings_path, arr_new, allow_pickle=False)
                    del arr_new
                    self.logger.info("Saved embeddings, not enough for PCA training")
        elif self.has_pca_training_data and temp_embeddings_path and os.path.exists(temp_embeddings_path):
            arr_new = np.load(temp_embeddings_path)
            reduced_new = self.pca_reducer.transform(arr_new)
            if existing:
                arr_old = np.load(self.embeddings_path)
                combined = np.vstack([arr_old, reduced_new])
                combined = np.ascontiguousarray(combined, dtype=np.float32)
                np.save(self.embeddings_path, combined, allow_pickle=False)
                del arr_old, combined
            else:
                np.save(self.embeddings_path, reduced_new, allow_pickle=False)
            del arr_new, reduced_new
            self.logger.info("New embeddings transformed & appended via PCA")

    def _merge_embeddings_without_pca(self, temp_path: str):
        self.logger.info("Merging new embeddings with existing (no PCA).")
        if os.path.exists(self.embeddings_path) and os.path.getsize(self.embeddings_path) > 0:
            try:
                arr_old = np.load(self.embeddings_path)
                arr_new = np.load(temp_path)
                merged = np.vstack([arr_old, arr_new])
                merged = np.ascontiguousarray(merged, dtype=np.float32)
                np.save(self.embeddings_path, merged, allow_pickle=False)
                del arr_old, arr_new, merged
            except Exception as e:
                self.logger.error(f"Error merging embeddings: {e}")
                raise
        else:
            shutil.copy(temp_path, self.embeddings_path)

        self._save_metadata()
        self.logger.info("Merged embeddings + updated metadata")

    def close(self):
        try:
            self._save_metadata()
            self.documents = []
            self.document_metadata = []
            gc.collect()
            self.logger.info("VectorDB resources cleaned up")
        except Exception as e:
            self.logger.error(f"Error during close: {e}")

    def _detect_entities(self, text: str) -> Dict[str, List[str]]:
        doc = self.nlp(text)
        ent_map = {}
        for ent in doc.ents:
            if ent.label_ not in ent_map:
                ent_map[ent.label_] = []
            ent_map[ent.label_].append(ent.text)
        return ent_map

    def _expand_context_for_names(self, chunk_indices: List[int], n_context: int = 1) -> List[int]:
        expanded = set(chunk_indices)
        for idx in chunk_indices:
            if idx < 0 or idx >= len(self.document_metadata):
                continue
            meta = self.document_metadata[idx]
            src = meta.get("source", "")
            b_idx = meta.get("batch_idx", -1)
            c_idx = meta.get("chunk_idx", -1)
            if not src or b_idx < 0 or c_idx < 0:
                continue
            if src not in self.document_chunk_map:
                continue

            sc = self.document_chunk_map[src]
            pos = -1
            for i, info in enumerate(sc):
                if info["index"] == idx:
                    pos = i
                    break
            if pos == -1:
                continue

            start = max(0, pos - n_context)
            for i in range(start, pos):
                nb_idx = sc[i]["index"]
                if 0 <= nb_idx < len(self.documents):
                    expanded.add(nb_idx)

            end = min(len(sc), pos + n_context + 1)
            for i in range(pos+1, end):
                nb_idx = sc[i]["index"]
                if 0 <= nb_idx < len(self.documents):
                    expanded.add(nb_idx)

        return list(expanded)

    def _boost_name_chunks(
        self,
        combined_scores: Dict[int, float],
        name_query_text: str,
    ):
        """
        For name queries, if a chunk's text includes the exact name ignoring case,
        add a big bonus. This ensures those chunks appear first.
        """
        name_lower = name_query_text.lower()
        for doc_idx in combined_scores.keys():
            if 0 <= doc_idx < len(self.documents):
                chunk_text = self.documents[doc_idx].lower()
                # if it contains the person's name substring
                if name_lower in chunk_text:
                    combined_scores[doc_idx] += 999.0  # big boost

    def _combine_results(
        self,
        bm25_results: Dict[str, Any],
        vector_results: Dict[str, Any],
        bm25_weight: float,
        vector_weight: float,
        n_results: int
    ) -> Dict[str, Any]:
        bm25_scores = bm25_results.get("scores", [])
        vector_dists = vector_results.get("distances", [])
        is_name_query = vector_results.get("is_name_query", False)

        # Convert distances to similarity
        if len(vector_dists) > 0:
            sims = 1.0 - np.array(vector_dists)
            max_sim = np.max(sims) if sims.size > 0 else 1.0
            if max_sim > 0:
                sims /= max_sim
        else:
            sims = []

        combined = {}
        # Merge BM25 partial
        for i, doc_id in enumerate(bm25_results.get("ids", [])):
            if i < len(bm25_scores):
                score = bm25_scores[i]
                if doc_id.startswith("doc_"):
                    didx = int(doc_id[4:])
                else:
                    didx = int(doc_id)
                combined[didx] = bm25_weight * score

        # Merge vector partial
        for i, doc_id in enumerate(vector_results.get("ids", [])):
            if i < len(sims):
                if doc_id.startswith("doc_"):
                    didx = int(doc_id[4:])
                else:
                    didx = int(doc_id)
                vscore = vector_weight * sims[i]
                if didx in combined:
                    combined[didx] += vscore
                else:
                    combined[didx] = vscore

        # If name query, boost any chunk that literally has that name
        if is_name_query:
            # Heuristic: take the first PERSON entity from the query
            # Or if multiple, handle them all
            person_ents = self._detect_entities(vector_results.get("query", ""))
            if "PERSON" in person_ents:
                for nm in person_ents["PERSON"]:
                    # e.g. "matthew buban"
                    self._boost_name_chunks(combined, nm)

        # Now sort
        sorted_idxs = sorted(combined.keys(), key=lambda k: combined[k], reverse=True)

        # If name query => show *all* matched docs
        # Otherwise => only top n_results
        if is_name_query:
            top_idxs = sorted_idxs  # all
        else:
            top_idxs = sorted_idxs[:n_results]

        out_ids = []
        out_docs = []
        out_metas = []
        out_dists = []
        for idx in top_idxs:
            if 0 <= idx < len(self.documents):
                did = f"doc_{idx}"
                out_ids.append(did)
                out_docs.append(self.documents[idx])
                out_metas.append(self.document_metadata[idx])
                out_dists.append(1.0 - combined[idx])

        return {
            "query": vector_results.get("query", ""),
            "ids": out_ids,
            "documents": out_docs,
            "metadatas": out_metas,
            "distances": out_dists,
            "similarities": [combined[int(x[4:])] for x in out_ids],
            "is_name_query": is_name_query,
        }

    def query(self, query_text: str, n_results: int = 5) -> Dict[str, Any]:
        if not self.documents:
            self.logger.warning("No documents to search in VectorDB")
            return {
                "query": query_text,
                "ids": [],
                "documents": [],
                "metadatas": [],
                "distances": [],
            }
        if not query_text or len(query_text.strip()) < 2:
            self.logger.info("Query too short, returning empty results")
            return {
                "query": query_text,
                "ids": [],
                "documents": [],
                "metadatas": [],
                "distances": [],
            }
        if len(query_text) > 2000:
            query_text = query_text[:2000]
            self.logger.info("Truncated query to 2000 chars")

        eff_thresh = self._get_dynamic_threshold(query_text)
        self.logger.info(f"Using dynamic threshold: {eff_thresh}")

        self.logger.info("Generating embedding for query")
        q_emb = self.embedder.embed_string(query_text)
        q_emb = np.ascontiguousarray(q_emb, dtype=np.float32)

        if self.use_pca and self.pca_reducer and self.pca_reducer.is_fitted:
            self.logger.info("Applying PCA to query embedding")
            q_emb = self.pca_reducer.transform(q_emb.reshape(1, -1))[0]

        count = self._get_embedding_count()
        self.logger.info(f"Searching against {count} embeddings")

        if count == 0:
            self.logger.warning("No embeddings found, returning empty")
            return {
                "query": query_text,
                "ids": [],
                "documents": [],
                "metadatas": [],
                "distances": [],
            }

        sims = np.zeros(count, dtype=np.float32)
        batch_s = min(100, count)
        self.logger.info(f"Processing in batches of {batch_s}")
        arr = np.load(self.embeddings_path, mmap_mode="r")

        for i in range(0, count, batch_s):
            end = min(i + batch_s, count)
            sub = np.ascontiguousarray(arr[i:end], dtype=np.float32)
            sub_sims = cosine_similarity([q_emb], sub)[0]
            sims[i:end] = sub_sims
            del sub, sub_sims
            gc.collect()

        doc = self.nlp(query_text)
        person_ents = [ent for ent in doc.ents if ent.label_ == "PERSON"]
        is_name_query = (len(person_ents) > 0)

        if is_name_query:
            self.logger.info(f"Detected name query: {[p.text for p in person_ents]}")
            filtered_idx = np.arange(len(sims))
        else:
            filtered_idx = np.where(sims >= eff_thresh)[0]

        if filtered_idx.size == 0 and len(query_text.split()) <= 3:
            self.logger.info("No results above threshold; lowering to 0.05 for short query")
            filtered_idx = np.where(sims >= 0.05)[0]

        if filtered_idx.size == 0:
            self.logger.info(f"No results above threshold {eff_thresh}")
            return {
                "query": query_text,
                "ids": [],
                "documents": [],
                "metadatas": [],
                "distances": [],
                "is_name_query": is_name_query,
            }

        # sort desc by similarity
        sorted_idx = filtered_idx[np.argsort(-sims[filtered_idx])]
        # simply take the top n; if name query, the user might want the entire set
        # but let's keep it consistent: we won't deduplicate. 
        # We'll let the top-level combine in hybrid or user do so if needed.
        top_idx = sorted_idx[:n_results]

        result = {
            "query": query_text,
            "ids": [f"doc_{i}" for i in top_idx],
            "documents": [self.documents[i] for i in top_idx],
            "metadatas": [self.document_metadata[i] for i in top_idx],
            "distances": [1.0 - sims[i] for i in top_idx],
            "is_name_query": is_name_query,
        }

        del arr, sims
        gc.collect()
        self.logger.info(
            f"Found {len(result['ids'])} unique results above threshold {eff_thresh}"
        )
        return result

    def hybrid_query(self, query_text: str, n_results: int = 8) -> Dict[str, Any]:
        ents = self._detect_entities(query_text)
        has_name = ("PERSON" in ents)

        vec_res = self.query(query_text, n_results * 2)
        if has_name:
            bm25_lmt = n_results * 4
        else:
            bm25_lmt = n_results * 2
        bm25_res = self.bm25_retriever.query(query_text, bm25_lmt)

        if has_name:
            bm25_weight = 0.9
            vector_weight = 0.1
        elif any(t in ents for t in ["ORG", "GPE", "PRODUCT", "WORK_OF_ART"]):
            bm25_weight = 0.7
            vector_weight = 0.3
        else:
            bm25_weight = 0.3
            vector_weight = 0.7

        combined = self._combine_results(bm25_res, vec_res, bm25_weight, vector_weight, n_results)
        if has_name and combined.get("ids"):
            old_idxs = [int(x[4:]) for x in combined["ids"]]
            expanded = self._expand_context_for_names(old_idxs, n_context=2)
            if len(expanded) > len(old_idxs):
                self.logger.info(f"Expanded context from {len(old_idxs)} to {len(expanded)} chunks")
                old_ids = combined["ids"]
                old_dists = combined["distances"]
                dist_map = {old_ids[i]: old_dists[i] for i in range(len(old_ids))}
                new_ids, new_docs, new_metas, new_dists = [], [], [], []
                for idx in expanded:
                    did = f"doc_{idx}"
                    new_ids.append(did)
                    new_docs.append(self.documents[idx])
                    new_metas.append(self.document_metadata[idx])
                    if did in dist_map:
                        new_dists.append(dist_map[did])
                    else:
                        new_dists.append(0.5)
                combined["ids"] = new_ids
                combined["documents"] = new_docs
                combined["metadatas"] = new_metas
                combined["distances"] = new_dists
                combined["context_expanded"] = True

        return combined

    def _get_dynamic_threshold(self, text: str) -> float:
        doc = self.nlp(text)
        has_person = any(ent.label_ == "PERSON" for ent in doc.ents)
        if has_person:
            self.logger.info("Query has person name, using threshold=0.05")
            return 0.05
        words = text.split()
        if len(words) <= 3:
            return 0.15
        return self.similarity_threshold

    def _format_chunk_for_context(self, doc_text: str, meta: Dict[str, Any], sim: float) -> str:
        source = meta.get("source", "")
        filename = os.path.basename(source) if source else meta.get("filename", "unknown")
        return f"Source: {filename}\n\n{doc_text}\n\nRelevance: {sim*100:.2f}%\n"

    def get_context_for_query(
        self,
        query_text: str,
        n_results: int = 8,
        debug_mode: bool = False,
        return_chunks: bool = False,
    ) -> Union[str, Tuple[str, List], Tuple[str, Dict, List]]:
        results = self.hybrid_query(query_text, n_results)
        docs = results["documents"]
        if not docs:
            if debug_mode and return_chunks:
                return "", {"chunks": []}, []
            elif debug_mode:
                return "", {"chunks": []}
            elif return_chunks:
                return "", []
            else:
                return ""

        context_parts = []
        chunk_list = []
        for i, (dtx, md, dist, sim) in enumerate(
            zip(
                results["documents"],
                results["metadatas"],
                results["distances"],
                results.get("similarities", []),
            )
        ):
            chunk_str = self._format_chunk_for_context(dtx, md, sim)
            context_parts.append(chunk_str)
            chunk_list.append({
                "index": i,
                "text": dtx,
                "metadata": md,
                "distance": dist,
                "similarity": sim,
            })

        context_str = "\n---\n".join(context_parts)
        debug_info = {
            "query": query_text,
            "num_chunks": len(docs),
            "context_expanded": results.get("context_expanded", False),
            "chunks": chunk_list,
        }

        if debug_mode and return_chunks:
            return context_str, debug_info, chunk_list
        elif debug_mode:
            return context_str, debug_info
        elif return_chunks:
            return context_str, chunk_list
        else:
            return context_str
