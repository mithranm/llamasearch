# llamasearch/core/vectordb.py

import os
import numpy as np
import logging
import gc
import shutil
import json
from typing import Dict, Any, Optional, List
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from tqdm import tqdm

from .embedding import Embedder
from .bm25 import BM25Retriever
from .chunker import MarkdownChunker
from .knowledge_graph import KnowledgeGraph
from ..setup_utils import find_project_root

logger = logging.getLogger(__name__)


class VectorDB:
    """
    Retains advanced features:
      - advanced chunker
      - name-based expansions
      - BM25 + vector
      - knowledge graph building (dynamic, no staff hardcode)
      - reintroduce _get_embedding_count to fix missing method
    """

    def __init__(
        self,
        collection_name="documents",
        embedder: Optional[Embedder] = None,
        chunk_size=150,
        text_embedding_size=512,
        chunk_overlap=100,
        min_chunk_size=50,
        batch_size=8,
        similarity_threshold=0.15,
        max_chunks=5000,
        persist=False,
        storage_dir: Optional[str] = None,
        use_pca=False,
    ):
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

        if storage_dir:
            self.storage_dir = storage_dir
        else:
            project_root = find_project_root()
            suff = "pca" if use_pca else "no_pca"
            self.storage_dir = os.path.join(project_root, f"vector_db_{suff}")

        if not self.persist and os.path.exists(self.storage_dir):
            logger.info(f"Clearing vector database dir at {self.storage_dir}")
            for fn in os.listdir(self.storage_dir):
                fp = os.path.join(self.storage_dir, fn)
                if os.path.isfile(fp):
                    os.unlink(fp)
                else:
                    shutil.rmtree(fp)
            logger.info("Vector db dir cleared")

        os.makedirs(self.storage_dir, exist_ok=True)

        self.metadata_path = os.path.join(
            self.storage_dir, f"{collection_name}_meta.json"
        )
        self.embeddings_path = os.path.join(
            self.storage_dir, f"{collection_name}_embeddings.npy"
        )

        self.embedder = embedder or Embedder(
            batch_size=16, max_length=text_embedding_size
        )

        self.chunker = MarkdownChunker(
            chunk_size=chunk_size,
            text_embedding_size=text_embedding_size,
            min_chunk_size=min_chunk_size,
            max_chunks=max_chunks,
            batch_size=batch_size,
        )

        self.bm25 = BM25Retriever()

        self.documents: List[str] = []
        self.document_metadata: List[dict] = []

        # knowledge graph
        self.kg = KnowledgeGraph()

        self._load_metadata()
        logger.info(f"Initialized VectorDB with {len(self.documents)} documents")
        logger.info(f"Storage: {self.storage_dir}")
        logger.info(
            f"Embeddings at: {self.embeddings_path}, meta at: {self.metadata_path}"
        )
        logger.info(f"Persistence: {self.persist}, use_pca: {self.use_pca}")

        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            import subprocess

            subprocess.run(
                ["python", "-m", "spacy", "download", "en_core_web_sm"], check=True
            )
            self.nlp = spacy.load("en_core_web_sm")

    def _load_metadata(self):
        if os.path.exists(self.metadata_path):
            logger.info(f"Loading metadata from {self.metadata_path}")
            try:
                with open(self.metadata_path, "r") as f:
                    data = json.load(f)
                self.documents = data.get("documents", [])
                self.document_metadata = data.get("metadata", [])
                logger.info(f"Loaded {len(self.documents)} doc entries")
            except Exception as e:
                logger.error(f"Error loading meta: {e}")
                self.documents = []
                self.document_metadata = []

    def _save_metadata(self):
        logger.info(f"Saving metadata ({len(self.documents)} docs)")

        # Create temp file in the same directory as the target to avoid cross-device issues
        dest_dir = os.path.dirname(self.metadata_path)
        temp_filename = f"temp_{os.path.basename(self.metadata_path)}"
        temp_path = os.path.join(dest_dir, temp_filename)

        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                meta = {"documents": self.documents, "metadata": self.document_metadata}
                json.dump(meta, f)

            # Use os.replace for atomic operation (now safe since both files are on same filesystem)
            os.replace(temp_path, self.metadata_path)
            logger.info("Metadata saved.")
        except Exception as e:
            logger.error(f"Error saving meta: {e}")
            if os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except Exception as e2:
                    logger.error(f"Error deleting temp file: {e2}")
                    pass

    def _get_embedding_count(self) -> int:
        if os.path.exists(self.embeddings_path):
            try:
                arr = np.load(self.embeddings_path, mmap_mode="r")
                return arr.shape[0]
            except Exception as e:
                logger.error(f"Error loading embeddings: {e}")
        return 0

    def _is_document_processed(self, file_path: str) -> bool:
        if not self.document_metadata:
            return False
        disk_mtime = os.path.getmtime(file_path)
        for md in self.document_metadata:
            if md.get("source") == file_path:
                if abs(md.get("mtime", 0) - disk_mtime) < 0.0001:
                    return True
                else:
                    old_len = len(self.documents)
                    self.documents = [
                        d
                        for i, d in enumerate(self.documents)
                        if self.document_metadata[i].get("source") != file_path
                    ]
                    self.document_metadata = [
                        x
                        for x in self.document_metadata
                        if x.get("source") != file_path
                    ]
                    logger.info(
                        f"File changed, removed {old_len - len(self.documents)} old chunks"
                    )
                    return False
        return False

    def add_document(self, file_path: str) -> int:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        if not file_path.lower().endswith(".md"):
            raise ValueError("Only markdown (.md) files are supported")

        if self.persist and self._is_document_processed(file_path):
            logger.info(f"Document already processed: {file_path}")
            c = sum(1 for m in self.document_metadata if m.get("source") == file_path)
            return c

        doc_mtime = os.path.getmtime(file_path)
        base_meta = {
            "source": file_path,
            "filename": os.path.basename(file_path),
            "mtime": doc_mtime,
        }

        # read entire text => build knowledge graph from it
        full_text = ""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                full_text = f.read()
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            return 0

        # build knowledge graph from text
        self.kg.build_from_text(full_text, context_file=file_path)

        added_chunks = 0

        # Create temp file in the same directory as the target embeddings file
        dest_dir = os.path.dirname(self.embeddings_path)
        temp_embeddings_filename = (
            f"temp_embeddings_{os.path.basename(self.embeddings_path)}"
        )
        temp_path = os.path.join(dest_dir, temp_embeddings_filename)

        try:
            batch_idx = 0
            batches = list(
                self.chunker.process_file_in_batches(file_path, self.batch_size)
            )

            with tqdm(
                total=len(batches),
                desc=f"Processing {os.path.basename(file_path)}",
                unit="batch",
            ) as pbar:
                for batch in batches:
                    if not batch:
                        continue
                    chunk_texts = [b["chunk"] for b in batch]
                    emb_texts = [b["embedding_text"] for b in batch]
                    meta_list = []
                    for i, bdat in enumerate(batch):
                        met = dict(base_meta)
                        met["batch_idx"] = batch_idx
                        met["chunk_idx"] = i
                        meta_list.append(met)

                    logger.info(
                        f"Generating embeddings for batch {batch_idx+1} with {len(batch)} chunks"
                    )
                    embeddings = self.embedder.embed_batch(emb_texts)
                    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

                    if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                        old = np.load(temp_path)
                        old = np.ascontiguousarray(old, dtype=np.float32)
                        merged = np.vstack([old, embeddings])
                        np.save(temp_path, merged, allow_pickle=False)
                        del old, merged
                    else:
                        np.save(temp_path, embeddings, allow_pickle=False)

                    self.documents.extend(chunk_texts)
                    self.document_metadata.extend(meta_list)

                    for txt, mt in zip(chunk_texts, meta_list):
                        self.bm25.add_document(txt, mt)

                    added_chunks += len(batch)
                    batch_idx += 1
                    del chunk_texts, emb_texts, meta_list, embeddings
                    gc.collect()
                    pbar.update(1)
            if added_chunks > 0:
                self._merge_embeddings(temp_path)
                self._save_metadata()

            if os.path.exists(temp_path):
                os.unlink(temp_path)
                logger.info("Cleaned up temp file")

            return added_chunks

        except Exception as e:
            logger.error(f"Error adding doc {file_path}: {e}")
            if os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except Exception as e2:
                    logger.error(f"Error deleting temp file: {e2}")
                    pass
            raise

    def _merge_embeddings(self, temp_path: str):
        logger.info("Merging new embeddings with existing DB.")
        if (
            os.path.exists(self.embeddings_path)
            and os.path.getsize(self.embeddings_path) > 0
        ):
            try:
                arr_old = np.load(self.embeddings_path)
                arr_new = np.load(temp_path)
                merged = np.vstack([arr_old, arr_new])
                merged = np.ascontiguousarray(merged, dtype=np.float32)

                # Create another temp file in the same directory for the merged result
                dest_dir = os.path.dirname(self.embeddings_path)
                merged_temp_filename = (
                    f"merged_{os.path.basename(self.embeddings_path)}"
                )
                merged_temp_path = os.path.join(dest_dir, merged_temp_filename)

                # Save to temp file in the destination directory
                np.save(merged_temp_path, merged, allow_pickle=False)

                # Replace the original with the merged data
                os.replace(merged_temp_path, self.embeddings_path)

                logger.info("Embeddings merged.")
            except Exception as e:
                logger.error(f"Error merging embeddings: {e}")
                raise
        else:
            # Simply copy the temp file to the destination
            shutil.copy2(temp_path, self.embeddings_path)

    def query(self, query_text: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Hybrid search with adjacency expansions for name queries.
        """
        if not self.documents:
            logger.warning("No docs in DB.")
            return {
                "query": query_text,
                "ids": [],
                "documents": [],
                "metadatas": [],
                "scores": [],
                "distances": [],
            }

        # vector
        vres = self._vector_search(query_text, n_ret=n_results * 2)
        # bm25
        bmres = self.bm25.query(query_text, n_results * 2)
        combined = self._combine_results(vres, bmres, n_ret=n_results * 2)

        # check name expansions
        doc = self.nlp(query_text)
        persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
        if persons:
            logger.info(f"Detected name query {persons}")
            expanded = self._expand_name_context(combined, persons)
            # re-sort
            expanded_sorted = sorted(expanded, key=lambda x: x["score"], reverse=True)
            final = expanded_sorted[:n_results]
        else:
            final = sorted(combined, key=lambda x: x["score"], reverse=True)[:n_results]

        # build final dict
        out = {
            "query": query_text,
            "ids": [f"doc_{x['index']}" for x in final],
            "documents": [x["chunk"] for x in final],
            "metadatas": [x["metadata"] for x in final],
            "scores": [x["score"] for x in final],
            "distances": [1.0 - x["score"] for x in final],
        }
        return out

    def _vector_search(self, text: str, n_ret=10) -> Dict[str, Any]:
        if not text.strip():
            return {
                "query": text,
                "ids": [],
                "documents": [],
                "metadatas": [],
                "scores": [],
                "distances": [],
            }
        if len(text) > 2000:
            text = text[:2000]

        q_emb = self.embedder.embed_string(text)
        q_emb = np.ascontiguousarray(q_emb, dtype=np.float32)
        c = self._get_embedding_count()
        if c == 0:
            logger.warning("No embeddings.")
            return {
                "query": text,
                "ids": [],
                "documents": [],
                "metadatas": [],
                "scores": [],
                "distances": [],
            }
        sims = np.zeros(c, dtype=np.float32)
        arr = np.load(self.embeddings_path, mmap_mode="r")
        bsz = min(100, c)
        for i in range(0, c, bsz):
            end = min(i + bsz, c)
            chunk = np.ascontiguousarray(arr[i:end], dtype=np.float32)
            q_emb_2d = q_emb.reshape(1, -1) if q_emb.ndim == 1 else q_emb
            sub_sims = cosine_similarity(q_emb_2d, chunk)[0]
            sims[i:end] = sub_sims

        # threshold
        filt_idx = np.where(sims >= self.similarity_threshold)[0]
        if len(filt_idx) == 0 and len(text.split()) <= 3:
            filt_idx = np.where(sims >= 0.05)[0]

        sorted_idx = filt_idx[np.argsort(-sims[filt_idx])]
        sorted_idx = sorted_idx[:n_ret]

        out = {
            "query": text,
            "ids": [f"doc_{idx}" for idx in sorted_idx],
            "documents": [self.documents[idx] for idx in sorted_idx],
            "metadatas": [self.document_metadata[idx] for idx in sorted_idx],
            "scores": [float(sims[idx]) for idx in sorted_idx],
            "distances": [float(1.0 - sims[idx]) for idx in sorted_idx],
        }
        return out

    def _combine_results(
        self, vres: Dict[str, Any], bmres: Dict[str, Any], n_ret: int
    ) -> List[Dict[str, Any]]:
        # naive additive approach
        combined_scores = {}
        # from bmres
        bm_ids = bmres.get("ids", [])
        bm_sc = bmres.get("scores", [])
        for i, docid in enumerate(bm_ids):
            if i < len(bm_sc):
                idx = int(docid[4:]) if docid.startswith("doc_") else int(docid)
                combined_scores[idx] = combined_scores.get(idx, 0.0) + bm_sc[i]
        # from vres
        vec_ids = vres.get("ids", [])
        vec_sc = vres.get("scores", [])
        for i, docid in enumerate(vec_ids):
            if i < len(vec_sc):
                idx = int(docid[4:]) if docid.startswith("doc_") else int(docid)
                combined_scores[idx] = combined_scores.get(idx, 0.0) + vec_sc[i]

        sorted_idx = sorted(
            combined_scores.keys(), key=lambda k: combined_scores[k], reverse=True
        )
        top = sorted_idx[: n_ret * 2]
        out = []
        for idx in top:
            if 0 <= idx < len(self.documents):
                out.append(
                    {
                        "index": idx,
                        "chunk": self.documents[idx],
                        "metadata": self.document_metadata[idx],
                        "score": combined_scores[idx],
                    }
                )
        return out

    def _expand_name_context(
        self, combined_list: List[Dict[str, Any]], persons: List[str]
    ) -> List[Dict[str, Any]]:
        """
        If chunk references any person's name, also add neighboring chunks from the same doc
        to provide adjacency expansions.
        We do partial scoring for neighbors.
        """
        # Build doc->(index, chunk, meta) mapping
        doc_map = {}
        for i, md in enumerate(self.document_metadata):
            src = md.get("source", "")
            if src not in doc_map:
                doc_map[src] = []
            doc_map[src].append((i, self.documents[i], md))

        # sort each doc by batch_idx, chunk_idx
        for src in doc_map:
            doc_map[src].sort(
                key=lambda x: (x[2].get("batch_idx", 0), x[2].get("chunk_idx", 0))
            )

        new_list = []
        visited = set()
        for item in combined_list:
            idx = item["index"]
            chunk = item["chunk"]
            src = item["metadata"].get("source", "")
            low_chunk = chunk.lower()
            has_name = False
            for p in persons:
                if p.lower() in low_chunk:
                    has_name = True
                    break
            if has_name:
                # add item
                new_list.append(item)
                visited.add(idx)
                # find neighbors
                if src in doc_map:
                    doc_ch = doc_map[src]
                    pos = -1
                    for j, (rid, _, _) in enumerate(doc_ch):
                        if rid == idx:
                            pos = j
                            break
                    if pos != -1:
                        # we can add immediate neighbors
                        for offset in [-1, 1]:
                            nbpos = pos + offset
                            if 0 <= nbpos < len(doc_ch):
                                nb_idx = doc_ch[nbpos][0]
                                if nb_idx not in visited:
                                    new_list.append(
                                        {
                                            "index": nb_idx,
                                            "chunk": self.documents[nb_idx],
                                            "metadata": self.document_metadata[nb_idx],
                                            "score": item["score"] * 0.5,
                                        }
                                    )
                                    visited.add(nb_idx)
            else:
                # normal chunk
                new_list.append(item)
                visited.add(idx)

        # deduplicate by index, keep max score
        dedup = {}
        for x in new_list:
            idx = x["index"]
            if idx in dedup:
                if x["score"] > dedup[idx]["score"]:
                    dedup[idx] = x
            else:
                dedup[idx] = x
        return list(dedup.values())

    def close(self):
        self._save_metadata()
        self.documents = []
        self.document_metadata = []
        gc.collect()
        logger.info("VectorDB resources cleaned up")
