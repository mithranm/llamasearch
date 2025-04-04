# llamasearch/core/vectordb.py

import os
import numpy as np
import logging
import gc
import shutil
import json
import networkx as nx
from collections import defaultdict
from pathlib import Path

from typing import Dict, Any, Optional, List, cast
from sklearn.metrics.pairwise import cosine_similarity

from .embedder import EnhancedEmbedder
from .bm25 import BM25Retriever
from .chunker import MarkdownChunker, HtmlChunker
from .knowledge_graph import KnowledgeGraph
from ..setup_utils import find_project_root

logger = logging.getLogger(__name__)


class VectorDB:
    """
    Enhanced VectorDB that integrates semantic chunking and knowledge graph-based retrieval.
    
    Features:
      - Triple heuristic: BM25 + vector similarity + graph entity relationships
      - Knowledge graph integration for entity-based expansion
      - Graph-based distance rankings
      - BM25 + vector + graph hybrid search
    """

    def __init__(
        self,
        collection_name="documents",
        embedder: Optional[EnhancedEmbedder] = None,
        chunk_size=500,
        text_embedding_size=512,
        chunk_overlap=100,
        min_chunk_size=200,
        chunker_batch_size: Optional[int] = None,
        embedder_batch_size: Optional[int] = None,
        similarity_threshold=0.15,
        max_chunks=5000,
        persist=False,
        storage_dir: Optional[str] = None,
        use_pca=False,
        graph_weight=0.3,
        bm25_weight=0.35,
        vector_weight=0.35,
    ):
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.text_embedding_size = text_embedding_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.embedder_batch_size = embedder_batch_size
        self.chunker_batch_size = chunker_batch_size
        self.similarity_threshold = similarity_threshold
        self.max_chunks = max_chunks
        self.persist = persist
        self.use_pca = use_pca
        
        # Ranking weights for the triple heuristic
        self.graph_weight = graph_weight
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight

        if storage_dir:
            self.storage_dir = storage_dir
        else:
            project_root = find_project_root()
            suff = "pca" if use_pca else "no_pca"
            self.storage_dir = os.path.join(project_root, f"vector_db_{suff}")

        # Possibly clear old data if not persisting
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
        self.kg_path = os.path.join(
            self.storage_dir, f"{collection_name}_knowledge_graph.json"
        )

        # Setup embedder
        self.embedder = embedder or EnhancedEmbedder(
            batch_size=embedder_batch_size,
            max_length=text_embedding_size,
        )

        # Setup chunkers
        chunker_params = {
            'chunk_size': chunk_size,
            'text_embedding_size': text_embedding_size,
            'min_chunk_size': min_chunk_size,
            'max_chunks': max_chunks,
            'batch_size': chunker_batch_size,
            'overlap_size': chunk_overlap,
            'semantic_headers_only': True,
            'min_section_length': 100,
        }
        
        self.markdown_chunker = MarkdownChunker(**chunker_params)
        self.html_chunker = HtmlChunker(**chunker_params)
        self.chunker = self.markdown_chunker  # For backward compatibility

        self.bm25 = BM25Retriever()

        self.documents: List[str] = []
        self.document_metadata: List[Dict[str, Any]] = []

        # Knowledge graph and entity index
        self.kg = KnowledgeGraph()
        self.entity_to_chunks = defaultdict(set)
        self.chunk_to_entities = defaultdict(set)
        self.graph = nx.Graph()

        # Load any existing data
        self._load_metadata()
        self._load_knowledge_graph()
        self._build_entity_indices()

        logger.info(f"Initialized VectorDB with {len(self.documents)} documents")
        logger.info(f"Storage dir: {self.storage_dir}")
        logger.info(f"Knowledge graph has {len(self.kg.entities)} entities")
        logger.info(f"Embeddings at: {self.embeddings_path}")
        logger.info(f"Persistence: {self.persist}, use_pca: {self.use_pca}")
        logger.info(f"Search weights => vector={self.vector_weight}, bm25={self.bm25_weight}, graph={self.graph_weight}")


    def _load_metadata(self):
        if os.path.exists(self.metadata_path):
            logger.info(f"Loading metadata from {self.metadata_path}")
            try:
                with open(self.metadata_path, "r") as f:
                    data = json.load(f)
                self.documents = data.get("documents", [])
                self.document_metadata = data.get("metadata", [])
                # Rebuild BM25 index from loaded documents
                for doc, meta in zip(self.documents, self.document_metadata):
                    self.bm25.add_document(doc, meta)
                logger.info(f"Loaded {len(self.documents)} doc entries from metadata")
            except Exception as e:
                logger.error(f"Error loading meta: {e}")
                self.documents = []
                self.document_metadata = []

    def _save_metadata(self):
        logger.info(f"Saving metadata for {len(self.documents)} documents.")
        temp_fn = f"temp_{os.path.basename(self.metadata_path)}"
        temp_path = os.path.join(os.path.dirname(self.metadata_path), temp_fn)
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                meta = {
                    "documents": self.documents,
                    "metadata": self.document_metadata
                }
                json.dump(meta, f)
            os.replace(temp_path, self.metadata_path)
            logger.info("Metadata saved.")
        except Exception as e:
            logger.error(f"Error saving meta: {e}")
            if os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass

    def _load_knowledge_graph(self):
        if os.path.exists(self.kg_path):
            logger.info(f"Loading knowledge graph from {self.kg_path}")
            try:
                with open(self.kg_path, "r", encoding="utf-8") as f:
                    self.kg.entities = json.load(f)
                logger.info(f"KG loaded with {len(self.kg.entities)} entities.")
            except Exception as e:
                logger.error(f"Error loading KG: {e}")
                self.kg.entities = {}

    def _save_knowledge_graph(self):
        logger.info(f"Saving knowledge graph with {len(self.kg.entities)} entities.")
        temp_fn = f"temp_{os.path.basename(self.kg_path)}"
        temp_path = os.path.join(os.path.dirname(self.kg_path), temp_fn)
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(self.kg.entities, f, ensure_ascii=False)
            os.replace(temp_path, self.kg_path)

            # Also store a copy in data/ for convenience
            data_dir = os.path.join(find_project_root(), "data")
            os.makedirs(data_dir, exist_ok=True)
            data_path = os.path.join(data_dir, f"{self.collection_name}_knowledge_graph.json")
            with open(data_path, "w", encoding="utf-8") as f:
                json.dump(self.kg.entities, f, ensure_ascii=False)

            logger.info(f"KG saved to {self.kg_path} and {data_path}")
        except Exception as e:
            logger.error(f"Error saving KG: {e}")
            if os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass

    def _build_entity_indices(self):
        logger.info("Building entity→chunk and chunk→entity indices + relationship graph")
        self.entity_to_chunks.clear()
        self.chunk_to_entities.clear()
        self.graph.clear()

        # 1) For each chunk
        for i, (chunk_text, meta) in enumerate(zip(self.documents, self.document_metadata)):
            if "entities" not in meta:
                meta["entities"] = self._extract_entities_from_chunk(chunk_text)
            for ent in meta["entities"]:
                etext = ent["text"]
                self.entity_to_chunks[etext].add(i)
                self.chunk_to_entities[i].add(etext)
                if not self.graph.has_node(etext):
                    self.graph.add_node(etext, label=ent["label"])

        # 2) build edges from KG co_mentions
        for entity, info in self.kg.entities.items():
            co = info.get("co_mentions", {})
            for related_ent, co_list in co.items():
                weight = len(co_list)
                if self.graph.has_edge(entity, related_ent):
                    self.graph[entity][related_ent]["weight"] += weight  # type: ignore[operator]
                else:
                    self.graph.add_edge(entity, related_ent, weight=weight)

        logger.info(f"entity_to_chunks: {len(self.entity_to_chunks)} entity keys")
        logger.info(f"chunk_to_entities: {len(self.chunk_to_entities)} chunks total")
        logger.info(f"Graph: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")

    def _extract_entities_from_chunk(self, text: str) -> List[Dict[str, Any]]:
        """Use spaCy to extract named entities from chunk text."""
        entities = []
        try:
            if not text or len(text) < 5:
                return entities
            # If we have spaCy, use it
            if not self.kg.nlp:
                return entities
            doc = self.kg.nlp(text)
            for ent in doc.ents:
                if ent.text.strip():
                    # Use proper dict creation instead of += for merging
                    entity_data: Dict[str, Any] = {"text": ent.text, "label": ent.label_}
                    entities.append(entity_data)
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
        return entities

    def _get_embedding_count(self) -> int:
        if os.path.exists(self.embeddings_path):
            try:
                arr = np.load(self.embeddings_path, mmap_mode="r")
                return arr.shape[0]
            except Exception as e:
                logger.error(f"Error loading embeddings array: {e}")
        return 0

    def _merge_embeddings(self, temp_path: str):
        logger.info("Merging newly created embeddings with existing DB.")
        if (
            os.path.exists(self.embeddings_path)
            and os.path.getsize(self.embeddings_path) > 0
        ):
            try:
                arr_old = np.load(self.embeddings_path)
                arr_new = np.load(temp_path)
                merged = np.vstack([arr_old, arr_new])
                merged = np.ascontiguousarray(merged, dtype=np.float32)
                merged_temp = os.path.join(
                    os.path.dirname(self.embeddings_path),
                    f"merged_{os.path.basename(self.embeddings_path)}"
                )
                np.save(merged_temp, merged, allow_pickle=False)
                os.replace(merged_temp, self.embeddings_path)
                logger.info("Embeddings merged successfully.")
            except Exception as e:
                logger.error(f"Error merging embeddings: {e}")
                raise
        else:
            shutil.copy2(temp_path, self.embeddings_path)

    def _is_document_processed(self, file_path: str) -> bool:
        """Check if a file was previously processed (by mtime)."""
        if not self.document_metadata:
            return False
        disk_mtime = os.path.getmtime(file_path)
        for md in self.document_metadata:
            if md.get("source") == file_path:
                if abs(md.get("mtime", 0) - disk_mtime) < 0.0001:
                    return True
                else:
                    # file changed => remove old
                    old_len = len(self.documents)
                    self.documents = [
                        d for i, d in enumerate(self.documents)
                        if self.document_metadata[i].get("source") != file_path
                    ]
                    self.document_metadata = [
                        x for x in self.document_metadata
                        if x.get("source") != file_path
                    ]
                    logger.info(f"File changed => removed {old_len - len(self.documents)} old chunks")
                    return False
        return False

    def add_document(self, file_path: str) -> int:
        """Index a single .md file by chunking + embedding each chunk."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        ext = Path(file_path).suffix.lower()
        if ext not in [".md", ".html", ".htm"]:
            raise ValueError("Only markdown and HTML files are supported")
            
        chunker = self.markdown_chunker if ext == ".md" else self.html_chunker

        if self.persist and self._is_document_processed(file_path):
            logger.info(f"Document is already processed: {file_path}")
            c = sum(1 for m in self.document_metadata if m.get("source") == file_path)
            return c

        doc_mtime = os.path.getmtime(file_path)
        base_meta = {
            "source": file_path,
            "filename": os.path.basename(file_path),
            "mtime": doc_mtime,
        }
        with open(file_path, "r", encoding="utf-8") as f:
            full_text = f.read()

        # Build knowledge graph from full text
        self.kg.build_from_text(full_text, context_file=file_path)
        
        # Use appropriate chunker based on file type
        chunker = self.markdown_chunker if ext == ".md" else self.html_chunker

        added_chunks = 0
        dest_dir = os.path.dirname(self.embeddings_path)
        temp_fn = f"temp_embeddings_{os.path.basename(self.embeddings_path)}"
        temp_path = os.path.join(dest_dir, temp_fn)

        try:
            all_batches = list(chunker.process_file_in_batches(file_path, self.chunker_batch_size))
            total_chunks = sum(len(b) for b in all_batches)
            logger.info(f"{os.path.basename(file_path)} => {total_chunks} chunks in {len(all_batches)} batches")

            for batch_idx, batch in enumerate(all_batches):
                if not batch:
                    continue
                texts = [x["chunk"] for x in batch]
                emb_texts = [x["embedding_text"] for x in batch]
                meta_list = []
                for i, bdat in enumerate(batch):
                    met = dict(base_meta)
                    met["batch_idx"] = batch_idx
                    met["chunk_idx"] = i
                    if "metadata" in bdat:
                        for k, v in bdat["metadata"].items():
                            met[k] = v
                    meta_list.append(met)

                embeddings = self.embedder.embed_batch(emb_texts)
                embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

                if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                    old = np.load(temp_path)
                    old = np.ascontiguousarray(old, dtype=np.float32)
                    merged = np.vstack([old, embeddings])
                    merged = np.ascontiguousarray(merged, dtype=np.float32)
                    np.save(temp_path, merged, allow_pickle=False)
                    del old, merged
                else:
                    np.save(temp_path, embeddings, allow_pickle=False)

                self.documents.extend(texts)
                self.document_metadata.extend(meta_list)
                for t, m in zip(texts, meta_list):
                    self.bm25.add_document(t, m)

                added_chunks += len(batch)

            if added_chunks > 0:
                self._merge_embeddings(temp_path)
                self._save_metadata()
                self._save_knowledge_graph()
                self._build_entity_indices()

            if os.path.exists(temp_path):
                os.unlink(temp_path)
            logger.info(f"Done indexing {file_path}: {added_chunks} chunks.")
            return added_chunks

        except Exception as e:
            logger.error(f"Error adding doc {file_path}: {e}")
            if os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass
            raise

    def add_document_chunks(self, file_path: str, chunks: List[Dict[str, Any]]) -> int:
        """If you already have chunk dicts from somewhere else, feed them in here."""
        if not chunks:
            logger.warning(f"No chunks provided for {file_path}")
            return 0
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        if self.persist and self._is_document_processed(file_path):
            logger.info(f"Doc is already processed: {file_path}")
            c = sum(1 for m in self.document_metadata if m.get("source") == file_path)
            return c

        doc_mtime = os.path.getmtime(file_path)
        base_meta = {
            "source": file_path,
            "filename": os.path.basename(file_path),
            "mtime": doc_mtime,
        }
        # Build knowledge graph from combined chunk text
        full_text = "\n\n".join(c["chunk"] for c in chunks)
        self.kg.build_from_text(full_text, context_file=file_path)

        added_chunks = 0
        dest_dir = os.path.dirname(self.embeddings_path)
        temp_fn = f"temp_embeddings_{os.path.basename(self.embeddings_path)}"
        temp_path = os.path.join(dest_dir, temp_fn)

        try:
            emb_texts = [c.get("embedding_text") or c["chunk"] for c in chunks]
            embeddings = self.embedder.embed_batch(emb_texts)
            embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

            if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                old = np.load(temp_path)
                old = np.ascontiguousarray(old, dtype=np.float32)
                merged = np.vstack([old, embeddings])
                merged = np.ascontiguousarray(merged, dtype=np.float32)
                np.save(temp_path, merged, allow_pickle=False)
                del old, merged
            else:
                np.save(temp_path, embeddings, allow_pickle=False)

            chunk_texts = [chunk["chunk"] for chunk in chunks]
            meta_list = []
            for i, cdat in enumerate(chunks):
                met = dict(base_meta)
                met["batch_idx"] = 0
                met["chunk_idx"] = i
                if "metadata" in cdat:
                    for k, v in cdat["metadata"].items():
                        met[k] = v
                meta_list.append(met)

            self.documents.extend(chunk_texts)
            self.document_metadata.extend(meta_list)
            for txt, m in zip(chunk_texts, meta_list):
                self.bm25.add_document(txt, m)

            added_chunks += len(chunks)

            self._merge_embeddings(temp_path)
            self._save_metadata()
            self._save_knowledge_graph()
            self._build_entity_indices()

            logger.info(f"Added {len(chunks)} chunks from {os.path.basename(file_path)}")
            return len(chunks)

        except Exception as e:
            logger.error(f"Error adding chunks from {file_path}: {e}")
            if os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass
            raise

    def query(self, query_text: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Hybrid search with graph-based ranking:
          - vector similarity
          - BM25
          - knowledge graph entity relationships
        Return top n_results, ensuring they are not forcibly deduplicated out.
        """
        if not self.documents:
            logger.warning("No docs in DB; returning empty results.")
            return {
                "query": query_text,
                "ids": [],
                "documents": [],
                "metadatas": [],
                "scores": [],
                "distances": [],
                "entities": [],
            }

        # Extract query entities
        query_entities = self._extract_entities_from_chunk(query_text)
        entity_names = [e["text"] for e in query_entities]

        # vector search
        vec_res = self._vector_search(query_text, n_ret=n_results * 2)
        
        # Handle empty vector search results safely
        vec_idx = []
        vec_scores = {}
        if vec_res and isinstance(vec_res, dict) and vec_res.get("ids"):
            try:
                vec_idx = [int(iid[4:]) for iid in vec_res["ids"]]
                vec_scores = {idx: s for idx, s in zip(vec_idx, vec_res.get("scores", []))}
            except (IndexError, ValueError) as e:
                logger.warning(f"Error parsing vector search IDs: {e}")
                vec_idx = []
                vec_scores = {}

        # BM25
        bm_idx = []
        bm_scores = {}
        try:
            bm_res = self.bm25.query(query_text, n_results * 2)
            if bm_res and isinstance(bm_res, dict) and bm_res.get("ids"):
                bm_idx = [int(iid[4:]) for iid in bm_res["ids"]]
                bm_scores = {idx: s for idx, s in zip(bm_idx, bm_res.get("scores", []))}
        except Exception as e:
            logger.warning(f"BM25 error: {e}")

        # Graph-based
        graph_scores = {}
        if entity_names:
            graph_scores = self._graph_search(entity_names, n_ret=n_results * 2)

        combined_scores = {}
        all_indices = set(vec_idx) | set(bm_idx) | set(graph_scores.keys())
        for idx in all_indices:
            if 0 <= idx < len(self.documents):
                v = vec_scores.get(idx, 0) * self.vector_weight
                b = bm_scores.get(idx, 0) * self.bm25_weight
                g = graph_scores.get(idx, 0) * self.graph_weight
                combined_scores[idx] = (v + b + g)

        # Sort by combined desc
        sorted_idx = sorted(combined_scores.keys(), key=lambda i: combined_scores[i], reverse=True)

        # Deduplicate by content similarity
        final_indices = []
        seen_content = set()
        for idx in sorted_idx:
            content = self.documents[idx].strip()
            content_hash = hash(content)
            if content_hash not in seen_content:
                final_indices.append(idx)
                seen_content.add(content_hash)
                if len(final_indices) >= n_results:
                    break

        # Gather final results
        docs = []
        metas = []
        scores = []
        dists = []
        for i in final_indices:
            docs.append(self.documents[i])
            metas.append(self.document_metadata[i])
            sc = combined_scores[i]
            scores.append(sc)
            dists.append(1.0 - min(sc, 1.0))  # trivial distance

        # Return in the standard format
        return {
            "query": query_text,
            "ids": [f"doc_{i}" for i in final_indices],
            "documents": docs,
            "metadatas": metas,
            "scores": scores,
            "distances": dists,
            "entities": query_entities,
        }

    def _vector_search(self, text: str, n_ret=10) -> Dict[str, Any]:
        """Cosine similarity search."""
        empty_result = {
            "query": text,
            "ids": [],
            "documents": [],
            "metadatas": [],
            "scores": [],
            "distances": [],
        }
        
        if not text.strip():
            return empty_result
        if len(text) > 2000:
            text = text[:2000]

        try:
            query_embedding = self.embedder.embed_string(text)
            if not os.path.exists(self.embeddings_path):
                logger.warning(f"No embeddings file found at {self.embeddings_path}")
                return empty_result
            
            try:
                embeddings = np.load(self.embeddings_path)
                if len(embeddings) == 0:
                    logger.warning("Embeddings file is empty")
                    return empty_result
            except Exception as e:
                logger.error(f"Error loading embeddings from {self.embeddings_path}: {e}")
                return empty_result

            # Calculate similarity scores
            try:
                similarities = cosine_similarity(query_embedding.reshape(1, -1), embeddings)[0]
            except Exception as e:
                logger.error(f"Error calculating cosine similarity: {e}")
                return empty_result

            # Get indices of top matches
            try:
                top_indices = np.argsort(similarities)[-n_ret:][::-1]
                scores = similarities[top_indices]
            except Exception as e:
                logger.error(f"Error getting top matches: {e}")
                return empty_result

            # Filter by threshold
            mask = scores >= self.similarity_threshold
            top_indices = top_indices[mask]
            scores = scores[mask]

            if len(top_indices) == 0:
                logger.info(f"No results above threshold {self.similarity_threshold}")
                return empty_result

            return {
                "query": text,
                "ids": [f"doc_{i}" for i in top_indices],
                "scores": scores.tolist(),
                "distances": (1.0 - scores).tolist(),
            }

        except Exception as e:
            logger.error(f"Vector search error: {e}", exc_info=True)
            return empty_result

    def _graph_search(self, entities: List[str], n_ret: int = 10) -> Dict[int, float]:
        """Gather relevant chunks by entity co‐mentions in the knowledge-graph graph."""
        if not entities or not self.graph.nodes:
            return {}
            
        scores: Dict[int, float] = defaultdict(float)
        
        # direct matches
        for ent in entities:
            for ch_idx in self.entity_to_chunks.get(ent, []):
                scores[ch_idx] += 1.0
                
        # neighbors
        for ent in entities:
            if ent not in self.graph:
                continue
            for nbr in self.graph[ent]:
                # Get edge data and ensure proper type casting
                edge_data = cast(Dict[str, Any], self.graph[ent][nbr])
                weight_value = edge_data.get("weight", 1.0)
                # Ensure we have a float
                weight = float(weight_value) if weight_value is not None else 1.0
                    
                rel_score = min(1.0, weight / 10.0)
                for ch_idx in self.entity_to_chunks.get(nbr, []):
                    scores[ch_idx] += rel_score
        
        # normalize
        if scores:
            score_values = [float(v) for v in scores.values()]
            max_score = max(score_values) if score_values else 0.0
            if max_score > 0:
                for k in scores:
                    scores[k] /= max_score
                    
        # sort + limit
        return dict(sorted(scores.items(), key=lambda x: (float(x[1]), x[0]), reverse=True)[:n_ret])

    def close(self):
        """Flush metadata, knowledge graph, etc."""
        self._save_metadata()
        self._save_knowledge_graph()
        self.documents = []
        self.document_metadata = []
        gc.collect()
        logger.info("VectorDB resources cleaned up")
