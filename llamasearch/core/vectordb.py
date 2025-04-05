# llamasearch/core/vectordb.py

import os
import numpy as np
import logging
import gc
import shutil
import json
import networkx as nx
from collections import defaultdict
from typing import Dict, Any, Optional, List

from sklearn.metrics.pairwise import cosine_similarity

from .embedder import EnhancedEmbedder
from .bm25 import BM25Retriever
from .chunker import MarkdownChunker, HtmlChunker
from .knowledge_graph import KnowledgeGraph
from ..setup_utils import find_project_root

logger = logging.getLogger(__name__)


class VectorDB:
    """
    Enhanced VectorDB that integrates semantic chunking and knowledge graph–based retrieval.

    Features:
      - Triple heuristic: BM25 + vector similarity + graph entity relationships.
      - Consolidates document storage to reduce duplicate/overlapping chunks.
      - Preserves source traceability via metadata.
    """
    def __init__(
        self,
        collection_name: str = "documents",
        embedder: Optional[EnhancedEmbedder] = None,
        chunk_size: int = 500,
        text_embedding_size: int = 512,
        chunk_overlap: int = 100,
        min_chunk_size: int = 200,
        chunker_batch_size: Optional[int] = None,
        embedder_batch_size: Optional[int] = None,
        similarity_threshold: float = 0.15,
        max_chunks: int = 5000,
        persist: bool = False,
        storage_dir: Optional[str] = None,
        use_pca: bool = False,
        graph_weight: float = 0.3,
        bm25_weight: float = 0.35,
        vector_weight: float = 0.35,
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

        self.graph_weight = graph_weight
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight

        # Determine storage directory
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
            logger.info("Vector DB directory cleared")
        os.makedirs(self.storage_dir, exist_ok=True)

        self.metadata_path = os.path.join(self.storage_dir, f"{collection_name}_meta.json")
        self.embeddings_path = os.path.join(self.storage_dir, f"{collection_name}_embeddings.npy")
        self.kg_path = os.path.join(self.storage_dir, f"{collection_name}_knowledge_graph.json")

        self.embedder = embedder or EnhancedEmbedder(
            batch_size=embedder_batch_size,
            max_length=text_embedding_size,
        )

        # Initialize chunkers
        chunker_params: Dict[str, Any] = {
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
        self.chunker = self.markdown_chunker  # default

        self.bm25 = BM25Retriever()

        self.documents: List[str] = []
        self.document_metadata: List[Dict[str, Any]] = []

        self.kg = KnowledgeGraph()

        self.entity_to_chunks: Dict[str, set] = defaultdict(set)
        self.chunk_to_entities: Dict[int, set] = defaultdict(set)
        self.graph = nx.Graph()

        self._load_metadata()
        self._load_knowledge_graph()
        self._build_entity_indices()

        logger.info(f"Initialized VectorDB with {len(self.documents)} documents")
        logger.info(f"Storage dir: {self.storage_dir}")
        logger.info(f"Knowledge graph has {len(self.kg.entities)} entities")
        logger.info(f"Embeddings file: {self.embeddings_path}")
        logger.info(f"Persistence: {self.persist}, use_pca: {self.use_pca}")
        logger.info(f"Search weights => vector={self.vector_weight}, bm25={self.bm25_weight}, graph={self.graph_weight}")

    def _load_metadata(self) -> None:
        if os.path.exists(self.metadata_path):
            logger.info(f"Loading metadata from {self.metadata_path}")
            try:
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.documents = data.get("documents", [])
                self.document_metadata = data.get("metadata", [])
                for doc, meta in zip(self.documents, self.document_metadata):
                    self.bm25.add_document(doc, meta)
                logger.info(f"Loaded {len(self.documents)} document entries from metadata")
            except Exception as e:
                logger.error(f"Error loading metadata: {e}")
                self.documents = []
                self.document_metadata = []

    def _save_metadata(self) -> None:
        logger.info(f"Saving metadata for {len(self.documents)} documents.")
        temp_fn = f"temp_{os.path.basename(self.metadata_path)}"
        temp_path = os.path.join(os.path.dirname(self.metadata_path), temp_fn)
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                meta = {"documents": self.documents, "metadata": self.document_metadata}
                json.dump(meta, f)
            os.replace(temp_path, self.metadata_path)
            logger.info("Metadata saved.")
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def _load_knowledge_graph(self) -> None:
        if os.path.exists(self.kg_path):
            logger.info(f"Loading knowledge graph from {self.kg_path}")
            try:
                with open(self.kg_path, "r", encoding="utf-8") as f:
                    self.kg.entities = json.load(f)
                logger.info(f"Knowledge graph loaded with {len(self.kg.entities)} entities.")
            except Exception as e:
                logger.error(f"Error loading knowledge graph: {e}")
                self.kg.entities = {}

    def _save_knowledge_graph(self) -> None:
        logger.info(f"Saving knowledge graph with {len(self.kg.entities)} entities.")
        temp_fn = f"temp_{os.path.basename(self.kg_path)}"
        temp_path = os.path.join(os.path.dirname(self.kg_path), temp_fn)
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(self.kg.entities, f, ensure_ascii=False)
            os.replace(temp_path, self.kg_path)
            project_root = find_project_root()
            data_dir = os.path.join(project_root, "data")
            os.makedirs(data_dir, exist_ok=True)
            data_path = os.path.join(data_dir, f"{self.collection_name}_knowledge_graph.json")
            with open(data_path, "w", encoding="utf-8") as f:
                json.dump(self.kg.entities, f, ensure_ascii=False)
            logger.info(f"Knowledge graph saved to {self.kg_path} and {data_path}")
        except Exception as e:
            logger.error(f"Error saving knowledge graph: {e}")
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def _build_entity_indices(self) -> None:
        logger.info("Building entity-to-chunk and chunk-to-entity indices and relationship graph")
        self.entity_to_chunks.clear()
        self.chunk_to_entities.clear()
        self.graph.clear()
        for i, (chunk_text, meta) in enumerate(zip(self.documents, self.document_metadata)):
            if "entities" not in meta:
                meta["entities"] = self._extract_entities_from_chunk(chunk_text)
            for ent in meta["entities"]:
                etext = ent["text"]
                self.entity_to_chunks[etext].add(i)
                self.chunk_to_entities[i].add(etext)
                if not self.graph.has_node(etext):
                    self.graph.add_node(etext, label=ent["label"])
        for entity, info in self.kg.entities.items():
            co = info.get("co_mentions", {})
            for related_ent, co_list in co.items():
                if self.graph.has_edge(entity, related_ent):
                    edge_data: Dict[str, Any] = self.graph[entity][related_ent]
                    current_weight: float = float(edge_data.get("weight", 0.0))
                    new_weight: float = current_weight + float(len(co_list))
                    edge_data["weight"] = new_weight
                else:
                    self.graph.add_edge(entity, related_ent, weight=float(len(co_list)))
        logger.info(f"entity_to_chunks: {len(self.entity_to_chunks)} entity keys")
        logger.info(f"chunk_to_entities: {len(self.chunk_to_entities)} chunks total")
        logger.info(f"Graph: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")

    def _extract_entities_from_chunk(self, text: str) -> List[Dict[str, Any]]:
        entities: List[Dict[str, Any]] = []
        try:
            if not text or len(text) < 5 or not self.kg.nlp:
                return entities
            doc = self.kg.nlp(text)
            for ent in doc.ents:
                if ent.text.strip():
                    entities.append({"text": ent.text.strip(), "label": ent.label_})
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
        return entities

    def _merge_embeddings(self, temp_path: str) -> None:
        logger.info("Merging newly created embeddings with existing DB.")
        if os.path.exists(self.embeddings_path) and os.path.getsize(self.embeddings_path) > 0:
            try:
                arr_old = np.load(self.embeddings_path, allow_pickle=False)
                arr_new = np.load(temp_path, allow_pickle=False)
                merged = np.vstack([arr_old, arr_new])
                np.save(self.embeddings_path, merged, allow_pickle=False)
                logger.info("Embeddings merged successfully.")
            except Exception as e:
                logger.error(f"Error merging embeddings: {e}")
                raise
        else:
            shutil.copy2(temp_path, self.embeddings_path)

    # Re-introduced as a public method for external use
    def is_document_processed(self, file_path: str) -> bool:
        """Checks if the given file has already been processed based on its metadata and modification time."""
        if not self.document_metadata:
            return False
        disk_mtime = os.path.getmtime(file_path)
        for md in self.document_metadata:
            if md.get("source") == file_path:
                if abs(md.get("mtime", 0) - disk_mtime) < 1e-4:
                    return True
                else:
                    old_len = len(self.documents)
                    self.documents = [d for i, d in enumerate(self.documents)
                                      if self.document_metadata[i].get("source") != file_path]
                    self.document_metadata = [x for x in self.document_metadata
                                              if x.get("source") != file_path]
                    logger.info(f"File changed => removed {old_len - len(self.documents)} old chunks")
                    return False
        return False

    def add_document_chunks(self, file_path: str, chunks: List[Dict[str, Any]]) -> int:
        if not chunks:
            logger.warning(f"No chunks provided for {file_path}")
            return 0
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        if self.persist and self.is_document_processed(file_path):
            logger.info(f"Document already processed: {file_path}")
            return sum(1 for m in self.document_metadata if m.get("source") == file_path)
        doc_mtime = os.path.getmtime(file_path)
        base_meta: Dict[str, Any] = {"source": file_path, "filename": os.path.basename(file_path), "mtime": doc_mtime}
        full_text = "\n\n".join(c["chunk"] for c in chunks)
        self.kg.build_from_text(full_text, context_file=file_path)
        added_chunks = 0
        dest_dir = os.path.dirname(self.embeddings_path)
        temp_fn = f"temp_embeddings_{os.path.basename(self.embeddings_path)}"
        temp_path = os.path.join(dest_dir, temp_fn)
        try:
            batch_size = self.chunker_batch_size or 100
            all_batches = [chunks[i:i + batch_size] for i in range(0, len(chunks), batch_size)]
            total_chunks = len(chunks)
            logger.info(f"{os.path.basename(file_path)} => {total_chunks} chunks in {len(all_batches)} batches")
            for batch_idx, batch in enumerate(all_batches):
                if not batch:
                    continue
                texts = [x["chunk"] for x in batch]
                emb_texts = [x.get("embedding_text") or x["chunk"] for x in batch]
                meta_list: List[Dict[str, Any]] = []
                for i, bdat in enumerate(batch):
                    met = dict(base_meta)
                    met["batch_idx"] = batch_idx
                    met["chunk_idx"] = i
                    if "metadata" in bdat:
                        for k, v in bdat["metadata"].items():
                            if k == "source" and v:
                                met["source"] = v
                            else:
                                met[k] = v
                    meta_list.append(met)
                embeddings = self.embedder.embed_batch(emb_texts)
                embeddings = np.array(embeddings, dtype=np.float32)
                if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                    old = np.load(temp_path, allow_pickle=False)
                    merged = np.vstack([old, embeddings])
                    np.save(temp_path, merged, allow_pickle=False)
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
            logger.error(f"Error adding chunks from {file_path}: {e}")
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise

    def _vector_search(self, text: str, n_ret: int = 10) -> Dict[str, Any]:
        empty_result = {
            "query": text,
            "ids": [],
            "documents": [],
            "metadatas": [],
            "scores": [],
            "distances": [],
            "score_details": [],
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
            embeddings = np.load(self.embeddings_path, allow_pickle=False)
            embeddings = np.array(embeddings, dtype=np.float32)
            scores = cosine_similarity(query_embedding.reshape(1, -1), embeddings)[0]
            top_indices = np.argsort(scores)[::-1][:n_ret]
            top_scores = scores[top_indices]
            score_details = []
            for idx, score in zip(top_indices, top_scores):
                detail = {
                    "index": int(idx),
                    "cosine_similarity": float(score),
                    "formula": f"cos_sim(doc_{idx}) = {score:.4f}",
                    "latex": f"\\text{{cos\\_sim}}_{{{idx}}} = \\frac{{q \\cdot d_{{{idx}}}}}{{\\|q\\| \\cdot \\|d_{{{idx}}}\\|}} = {score:.4f}"
                }
                score_details.append(detail)
            mask = top_scores >= self.similarity_threshold
            top_indices = top_indices[mask]
            top_scores = top_scores[mask]
            score_details = [d for d, s in zip(score_details, top_scores) if s >= self.similarity_threshold]
            if len(top_indices) == 0:
                logger.info(f"No results above threshold {self.similarity_threshold}")
                return empty_result
            return {
                "query": text,
                "ids": [f"doc_{i}" for i in top_indices],
                "scores": top_scores.tolist(),
                "distances": (1.0 - top_scores).tolist(),
                "score_details": score_details
            }
        except Exception as e:
            logger.error(f"Vector search error: {e}", exc_info=True)
            return empty_result

    def _graph_search(self, entities: List[str], n_ret: int = 10) -> Dict[str, Any]:
        if not entities or not self.graph.nodes:
            return {"scores": {}, "score_details": {}}
        scores: Dict[int, float] = {}
        score_details: Dict[int, Dict[str, Any]] = {}
        def init_score_details() -> Dict[str, Any]:
            return {"direct_matches": [], "neighbor_matches": [], "total_score": 0.0, "formula": "", "latex": ""}
        for ent in entities:
            for ch_idx in self.entity_to_chunks.get(ent, set()):
                scores[ch_idx] = scores.get(ch_idx, 0.0) + 1.0
                if ch_idx not in score_details:
                    score_details[ch_idx] = init_score_details()
                score_details[ch_idx]["direct_matches"].append({"entity": ent, "score": 1.0})
        for ent in entities:
            if ent not in self.graph:
                continue
            for nbr in self.graph[ent]:
                raw_weight = self.graph[ent][nbr].get("weight", 0.0)
                try:
                    # Handle case where raw_weight might be a dict or other complex type
                    if isinstance(raw_weight, dict):
                        weight = 0.0  # Default value for dict type
                    else:
                        weight = float(raw_weight)
                except (ValueError, TypeError):
                    weight = 0.0
                rel_score = min(1.0, weight / 10.0)
                for ch_idx in self.entity_to_chunks.get(nbr, set()):
                    scores[ch_idx] = scores.get(ch_idx, 0.0) + rel_score
                    if ch_idx not in score_details:
                        score_details[ch_idx] = init_score_details()
                    score_details[ch_idx]["neighbor_matches"].append({
                        "entity": ent,
                        "neighbor": nbr,
                        "weight": weight,
                        "score": rel_score
                    })
        if scores:
            # Ensure all values are explicitly converted to float
            float_values = [float(v) for v in scores.values()]
            max_score = max(float_values) if float_values else 0.0
            if max_score > 0:
                for k in scores:
                    scores[k] = scores[k] / max_score
                    detail = score_details[k]
                    direct_terms = [f"1.0 ({m['entity']})" for m in detail["direct_matches"]]
                    neighbor_terms = [f"min(1.0, {m['weight']}/10) ({m['entity']}->{m['neighbor']})" for m in detail["neighbor_matches"]]
                    formula_parts = []
                    if direct_terms:
                        formula_parts.append(" + ".join(direct_terms))
                    if neighbor_terms:
                        formula_parts.append(" + ".join(neighbor_terms))
                    raw_score = sum(m["score"] for m in detail["direct_matches"]) + \
                                sum(m["score"] for m in detail["neighbor_matches"])
                    detail["total_score"] = scores[k]
                    detail["raw_score"] = raw_score
                    detail["max_score"] = max_score
                    detail["formula"] = f"graph_{k} = ({' + '.join(formula_parts)}) / {max_score:.4f} = {scores[k]:.4f}"
                    latex_parts = []
                    if direct_terms:
                        latex_parts.append(" + ".join("1.0" for _ in direct_terms))
                    if neighbor_terms:
                        latex_parts.append(" + ".join(f"\\min(1, w_{{{i}}}/10)" for i in range(len(neighbor_terms))))
                    detail["latex"] = f"\\text{{graph}}_{{{k}}} = \\frac{{{' + '.join(latex_parts)}}}{{{max_score:.4f}}} = {scores[k]:.4f}"
        sorted_items = sorted(scores.items(), key=lambda x: (x[1], x[0]), reverse=True)[:n_ret]
        final_scores = {k: v for k, v in sorted_items}
        final_details = {k: score_details[k] for k in final_scores}
        return {"scores": final_scores, "score_details": final_details}

    def query(self, query_text: str, show_retrieved_chunks: bool = True, debug_mode: bool = False) -> Dict[str, Any]:
        if not self.documents:
            logger.warning("No docs in DB; returning empty results.")
            return {
                "query": query_text,
                "ids": [],
                "documents": [],
                "metadatas": [],
                "scores": [],
                "distances": [],
                "score_details": [],
            }
        # BM25 search
        bm_res = self.bm25.query(query_text, n_results=10)
        bm_idx: List[int] = [int(iid[4:]) for iid in bm_res.get("ids", [])]
        bm_scores: Dict[int, float] = {idx: float(s) for idx, s in zip(bm_idx, bm_res.get("scores", []))}
        bm_details: Dict[int, Dict[str, Any]] = {d["index"]: d for d in bm_res.get("score_details", [])}
        # Vector search
        vec_res = self._vector_search(query_text, n_ret=10)
        vec_idx: List[int] = [int(iid[4:]) for iid in vec_res.get("ids", [])]
        vec_scores: Dict[int, float] = {idx: float(s) for idx, s in zip(vec_idx, vec_res.get("scores", []))}
        vec_details: Dict[int, Dict[str, Any]] = {d["index"]: d for d in vec_res.get("score_details", [])}
        # For query entities, simply use title-cased tokens
        query_entities = [token for token in query_text.split() if token.istitle()]
        # Graph search
        graph_res = self._graph_search(query_entities, n_ret=10)
        graph_scores: Dict[int, float] = {int(k): float(v) for k, v in graph_res.get("scores", {}).items()}
        graph_details: Dict[int, Dict[str, Any]] = graph_res.get("score_details", {})

        combined_scores: Dict[int, float] = {}
        score_details: Dict[int, Dict[str, Any]] = {}
        all_indices = set(bm_idx) | set(vec_idx) | set(graph_scores.keys())
        for idx in all_indices:
            vec = vec_scores.get(idx, 0.0)
            bm25 = bm_scores.get(idx, 0.0)
            graph = graph_scores.get(idx, 0.0)
            v = vec * self.vector_weight
            b = bm25 * self.bm25_weight
            g = graph * self.graph_weight
            combined = v + b + g
            combined_scores[idx] = combined
            detail = {
                "index": idx,
                "vector_score": vec,
                "bm25_score": bm25,
                "graph_score": graph,
                "vector_weight": self.vector_weight,
                "bm25_weight": self.bm25_weight,
                "graph_weight": self.graph_weight,
                "combined_score": combined,
                "vector_formula": vec_details.get(idx, {}).get("formula", f"{vec:.4f}"),
                "bm25_formula": bm_details.get(idx, {}).get("formula", f"{bm25:.4f}"),
                "graph_formula": graph_details.get(idx, {}).get("formula", f"{graph:.4f}"),
                "combined_formula": f"{v:.4f} + {b:.4f} + {g:.4f} = {combined:.4f}",
                "latex": f"\\text{{score}}_{{{idx}}} = {vec:.4f}\\times{self.vector_weight} + {bm25:.4f}\\times{self.bm25_weight} + {graph:.4f}\\times{self.graph_weight} = {combined:.4f}",
                "metadata": self.document_metadata[idx] if idx < len(self.document_metadata) else {}
            }
            score_details[idx] = detail

        def jaccard_similarity(text1: str, text2: str) -> float:
            tokens1 = set(text1.lower().split())
            tokens2 = set(text2.lower().split())
            if not tokens1 or not tokens2:
                return 0.0
            return len(tokens1.intersection(tokens2)) / len(tokens1.union(tokens2))

        final_indices: List[int] = []
        for idx in sorted(combined_scores, key=lambda i: combined_scores[i], reverse=True):
            candidate_content = self.documents[idx].strip()
            if any(jaccard_similarity(candidate_content, self.documents[sel]) > 0.8 for sel in final_indices):
                continue
            final_indices.append(idx)
            if len(final_indices) >= 5:
                break

        docs = [self.documents[i] for i in final_indices]
        metas = [self.document_metadata[i] for i in final_indices]
        scores_list = [combined_scores[i] for i in final_indices]
        distances = [1.0 - min(s, 1.0) for s in scores_list]
        details_list = [score_details[i] for i in final_indices]

        debug_chunks = []
        for i, doc_text in enumerate(docs):
            debug_chunks.append({
                "id": f"doc_{final_indices[i]}",
                "score": scores_list[i],
                "metadata": metas[i],
                "text": (doc_text[:200] + "...") if len(doc_text) > 200 else doc_text
            })

        return {
            "query": query_text,
            "ids": [f"doc_{i}" for i in final_indices],
            "documents": docs,
            "metadatas": metas,
            "scores": scores_list,
            "distances": distances,
            "entities": query_entities,
            "score_details": details_list,
            "debug_chunks": debug_chunks
        }

    def close(self) -> None:
        self._save_metadata()
        self._save_knowledge_graph()
        self.documents = []
        self.document_metadata = []
        gc.collect()
        logger.info("VectorDB resources cleaned up.")
