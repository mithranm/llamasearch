import os
import numpy as np
import logging
import gc
import shutil
import json
import networkx as nx
from collections import defaultdict
from typing import Dict, Any, Optional, List

from ..utils import find_project_root, log_query, NumpyEncoder
from .embedder import EnhancedEmbedder
from .bm25 import BM25Retriever
from .chunker import MarkdownChunker, HtmlChunker
from .knowledge_graph import KnowledgeGraph
from sklearn.metrics.pairwise import cosine_similarity

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
            # Use the consolidated index directory structure
            self.storage_dir = os.path.join(project_root, "index", f"vector_{suff}")

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

        # Initialize the knowledge graph (which now includes inter-document relationships)
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

        # Ensure llm_instance attribute exists (to be set externally)
        self.llm_instance = None

    def _load_metadata(self) -> None:
        if os.path.exists(self.metadata_path):
            logger.info(f"Loading metadata from {self.metadata_path}")
            try:
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.documents = data.get("documents", [])
                self.document_metadata = data.get("metadata", [])
                for doc, meta in zip(self.documents, self.document_metadata):
                    # Extract ID from metadata for BM25 indexing
                    doc_id = str(meta.get('id', '')) if isinstance(meta, dict) else str(meta)
                    self.bm25.add_document(doc, doc_id)
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
                json.dump(self.kg.entities, f, ensure_ascii=False, cls=NumpyEncoder)
            os.replace(temp_path, self.kg_path)
            project_root = find_project_root()
            data_dir = os.path.join(project_root, "data")
            os.makedirs(data_dir, exist_ok=True)
            data_path = os.path.join(data_dir, f"{self.collection_name}_knowledge_graph.json")
            with open(data_path, "w", encoding="utf-8") as f:
                json.dump(self.kg.entities, f, ensure_ascii=False, cls=NumpyEncoder)
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

    def is_document_processed(self, file_path: str) -> bool:
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
        # Here we pass an empty list for hyperlinks; you can modify this if hyperlink info is available.
        self.kg.build_from_text(full_text, context_file=file_path, hyperlinks=[])
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
                # Calculate the starting index for this batch
                doc_start_idx = len(self.documents) - len(texts)
                
                for i, (t, m) in enumerate(zip(texts, meta_list)):
                    # Generate a consistent document ID using the pattern doc_X
                    # where X is the index of the document in the global document list
                    doc_idx = doc_start_idx + i
                    doc_id = f"doc_{doc_idx}"
                    
                    # Store the ID in the metadata for future reference
                    m['id'] = doc_idx
                    
                    # Add to BM25 index with the consistent ID
                    self.bm25.add_document(t, doc_id)
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
                    if isinstance(raw_weight, dict):
                        weight = 0.0
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

    def vectordb_query(self, query_text: str, show_retrieved_chunks: bool = True, query_entities: Optional[List[str]] = None) -> Dict[str, Any]:
        """Query the vector database with better error handling and diagnostics"""
        try:
            # Initialize empty response structure for fallback
            empty_response = {
                "query": query_text,
                "ids": [],
                "documents": [],
                "metadatas": [],
                "scores": [],
                "distances": [],
                "score_details": [],
                "debug_chunks": [],
                "error": None
            }
            
            # Validate if documents are available
            if not self.documents:
                logger.warning("No docs in DB; returning empty results.")
                empty_response["error"] = "No documents in database"
                return empty_response
                
            # BM25 search with robust error handling
            logger.info(f"Performing BM25 search for query: {query_text}")
            bm_res = self.bm25.query(query_text, n_results=10)
            bm_idx: List[int] = []
            bm_scores: Dict[int, float] = {}
            
            # Log what we got from BM25
            bm_ids = bm_res.get('ids', [])
            logger.debug(f"BM25 returned {len(bm_ids)} results")
            if bm_ids and len(bm_ids) > 0:
                logger.debug(f"BM25 IDs sample: {bm_ids[:min(3, len(bm_ids))]}")
            
            # Safely convert IDs to indices, handling empty strings or other formatting issues
            for i, iid in enumerate(bm_ids):
                try:
                    # Try to extract the numeric index from IDs in the format "doc_X"
                    if isinstance(iid, str) and iid.startswith("doc_"):
                        idx = int(iid[4:])
                    else:
                        # If no doc_ prefix, try to convert directly if it's numeric
                        idx = int(iid) if iid and isinstance(iid, str) and iid.strip() else i
                    
                    # Validate that the index is within bounds
                    if 0 <= idx < len(self.documents):
                        bm_idx.append(idx)
                        # Get the corresponding score
                        scores = bm_res.get("scores", [])
                        if i < len(scores):
                            bm_scores[idx] = float(scores[i])
                    else:
                        logger.warning(f"BM25 returned out of bounds index: {idx} (max: {len(self.documents)-1})")
                except (ValueError, TypeError) as e:
                    logger.warning(f"Could not parse BM25 ID: '{iid}', error: {str(e)}")
            
            bm_details: Dict[int, Dict[str, Any]] = {}
            for d in bm_res.get("score_details", []):
                if isinstance(d, dict) and "index" in d:
                    bm_details[d["index"]] = d
            
            # Vector search with robust error handling
            logger.info(f"Performing vector search for query: {query_text}")
            vec_res = self._vector_search(query_text, n_ret=10)
            vec_idx: List[int] = []
            vec_scores: Dict[int, float] = {}
            
            # Log what we got from vector search
            vec_ids = vec_res.get('ids', [])
            logger.debug(f"Vector search returned {len(vec_ids)} results")
            if vec_ids and len(vec_ids) > 0:
                logger.debug(f"Vector IDs sample: {vec_ids[:min(3, len(vec_ids))]}")
            
            # Safely convert IDs to indices with validation
            for i, iid in enumerate(vec_ids):
                try:
                    # Try to extract the numeric index from IDs in the format "doc_X"
                    if isinstance(iid, str) and iid.startswith("doc_"):
                        idx = int(iid[4:])
                    else:
                        # If no doc_ prefix, try to convert directly if it's numeric
                        idx = int(iid) if iid and isinstance(iid, str) and iid.strip() else i
                    
                    # Validate that the index is within bounds
                    if 0 <= idx < len(self.documents):
                        vec_idx.append(idx)
                        # Get the corresponding score
                        scores = vec_res.get("scores", [])
                        if i < len(scores):
                            vec_scores[idx] = float(scores[i])
                    else:
                        logger.warning(f"Vector search returned out of bounds index: {idx} (max: {len(self.documents)-1})")
                except (ValueError, TypeError) as e:
                    logger.warning(f"Could not parse vector search ID: '{iid}', error: {str(e)}")
                    
            vec_details: Dict[int, Dict[str, Any]] = {}
            for d in vec_res.get("score_details", []):
                if isinstance(d, dict) and "index" in d:
                    vec_details[d["index"]] = d
        
            # Extract query entities using the knowledge graph's spaCy model if available.
            if hasattr(self, 'kg') and self.kg.nlp is not None:
                doc = self.kg.nlp(query_text)
                query_entities = [ent.text.strip() for ent in doc.ents if ent.text.strip()]
                logger.info(f"Extracted entities using KG NLP: {query_entities}")
            else:
                # Fallback: use title-cased tokens
                query_entities = [token for token in query_text.split() if token.istitle()]
                logger.info(f"Using title-cased tokens as entities (KG NLP not available): {query_entities}")
            
            # Graph search
            graph_res = self._graph_search(query_entities, n_ret=10)
            graph_scores: Dict[int, float] = {}
            if graph_res is not None and "scores" in graph_res and graph_res["scores"] is not None:
                for k, v in graph_res["scores"].items():
                    try:
                        graph_scores[int(k)] = float(v)
                    except (ValueError, TypeError):
                        logger.warning(f"Could not convert graph score key={k}, value={v}")
            
            graph_details: Dict[int, Dict[str, Any]] = {}
            if graph_res is not None and "score_details" in graph_res and graph_res["score_details"] is not None:
                graph_details = graph_res["score_details"]

            combined_scores: Dict[int, float] = {}
            score_details: Dict[int, Dict[str, Any]] = {}
            all_indices = set(bm_idx) | set(vec_idx) | set(graph_scores.keys())
            for idx in all_indices:
                # Ensure index is valid
                if not (0 <= idx < len(self.documents)):
                    logger.warning(f"Skipping out of bounds index in combined scores: {idx}")
                    continue
                    
                vec = vec_scores.get(idx, 0.0)
                bm25 = bm_scores.get(idx, 0.0)
                graph = graph_scores.get(idx, 0.0)
                v = vec * self.vector_weight
                b = bm25 * self.bm25_weight
                g = graph * self.graph_weight
                combined = v + b + g
                combined_scores[idx] = combined
                
                vector_formula = f"{vec:.4f}"
                if idx in vec_details and "formula" in vec_details[idx]:
                    vector_formula = vec_details[idx]["formula"]
                    
                bm25_formula = f"{bm25:.4f}"
                if idx in bm_details and "formula" in bm_details[idx]:
                    bm25_formula = bm_details[idx]["formula"]
                    
                graph_formula = f"{graph:.4f}"
                if idx in graph_details and "formula" in graph_details[idx]:
                    graph_formula = graph_details[idx]["formula"]
                
                # Create document detail
                detail = {
                    "index": idx,
                    "vector_score": vec,
                    "bm25_score": bm25,
                    "graph_score": graph,
                    "vector_weight": self.vector_weight,
                    "bm25_weight": self.bm25_weight,
                    "graph_weight": self.graph_weight,
                    "combined_score": combined,
                    "vector_formula": vector_formula,
                    "bm25_formula": bm25_formula,
                    "graph_formula": graph_formula,
                    "combined_formula": f"{v:.4f} + {b:.4f} + {g:.4f} = {combined:.4f}",
                    "latex": f"\\text{{score}}_{{{idx}}} = {vec:.4f}\\times{self.vector_weight} + {bm25:.4f}\\times{self.bm25_weight} + {graph:.4f}\\times{self.graph_weight} = {combined:.4f}"
                }
                
                # Safely add metadata
                if idx < len(self.document_metadata):
                    detail["metadata"] = self.document_metadata[idx]
                else:
                    detail["metadata"] = {"error": "metadata missing for this document"}
                
                score_details[idx] = detail

            def jaccard_similarity(text1: str, text2: str) -> float:
                """Calculate Jaccard similarity between two text strings"""
                tokens1 = set(text1.lower().split())
                tokens2 = set(text2.lower().split())
                if not tokens1 or not tokens2:
                    return 0.0
                return len(tokens1.intersection(tokens2)) / len(tokens1.union(tokens2))

            # Final results processing with bounds checking
            final_indices: List[int] = []
            logger.info(f"Processing {len(combined_scores)} candidate documents for final results")
            
            for idx in sorted(combined_scores, key=lambda i: combined_scores[i], reverse=True):
                # Ensure index is valid
                if not (0 <= idx < len(self.documents)):
                    logger.warning(f"Combined scores had invalid index: {idx} (max: {len(self.documents)-1})")
                    continue
                    
                candidate_content = self.documents[idx].strip()
                if any(jaccard_similarity(candidate_content, self.documents[sel]) > 0.8 for sel in final_indices):
                    continue
                final_indices.append(idx)
                if len(final_indices) >= 5:
                    break
                    
            logger.info(f"Selected {len(final_indices)} final documents for context")

            # Create final response with bounds checking
            docs = []
            metas = []
            scores_list = []
            distances = []
            details_list = []
            
            for i in final_indices:
                try:
                    if 0 <= i < len(self.documents):
                        docs.append(self.documents[i])
                        
                        # Ensure metadata exists for this document
                        if i < len(self.document_metadata):
                            metas.append(self.document_metadata[i])
                        else:
                            metas.append({"source": "unknown", "error": "metadata missing"})
                            
                        scores_list.append(combined_scores[i])
                        distances.append(1.0 - min(combined_scores[i], 1.0))
                        
                        if i in score_details:
                            details_list.append(score_details[i])
                        else:
                            details_list.append({"index": i, "error": "missing score details"})
                    else:
                        logger.warning(f"Document index out of bounds: {i}")
                except Exception as e:
                    logger.error(f"Error processing document {i}: {str(e)}")

            # Create debug chunks safely
            debug_chunks = []
            for i, doc_text in enumerate(docs):
                # Create safe metadata reference
                safe_metadata = {}
                if i < len(metas):
                    safe_metadata = metas[i]
                    
                # Create safe score reference
                safe_score = 0.0
                if i < len(scores_list):
                    safe_score = scores_list[i]
                    
                # Create document ID using index from final_indices
                doc_id = f"doc_{i}"
                if i < len(final_indices):
                    doc_id = f"doc_{final_indices[i]}"
                    
                debug_chunks.append({
                    "id": doc_id,
                    "score": safe_score,
                    "metadata": safe_metadata,
                    "text": (doc_text[:200] + "...") if len(doc_text) > 200 else doc_text
                })

            # Return final response
            return {
                "query": query_text,
                "ids": [f"doc_{i}" for i in final_indices],
                "documents": docs,
                "metadatas": metas,
                "scores": scores_list,
                "distances": distances,
                "entities": query_entities,
                "score_details": details_list,
                "debug_chunks": debug_chunks,
                "error": None
            }
            
        except Exception as e:
            # Log and return error
            logger.error(f"Error in vectordb_query: {str(e)}", exc_info=True)
            return {
                "query": query_text,
                "ids": [],
                "documents": [],
                "metadatas": [],
                "scores": [],
                "distances": [],
                "score_details": [],
                "debug_chunks": [],
                "error": str(e)
            }

    def close(self) -> None:
        self._save_metadata()
        self._save_knowledge_graph()
        self.documents = []
        self.document_metadata = []
        gc.collect()
        logger.info("VectorDB resources cleaned up.")

    def _log_query(self, query, context_chunks, response, debug_info=None):
        return log_query(query, context_chunks, response, debug_info)

    def _build_prompt(self, query: str, context: str, intent: dict) -> str:
        sys_msg = "You are a helpful AI assistant. Answer based on the provided context."
        logger.info(f"Context length: {len(context)}")
        if context:
            logger.info(f"Context preview: {context[:100]}...")
        prompt = f"assistant\n{sys_msg}\n\nuser\n"
        if context.strip():
            prompt += f"Context information:\n{context}\n\n"
        else:
            logger.warning("No context for prompt injection.")
        prompt += f"Question: {query}\n\nassistant\n"
        logger.info(f"Final prompt (truncated): {prompt[:300]}...")
        return prompt

    def _get_llm(self):
        if self.llm_instance is None:
            # Instead of trying to load LLM here, instruct that it must be set externally.
            logger.error("LLM instance not set in VectorDB. Please set self.llm_instance externally.")
            raise NotImplementedError("LLM loading should be handled by LlamaSearch; set self.llm_instance in VectorDB externally.")
        return self.llm_instance

    @property
    def documents_map(self) -> Dict[str, int]:
        """Map of document IDs to their index in the documents list."""
        return {str(meta.get('id', i)): i for i, meta in enumerate(self.document_metadata) if meta.get('id')}

    def add_document(self, doc_id: str, content: str, metadata: Dict[str, Any]) -> None:
        """Add a document to the database with its associated metadata."""
        # Ensure ID is in the metadata
        metadata['id'] = doc_id
        
        # Check if document already exists
        existing_idx = next((i for i, meta in enumerate(self.document_metadata) 
                           if meta.get('id') == doc_id), None)
        
        if existing_idx is not None:
            # Update existing document
            self.documents[existing_idx] = content
            self.document_metadata[existing_idx] = metadata
            logger.info(f"Updated document {doc_id}")
        else:
            # Add new document
            self.documents.append(content)
            self.document_metadata.append(metadata)
            logger.info(f"Added document {doc_id}")
            
        # Add to BM25 index
        # Extract ID from metadata for BM25 indexing
        bm25_doc_id = str(metadata.get('id', doc_id)) if isinstance(metadata, dict) else str(metadata)
        self.bm25.add_document(content, bm25_doc_id)
        
        # Save if persistence is enabled
        if self.persist:
            self._save_metadata()