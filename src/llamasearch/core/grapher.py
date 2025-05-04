import os
import json
import re
from typing import Dict, Any, List, Optional

from llamasearch.utils import setup_logging
from transformers.pipelines import pipeline

logger = setup_logging(__name__)

def get_phonetic_representation(name: str) -> str:
    """
    Produce a simple phonetic representation for approximate matching.
    """
    if not name:
        return ""
    text = name.lower()
    # Simplify certain character combos
    text = re.sub(r'zh|j', 'j', text)
    text = re.sub(r'ts|tz|z', 'z', text)
    text = re.sub(r'sh|sch|s', 's', text)
    text = re.sub(r'ch|tch|tsh', 'c', text)
    text = re.sub(r'ck|k|q', 'k', text)
    text = re.sub(r'f|ph', 'f', text)
    text = re.sub(r'gh|g', 'g', text)
    text = re.sub(r'ai|ay|ei|ey|e', 'e', text)
    text = re.sub(r'ow|oe|o', 'o', text)
    text = re.sub(r'oo|ou|u', 'u', text)
    text = re.sub(r'[^a-z0-9]', '', text)
    return text

def jaccard_similarity(text1: str, text2: str) -> float:
    """
    Jaccard similarity (based on word sets).
    """
    set1 = set(text1.lower().split())
    set2 = set(text2.lower().split())
    if not set1 or not set2:
        return 0.0
    return len(set1.intersection(set2)) / len(set1.union(set2))

def name_similarity(name1: str, name2: str) -> float:
    """
    Specialized similarity for personal names (multi-part, phonetic matching, etc.).
    Returns a score between 0.0 and 1.0.
    """
    if not name1 or not name2:
        return 0.0
    n1 = name1.lower()
    n2 = name2.lower()
    # Exact match
    if n1 == n2:
        return 1.0
    n1_parts = n1.split()
    n2_parts = n2.split()
    # Single name vs multi-part
    if len(n1_parts) == 1 and len(n2_parts) > 1:
        if n1_parts[0] in n2_parts:
            pos_bonus = 0.1 if n1_parts[0] == n2_parts[0] else 0
            return 0.8 + pos_bonus
    if len(n2_parts) == 1 and len(n1_parts) > 1:
        if n2_parts[0] in n1_parts:
            pos_bonus = 0.1 if n2_parts[0] == n1_parts[0] else 0
            return 0.8 + pos_bonus
    # Check for shared name parts
    common = set(n1_parts) & set(n2_parts)
    if common:
        proportion = len(common) / max(len(n1_parts), len(n2_parts))
        return proportion * 0.9
    # Fall back to phonetic
    p1 = get_phonetic_representation(n1)
    p2 = get_phonetic_representation(n2)
    if p1 and p2:
        if p1 in p2 or p2 in p1:
            return 0.7
        c1 = set(p1)
        c2 = set(p2)
        if c1 and c2:
            char_sim = len(c1 & c2) / len(c1 | c2)
            if char_sim > 0.6:
                return 0.6 * char_sim
    # Default to half-weighted Jaccard
    return jaccard_similarity(n1, n2) * 0.5

class TransformerEntity:
    def __init__(self, text: str, label: str) -> None:
        self.text = text
        self.label_ = label

    def __repr__(self) -> str:
        return f"{self.text} ({self.label_})"

class TransformerDoc:
    def __init__(self, text: str, ents: List[TransformerEntity]) -> None:
        self.text: str = text if text else ""
        self.ents: List[TransformerEntity] = ents

    @property
    def sents(self) -> List[str]:
        """
        Very basic sentence splitter.
        """
        if not self.text:
            return []
        return [s.strip() for s in self.text.split('.') if s.strip()]

    def __repr__(self) -> str:
        return f"TransformerDoc(ents={self.ents})"

def get_context_for_entity(doc: TransformerDoc, ent_text: str) -> str:
    """
    Returns the first sentence containing the entity text.
    """
    for sent in doc.sents:
        if ent_text.lower() in sent.lower():
            return sent
    return doc.text

def disambiguate_entity(
    base: str,
    new_context: str,
    candidates: List[Dict[str, Any]],
    threshold: float = 0.5
) -> Optional[Dict[str, Any]]:
    """
    Attempt to match 'base' entity to an existing candidate with similar context or name.
    """
    # Check if any candidate is labeled as person
    is_name = any(cand.get("label", "") == "PER" for cand in candidates)
    if is_name:
        # Person name logic
        for cand in candidates:
            cand_base = cand.get("base", "")
            sim = name_similarity(base, cand_base)
            # If strong name match, check context
            if sim >= 0.7:
                context_threshold = max(0.2, threshold - (sim - 0.7))
                for occ in cand.get("occurrences", []):
                    c_sim = jaccard_similarity(new_context, occ.get("context", ""))
                    if c_sim >= context_threshold:
                        return cand
                if sim >= 0.9:
                    return cand
        # Single-word name partial logic
        if len(base.split()) == 1:
            best_match = None
            best_score = 0.0
            for cand in candidates:
                cand_base = cand.get("base", "")
                cand_parts = cand_base.lower().split()
                if base.lower() in cand_parts:
                    # Add some position-based bonus
                    pos_score = 0.9 if base.lower() == cand_parts[0] else 0.7
                    best_c_sim = 0.0
                    for occ in cand.get("occurrences", []):
                        c_sim = jaccard_similarity(new_context, occ.get("context", ""))
                        best_c_sim = max(best_c_sim, c_sim)
                    total_score = pos_score + best_c_sim * 0.3
                    if total_score > best_score:
                        best_score = total_score
                        best_match = cand
            if best_match and best_score > threshold:
                return best_match
    else:
        # Non-person
        for cand in candidates:
            if cand.get("base", "").lower() == base.lower():
                for occ in cand.get("occurrences", []):
                    c_sim = jaccard_similarity(new_context, occ.get("context", ""))
                    if c_sim >= threshold:
                        return cand
    return None

class KnowledgeGraph:
    """
    Builds a global knowledge graph from documents using transformer-based NER.
    Provides 'add_document' for ingestion and 'search' for retrieval.
    """
    def __init__(self, storage_dir: Optional[str] = None) -> None:
        if storage_dir is None:
            raise ValueError("Storage directory must be specified")
        if not os.path.exists(storage_dir):
            raise FileNotFoundError(f"Storage directory does not exist: {storage_dir}")

        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)

        # Main graph file
        self.graph_path = os.path.join(self.storage_dir, "graph.json")
        # Entities file
        self.entity_indices_path = os.path.join(self.storage_dir, "entity_indices.json")
        # Name variants file
        self.name_variants_path = os.path.join(self.storage_dir, "name_variants.json")

        # NER pipeline
        logger.info("Initializing NER pipeline for KnowledgeGraph")
        self.ner_pipeline = pipeline(
            "ner",
            model="Babelscape/wikineural-multilingual-ner",
            aggregation_strategy="simple"
        )
        self.nlp = self._create_transformer_nlp()

        # Graph structures
        self.graph: Dict[str, Any] = {"nodes": [], "edges": []}
        self.entities: List[Dict[str, Any]] = []
        self.name_components: Dict[str, List[str]] = {}
        self.phonetic_map: Dict[str, List[str]] = {}

        self._load_or_init_graph()
        self._build_name_indices()

    def _create_transformer_nlp(self):
        """
        Returns a callable that runs NER on the text and constructs a TransformerDoc.
        """
        def nlp_func(text: str) -> TransformerDoc:
            try:
                results = self.ner_pipeline(text)
                if not isinstance(results, list):
                    results = list(results) if results else []
                ents = []
                for r in results:
                    if not isinstance(r, dict):
                        continue
                    ent_label = str(r.get("entity", "")).replace("B-", "").replace("I-", "")
                    word = str(r.get("word", ""))
                    if word.strip():
                        ents.append(TransformerEntity(text=word.strip(), label=ent_label))
                return TransformerDoc(text, ents)
            except Exception as exc:
                logger.error(f"Error in transformer NER: {exc}")
                return TransformerDoc(text, [])
        return nlp_func

    def _load_or_init_graph(self) -> None:
        """
        Load existing graph, or init empty if not found. Wrap in try/except for safe file operations.
        """
        if os.path.exists(self.graph_path):
            try:
                with open(self.graph_path, "r", encoding="utf-8") as f:
                    self.graph = json.load(f)
                logger.info(f"Loaded graph data from {self.graph_path}")
            except Exception as e:
                logger.error(f"Error loading graph data: {e}")
                self.graph = {"nodes": [], "edges": []}
        else:
            logger.info(f"No graph file found at {self.graph_path}; initializing empty graph.")
            self.graph = {"nodes": [], "edges": []}

        if os.path.exists(self.entity_indices_path):
            try:
                with open(self.entity_indices_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.entities = data.get("entities", [])
                logger.info(f"Loaded entity indices from {self.entity_indices_path}")
            except Exception as e:
                logger.error(f"Error loading entity indices: {e}")
                self.entities = []
        else:
            logger.info(f"No entity indices file found at {self.entity_indices_path}; initializing empty.")
            self.entities = []

        if os.path.exists(self.name_variants_path):
            try:
                with open(self.name_variants_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.name_components = data.get("name_components", {})
                    self.phonetic_map = data.get("phonetic_map", {})
                logger.info(f"Loaded name variants from {self.name_variants_path}")
            except Exception as e:
                logger.error(f"Error loading name variants: {e}")
                self.name_components = {}
                self.phonetic_map = {}

    def _build_name_indices(self) -> None:
        """
        Build name-component and phonetic maps from loaded entity data.
        """
        self.name_components.clear()
        self.phonetic_map.clear()
        for e in self.entities:
            if e.get("label") == "PER":
                base_name = e.get("base", "")
                if base_name:
                    # Phonetic
                    code = get_phonetic_representation(base_name)
                    if code:
                        self.phonetic_map.setdefault(code, [])
                        if base_name not in self.phonetic_map[code]:
                            self.phonetic_map[code].append(base_name)
                    # Name parts
                    parts = base_name.split()
                    if len(parts) > 1:
                        for p in parts:
                            if len(p) > 2:
                                low = p.lower()
                                self.name_components.setdefault(low, [])
                                if base_name not in self.name_components[low]:
                                    self.name_components[low].append(base_name)

    def _save_graph(self) -> None:
        """
        Internal method to save the graph structure (nodes/edges) to disk.
        """
        os.makedirs(os.path.dirname(self.graph_path), exist_ok=True)
        with open(self.graph_path, "w", encoding="utf-8") as f:
            json.dump(self.graph, f, ensure_ascii=False, indent=2)

    def save_knowledge_graph(self) -> None:
        """
        Public method to save the entire knowledge graph and entity indices to disk.
        """
        try:
            self._save_graph()
            with open(self.entity_indices_path, "w", encoding="utf-8") as f:
                json.dump({"entities": self.entities}, f, ensure_ascii=False, indent=2)
            with open(self.name_variants_path, "w", encoding="utf-8") as f:
                json.dump({
                    "name_components": self.name_components,
                    "phonetic_map": self.phonetic_map
                }, f, ensure_ascii=False, indent=2)
            logger.info("Knowledge graph and all indices saved successfully")
        except Exception as e:
            logger.error(f"Error saving knowledge graph: {e}")

    def _disambiguate_and_update(self, ent: Dict[str, Any], source: str, context: str) -> str:
        """
        Attempt to find an existing entity that matches 'ent' (based on name/context).
        If found, append an occurrence; else create a new entity.
        """
        base = ent["text"]
        label = ent["label"]
        threshold = 0.5
        if label == "PER":
            # Lower threshold for person
            threshold = 0.4

        # Find candidate entities
        candidates: List[Dict[str, Any]] = []
        if label == "PER":
            # Person: check name similarity
            for c in self.entities:
                if c.get("label") == "PER":
                    sim = name_similarity(base, c.get("base", ""))
                    if sim >= 0.7:
                        candidates.append(c)
        else:
            # Non-person
            for c in self.entities:
                if c.get("label") == label and c.get("base", "").lower() == base.lower():
                    candidates.append(c)

        # Disambiguate
        matched = disambiguate_entity(base, context, candidates, threshold)
        if matched:
            matched["occurrences"].append({"source": source, "context": context})
            return matched["key"]

        # No match => create new
        new_key = base
        count = 1
        existing_keys = {c["key"] for c in self.entities if "key" in c}
        while new_key in existing_keys:
            count += 1
            new_key = f"{base}_{count}"

        new_e = {
            "key": new_key,
            "base": base,
            "label": label,
            "occurrences": [{"source": source, "context": context}]
        }
        self.entities.append(new_e)

        # Update name indices if needed
        if label == "PER":
            code = get_phonetic_representation(base)
            if code:
                self.phonetic_map.setdefault(code, [])
                if base not in self.phonetic_map[code]:
                    self.phonetic_map[code].append(base)
            parts = base.split()
            if len(parts) > 1:
                for p in parts:
                    if len(p) > 2:
                        low = p.lower()
                        self.name_components.setdefault(low, [])
                        if base not in self.name_components[low]:
                            self.name_components[low].append(base)

        return new_key

    def build_from_text(self, text: str, source: str) -> None:
        """
        Extract NER from 'text' and incorporate into the knowledge graph.
        """
        doc = self.nlp(text)
        # Also try capitalized if text is all-lower
        if text.islower():
            capitalized = " ".join(w.capitalize() for w in text.split())
            doc2 = self.nlp(capitalized)
            all_ents = {f"{ent.text.lower()}_{ent.label_}": ent for ent in doc.ents}
            for e2 in doc2.ents:
                key = f"{e2.text.lower()}_{e2.label_}"
                if key not in all_ents:
                    all_ents[key] = e2
            entities = list(all_ents.values())
        else:
            entities = doc.ents

        # Ensure doc node in the graph
        doc_node_id = f"doc:{source}"
        if not any(n.get("id") == doc_node_id for n in self.graph["nodes"]):
            self.graph["nodes"].append({"id": doc_node_id, "source": source, "entities": []})

        # Now handle each entity
        for ent in entities:
            context = get_context_for_entity(doc, ent.text)
            ekey = self._disambiguate_and_update({"text": ent.text, "label": ent.label_}, source, context)

            # Ensure entity node
            if not any(n.get("id") == ekey for n in self.graph["nodes"]):
                self.graph["nodes"].append({"id": ekey, "base": ent.text, "label": ent.label_})

            # Connect doc node and entity node
            existing_edge = next(
                (edge for edge in self.graph["edges"]
                 if (edge["source"] == doc_node_id and edge["target"] == ekey)
                 or (edge["source"] == ekey and edge["target"] == doc_node_id)),
                None
            )
            if existing_edge:
                existing_edge["weight"] += 1
            else:
                self.graph["edges"].append({
                    "source": doc_node_id,
                    "target": ekey,
                    "weight": 1
                })

            # Add entity to doc node's 'entities'
            for node in self.graph["nodes"]:
                if node.get("id") == doc_node_id:
                    ents_list = node.setdefault("entities", [])
                    if ekey not in ents_list:
                        ents_list.append(ekey)
                    break

    def add_document(self, source: str, text: str) -> None:
        """
        Public method: Ingest a doc with ID 'source' and text 'text'.
        """
        self.build_from_text(text, source)
        self.save_knowledge_graph()

    def search(self, query_entities: List[str], limit: int = 10) -> Dict[str, Dict[str, float]]:
        """
        Return a dict where each matched entity maps to a dict of {doc_source: relevance_score}.
        """
        results: Dict[str, Dict[str, float]] = {}
        if not query_entities:
            return results

        # Expand queries with name components or phonetic
        expanded: List[str] = []
        for qe in query_entities:
            ql = qe.lower()
            if ql in self.name_components:
                expanded.extend(self.name_components[ql])
            ph = get_phonetic_representation(qe)
            if ph in self.phonetic_map:
                for nm in self.phonetic_map[ph]:
                    if nm not in expanded:
                        expanded.append(nm)
        if expanded:
            logger.info(f"Expanded queries {query_entities} -> {expanded}")
            query_entities = list(set(query_entities + expanded))

        # Scan each entity in self.entities
        for e in self.entities:
            label = e.get("label", "")
            base = e.get("base", "")
            ekey = e.get("key", "")
            # For each query entity, see if it matches
            for q_ent in query_entities:
                sim = 0.0
                if label == "PER":
                    sim = name_similarity(q_ent, base)
                    match = sim >= 0.6
                else:
                    ql = q_ent.lower()
                    b_lower = base.lower()
                    match = bool(ql == b_lower or ql in b_lower or b_lower in ql)
                    if match and ql == b_lower:
                        sim = 1.0
                    elif match:
                        sim = 0.7

                if match:
                    # Combine doc sources from occurrences + graph edges
                    doc_scores: Dict[str, float] = {}
                    for occ in e.get("occurrences", []):
                        doc_src = occ.get("source", "")
                        if doc_src:
                            sc = sim if label == "PER" else 0.7
                            doc_scores[doc_src] = max(doc_scores.get(doc_src, 0.0), sc)

                    # Also look at edges
                    for edge in self.graph["edges"]:
                        if edge.get("source") == ekey or edge.get("target") == ekey:
                            doc_node = (edge["source"] if edge["source"].startswith("doc:")
                                        else edge["target"] if edge["target"].startswith("doc:")
                                        else None)
                            if doc_node:
                                doc_src = doc_node[4:]  # strip "doc:"
                                wt = edge.get("weight", 1)
                                base_score = min(1.0, wt / 10.0)
                                sc = base_score * (sim if label == "PER" else 1.0)
                                doc_scores[doc_src] = max(doc_scores.get(doc_src, 0.0), sc)

                    if doc_scores:
                        if base not in results:
                            results[base] = {}
                        for ds, scv in doc_scores.items():
                            results[base][ds] = max(results[base].get(ds, 0.0), scv)

        # Limit top 'limit' docs per entity
        for ent_base, doc_dict in results.items():
            sorted_docs = sorted(doc_dict.items(), key=lambda x: x[1], reverse=True)
            results[ent_base] = dict(sorted_docs[:limit])
        return results
