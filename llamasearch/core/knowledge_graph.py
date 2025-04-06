import os
import json
import spacy
import sys
import subprocess
import logging
from typing import Dict, Any, List, Optional

from ..setup_utils import find_project_root

logger = logging.getLogger(__name__)

def jaccard_similarity(text1: str, text2: str) -> float:
    """Compute the Jaccard similarity between two texts."""
    tokens1 = set(text1.lower().split())
    tokens2 = set(text2.lower().split())
    if not tokens1 or not tokens2:
        return 0.0
    return len(tokens1.intersection(tokens2)) / len(tokens1.union(tokens2))

def load_reverse_lookup() -> Dict[str, str]:
    """
    Load the reverse lookup table from data/reverse_lookup.json.
    This maps a document's hash to its original URL.
    """
    project_root = find_project_root()
    reverse_lookup_path = os.path.join(project_root, "data", "reverse_lookup.json")
    if os.path.exists(reverse_lookup_path):
        with open(reverse_lookup_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                logger.info(f"Loaded reverse lookup table with {len(data)} entries.")
                return data
            except Exception as e:
                logger.error(f"Error loading reverse lookup: {e}")
    else:
        logger.warning("Reverse lookup table not found.")
    return {}

class KnowledgeGraph:
    """
    Builds a global knowledge graph from all documents using spaCy NER.
    For each recognized entity, it stores:
      - the entity's label,
      - a list of occurrences (each with a sentence, document source, and context),
      - and co-mention (relationship) information.
    
    In addition, it adds document nodes (prefixed with "doc:") for each document (identified by its source URL)
    and links them with the entities that occur in that document.
    
    When provided with hyperlinks (extracted by the crawler), the graph uses a reverse lookup table
    to add edges between document nodes, capturing inter-document links.
    
    The _disambiguate method is enhanced to consider the document source so that the same surface form 
    (e.g. "Sarah") is treated as different if it comes from clearly different contexts.
    """
    def __init__(self, storage_dir: Optional[str] = None):
        """Initialize the knowledge graph.
        
        Args:
            storage_dir: Directory to store knowledge graph files. If None, uses default index/knowledge_graph directory.
        """
        if storage_dir is None:
            project_root = find_project_root()
            storage_dir = os.path.join(project_root, "index", "knowledge_graph")
        
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)
        
        self.graph_path = os.path.join(self.storage_dir, "graph.json")
        self.entity_path = os.path.join(self.storage_dir, "entities.json")
        self.relation_path = os.path.join(self.storage_dir, "relations.json")
        
        # Load or initialize NLP pipeline
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("Loaded spaCy model successfully")
        except OSError:
            logger.warning("SpaCy model not found. Attempting to install...")
            try:
                subprocess.run(
                    [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
                    check=True,
                )
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("Installed and loaded spaCy model successfully")
            except Exception as e:
                logger.error(f"Failed to install spaCy model: {e}")
                self.nlp = spacy.blank("en")

        # Initialize or load graph data
        self._load_or_init_graph()

    def _load_or_init_graph(self) -> None:
        """Load existing graph data or initialize new graph."""
        try:
            with open(self.graph_path, "r", encoding="utf-8") as f:
                self.graph = json.load(f)
            with open(self.entity_path, "r", encoding="utf-8") as f:
                self.entities = json.load(f)
            with open(self.relation_path, "r", encoding="utf-8") as f:
                self.relations = json.load(f)
            logger.info(f"Loaded knowledge graph from {self.storage_dir}")
        except (FileNotFoundError, json.JSONDecodeError):
            self.graph = {"nodes": [], "edges": []}
            self.entities = {}
            self.relations = {}
            logger.info("Initialized new knowledge graph")

    def save(self) -> None:
        """Save graph data to storage directory."""
        os.makedirs(self.storage_dir, exist_ok=True)
        
        with open(self.graph_path, "w", encoding="utf-8") as f:
            json.dump(self.graph, f, indent=2, default=str)
        with open(self.entity_path, "w", encoding="utf-8") as f:
            json.dump(self.entities, f, indent=2, default=str)
        with open(self.relation_path, "w", encoding="utf-8") as f:
            json.dump(self.relations, f, indent=2, default=str)
        
        logger.info(f"Saved knowledge graph to {self.storage_dir}")

    def _disambiguate(self, base_name: str, sent_text: str, source: str, threshold: float = 0.5) -> str:
        """
        Disambiguate an entity occurrence.
        First, look for existing candidates that share the same base_name.
        If any candidate has an occurrence from the same source, return it.
        Otherwise, use jaccard similarity on the sentence context to determine if it matches.
        If no match is found, generate a new disambiguated key.
        """
        candidates = [k for k in self.entities if k.startswith(base_name)]
        for cand in candidates:
            for occ in self.entities[cand]["occurrences"]:
                # If the occurrence comes from the same document, consider it a match.
                if occ.get("source") == source:
                    return cand
                # Otherwise, check similarity of sentence context.
                if jaccard_similarity(occ["sentence"], sent_text) >= threshold:
                    return cand
        # No matching candidate found – if base_name is unused, return it; else append a suffix.
        if base_name not in self.entities:
            return base_name
        else:
            suffix = 2
            new_key = f"{base_name}_{suffix}"
            while new_key in self.entities:
                suffix += 1
                new_key = f"{base_name}_{suffix}"
            return new_key

    def _add_entity_occurrence(self, entity_text: str, label: str, sent_text: str, source: str) -> None:
        """
        Adds an occurrence for an entity, disambiguating it against existing entities.
        """
        key = self._disambiguate(entity_text, sent_text, source)
        if key not in self.entities:
            self.entities[key] = {
                "label": label,
                "occurrences": [],
                "co_mentions": {}
            }
        self.entities[key]["occurrences"].append({
            "sentence": sent_text,
            "source": source
        })

    def _add_co_mentions(self, disamb_entities: List[str], sent_text: str, source: str) -> None:
        """
        For all pairs of disambiguated entities in the same sentence, add co-mention edges.
        """
        for i in range(len(disamb_entities)):
            for j in range(i + 1, len(disamb_entities)):
                e1 = disamb_entities[i]
                e2 = disamb_entities[j]
                # Update co-mention for e1
                if e2 not in self.entities[e1]["co_mentions"]:
                    self.entities[e1]["co_mentions"][e2] = []
                self.entities[e1]["co_mentions"][e2].append({
                    "sentence": sent_text,
                    "source": source
                })
                # And vice versa
                if e1 not in self.entities[e2]["co_mentions"]:
                    self.entities[e2]["co_mentions"][e1] = []
                self.entities[e2]["co_mentions"][e1].append({
                    "sentence": sent_text,
                    "source": source
                })

    def _add_document_node(self, source: str) -> None:
        """
        Add a document node to the graph if it doesn't exist.
        Document nodes are keyed as "doc:" + source.
        """
        doc_id = f"doc:{source}"
        if doc_id not in self.graph["nodes"]:
            self.graph["nodes"].append({
                "id": doc_id,
                "source": source,
                "entities": set(),
                "hyperlinks": set()
            })
        # Also, add the document node to the overall graph dictionary.
        if doc_id not in self.relations:
            self.relations[doc_id] = {}

    def _link_entity_to_document(self, entity_key: str, source: str) -> None:
        """
        Link an entity node to a document node.
        We add an edge between the entity (identified by its disambiguated key)
        and the document node (doc:<source>).
        """
        doc_id = f"doc:{source}"
        self._add_document_node(source)
        # Record the relationship in the entity node.
        if "documents" not in self.entities[entity_key]:
            self.entities[entity_key]["documents"] = set()
        self.entities[entity_key]["documents"].add(source)
        # Also record in the document node.
        for node in self.graph["nodes"]:
            if node["id"] == doc_id:
                node["entities"].add(entity_key)
        # In the overall graph, add bidirectional edges.
        if entity_key not in self.relations:
            self.relations[entity_key] = {}
        # Add edge from entity to document.
        self.relations[entity_key][doc_id] = {"relation": "appears_in"}
        # And edge from document to entity.
        self.relations[doc_id][entity_key] = {"relation": "contains"}

    def _link_documents(self, source: str, hyperlinks: List[str]) -> None:
        """
        For each hyperlink in the current document, use the reverse lookup table to see if it matches
        an already-indexed document. If so, add an edge between the two document nodes.
        """
        current_doc_id = f"doc:{source}"
        self._add_document_node(source)
        for link in hyperlinks:
            # Normalize the link to match the lookup table format.
            normalized = link.strip().lower()
            # If the reverse lookup maps some hash to a URL that matches this link:
            for doc_hash, doc_url in load_reverse_lookup().items():
                if normalized in doc_url.lower():
                    target_doc_id = f"doc:{doc_url}"
                    self._add_document_node(doc_url)
                    # Add an edge between current_doc_id and target_doc_id.
                    if target_doc_id not in self.relations[current_doc_id]:
                        self.relations[current_doc_id][target_doc_id] = {"relation": "hyperlink"}
                    if current_doc_id not in self.relations[target_doc_id]:
                        self.relations[target_doc_id][current_doc_id] = {"relation": "hyperlink"}
                    # Also record in the document node metadata.
                    for node in self.graph["nodes"]:
                        if node["id"] == current_doc_id:
                            node["hyperlinks"].add(doc_url)
                    break  # Assuming one match per link; adjust if needed.

    def build_from_text(self, text: str, context_file: str = "", hyperlinks: Optional[List[str]] = None) -> None:
        """
        Build (or update) the global knowledge graph from the provided text.
        Each sentence is processed using spaCy; for each entity occurrence,
        we disambiguate based on its context (sentence and document source) and then record the occurrence.
        Co-mentions are then added between disambiguated entities in the same sentence.
        
        Additionally, a document node is created for the context_file, and every entity from this document
        is linked to it. If a list of hyperlinks is provided, they are used with the reverse lookup
        table to add inter-document links.
        """
        if self.nlp is None:
            raise RuntimeError("SpaCy NLP model is not loaded.")
        
        # Ensure a document node exists for this document.
        if context_file:
            self._add_document_node(context_file)
        
        doc = self.nlp(text)
        for sent in doc.sents:
            sent_text = sent.text.strip()
            if not sent_text:
                continue
            disamb_entities = []
            for ent in sent.ents:
                ent_text = ent.text.strip()
                if not ent_text:
                    continue
                # Disambiguate using both sentence and context_file (source).
                key = self._disambiguate(ent_text, sent_text, context_file)
                self._add_entity_occurrence(ent_text, ent.label_, sent_text, context_file)
                disamb_entities.append(key)
                # Link the entity to the document node.
                if context_file:
                    self._link_entity_to_document(key, context_file)
            # Add co-mention relationships among the disambiguated entities.
            if disamb_entities:
                self._add_co_mentions(disamb_entities, sent_text, context_file)
        
        # If hyperlinks were provided, process them to add inter-document edges.
        if hyperlinks and context_file:
            self._link_documents(context_file, hyperlinks)

    def get_entity_info(self, ent_key: str) -> Optional[Dict[str, Any]]:
        """Retrieve the information for a given disambiguated entity key."""
        return self.entities.get(ent_key)

    def get_all_entities(self) -> List[str]:
        """Return all disambiguated entity keys."""
        return list(self.entities.keys())

    def get_co_mentions(self, ent_key: str) -> Dict[str, Any]:
        """Return the co-mentions for the specified entity key."""
        entity = self.entities.get(ent_key)
        if not entity:
            return {}
        return entity.get("co_mentions", {})
