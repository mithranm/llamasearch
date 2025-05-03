import os
import json
import re
from typing import Dict, Any, List, Optional

from llamasearch.utils import setup_logging
from llamasearch.setup_utils import get_llamasearch_dir
from transformers.pipelines import pipeline

logger = setup_logging(__name__)

# --- Helper functions and classes for transformer-based NER and disambiguation ---

def get_phonetic_representation(name: str) -> str:
    """
    Get a simple phonetic representation of a name to match similar sounding names.
    This is a basic implementation that can be expanded with more sophisticated algorithms.
    """
    if not name:
        return ""
        
    # Convert to lowercase
    name = name.lower()
    
    # Replace common character groups with phonetic codes
    phonetic = name
    phonetic = re.sub(r'zh|j', 'j', phonetic)  # For Russian/Slavic names
    phonetic = re.sub(r'ts|tz|z', 'z', phonetic)
    phonetic = re.sub(r'sh|sch|s', 's', phonetic)
    phonetic = re.sub(r'ch|tch|tsh', 'c', phonetic)
    phonetic = re.sub(r'ck|k|q', 'k', phonetic)
    phonetic = re.sub(r'f|ph', 'f', phonetic)
    phonetic = re.sub(r'gh|g', 'g', phonetic)
    
    # Handle vowels similarly
    phonetic = re.sub(r'ai|ay|ei|ey|e', 'e', phonetic)
    phonetic = re.sub(r'ow|oe|o', 'o', phonetic)
    phonetic = re.sub(r'oo|ou|u', 'u', phonetic)
    
    # Remove any remaining non-alphanumeric characters
    phonetic = re.sub(r'[^a-z0-9]', '', phonetic)
    
    return phonetic

def jaccard_similarity(text1: str, text2: str) -> float:
    """Compute the Jaccard similarity between two texts (based on word sets)."""
    set1 = set(text1.lower().split())
    set2 = set(text2.lower().split())
    if not set1 or not set2:
        return 0.0
    return len(set1.intersection(set2)) / len(set1.union(set2))

def name_similarity(name1: str, name2: str) -> float:
    """
    Calculate similarity between two name strings.
    This function is specialized for personal names and handles:
    1. First name vs full name matching
    2. Different word orders
    3. Phonetic similarity for transliterated names
    
    Returns a similarity score from 0.0 to 1.0
    """
    if not name1 or not name2:
        return 0.0
        
    name1 = name1.lower()
    name2 = name2.lower()
    
    # Exact match
    if name1 == name2:
        return 1.0
        
    # Check for subset matches (first name only vs full name)
    name1_parts = name1.split()
    name2_parts = name2.split()
    
    # Single name vs multi-part name
    if len(name1_parts) == 1 and len(name2_parts) > 1:
        if name1_parts[0] in name2_parts:
            # First name match - higher score if it's the first component
            position_bonus = 0.1 if name1_parts[0] == name2_parts[0] else 0
            return 0.8 + position_bonus
            
    if len(name2_parts) == 1 and len(name1_parts) > 1:
        if name2_parts[0] in name1_parts:
            position_bonus = 0.1 if name2_parts[0] == name1_parts[0] else 0
            return 0.8 + position_bonus
            
    # Multi-part names - check for shared components
    common_parts = set(name1_parts) & set(name2_parts)
    if common_parts:
        proportion = len(common_parts) / max(len(name1_parts), len(name2_parts))
        return proportion * 0.9  # Scale slightly below full match
    
    # Check for phonetic similarity for non-exact matches
    phonetic1 = get_phonetic_representation(name1)
    phonetic2 = get_phonetic_representation(name2)
    
    if phonetic1 and phonetic2:
        # Check for phonetic subset match
        if phonetic1 in phonetic2 or phonetic2 in phonetic1:
            return 0.7
        
        # Else use general string similarity (edit distance would be better)
        chars1 = set(phonetic1)
        chars2 = set(phonetic2)
        if chars1 and chars2:
            char_similarity = len(chars1 & chars2) / len(chars1 | chars2)
            if char_similarity > 0.6:  # Threshold for meaningful similarity
                return 0.6 * char_similarity
    
    # Fall back to Jaccard for any remaining cases
    return jaccard_similarity(name1, name2) * 0.5  # Scale down for less reliable method

class TransformerEntity:
    def __init__(self, text: str, label: str) -> None:
        self.text = text
        self.label_ = label

    def __repr__(self) -> str:
        return f"{self.text} ({self.label_})"

class TransformerDoc:
    def __init__(self, text: str, ents: List[TransformerEntity]) -> None:
        self.text: str = text if text is not None else ""
        self.ents: List[TransformerEntity] = ents

    @property
    def sents(self) -> List[str]:
        if not self.text:
            return []
        # A very basic sentence splitter: split on period.
        return [sent.strip() for sent in self.text.split('.') if sent.strip()]

    def __repr__(self) -> str:
        return f"TransformerDoc(ents={self.ents})"

def merge_ner_results(ner_results: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Merge consecutive tokens that belong to the same entity.
    Assumes each element in ner_results is a dict.
    """
    if not ner_results:
        return []
    merged = []
    current = ner_results[0]
    current_index = int(current.get("index", 0))
    for result in ner_results[1:]:
        result = dict(result)
        next_index = int(result.get("index", 0))
        current_label = str(current.get("entity", "")).split("-")[-1]
        next_label = str(result.get("entity", "")).split("-")[-1]
        if next_index == current_index + 1 and current_label == next_label:
            current["word"] = str(current.get("word", "")) + str(result.get("word", "")).lstrip("##")
            current["index"] = next_index
            current_index = next_index
        else:
            merged.append({"text": str(current.get("word", "")), "label": current_label})
            current = result
            current_index = int(current.get("index", 0))
    merged.append({"text": str(current.get("word", "")), "label": str(current.get("entity", "")).split("-")[-1]})
    return merged

def get_context_for_entity(doc: TransformerDoc, ent_text: str) -> str:
    """
    Returns the first sentence from doc.sents that contains the entity text.
    If none is found, returns the full text.
    """
    for sent in doc.sents:
        if ent_text.lower() in sent.lower():
            return sent
    return doc.text

# --- Enhanced disambiguation logic ---

def disambiguate_entity(
    base: str, new_context: str, candidates: List[Dict[str, Any]], threshold: float = 0.5
) -> Optional[Dict[str, Any]]:
    """
    Enhanced disambiguation for entities, with special handling for personal names.
    
    For a given base entity name and new occurrence context, check candidate entities.
    For personal names, applies specialized name matching logic.
    
    Args:
        base: The base entity string to match
        new_context: Context where the entity appears
        candidates: List of candidate entities to check against
        threshold: Minimum similarity score to consider a match
        
    Returns:
        The matched candidate or None if no match found
    """
    # First check if this is likely a personal name
    is_name = False
    for candidate in candidates:
        if candidate.get("label", "") == "PER":
            is_name = True
            break
    
    if is_name:
        # Apply specialized name matching for personal entities
        for candidate in candidates:
            candidate_base = candidate.get("base", "")
            
            # Use name-specific similarity metric
            similarity = name_similarity(base, candidate_base)
            
            # If good name match, check context for confirmation
            if similarity >= 0.7:  # Higher threshold for name-only match
                # If strong name match, less context needed
                context_threshold = max(0.2, threshold - (similarity - 0.7))
                
                # Check if any occurrence context is similar
                for occ in candidate.get("occurrences", []):
                    context_sim = jaccard_similarity(new_context, occ.get("context", ""))
                    if context_sim >= context_threshold:
                        return candidate
                        
                # If very strong name match, return even with weak context
                if similarity >= 0.9:
                    return candidate
        
        # Special case: check if query is a component of a multi-part name
        if len(base.split()) == 1:  # Single word query like "Georgi"
            best_match = None
            best_score = 0
            
            for candidate in candidates:
                candidate_base = candidate.get("base", "")
                candidate_parts = candidate_base.lower().split()
                
                # Check if base is part of the candidate name
                if base.lower() in candidate_parts:
                    # Calculate score based on position (first name > middle > last)
                    position_score = 0.9 if base.lower() == candidate_parts[0] else 0.7
                    
                    # Include context similarity
                    best_context_sim = 0
                    for occ in candidate.get("occurrences", []):
                        context_sim = jaccard_similarity(new_context, occ.get("context", ""))
                        best_context_sim = max(best_context_sim, context_sim)
                    
                    # Combined score
                    score = position_score + best_context_sim * 0.3
                    
                    if score > best_score:
                        best_score = score
                        best_match = candidate
            
            # Return best match if above threshold
            if best_match and best_score > threshold:
                return best_match
    else:
        # Standard disambiguation for non-name entities
        for candidate in candidates:
            if candidate.get("base", "").lower() == base.lower():
                for occ in candidate.get("occurrences", []):
                    sim = jaccard_similarity(new_context, occ.get("context", ""))
                    if sim >= threshold:
                        return candidate
    
    return None

# --- Knowledge Graph Implementation with enhanced NER and disambiguation ---

class KnowledgeGraph:
    """
    Builds a global knowledge graph from documents using transformer-based NER.
    Enhanced with specialized handling for multilingual names.
    
    The graph maintains:
      - Document nodes (id: "doc:<source>", with source and a list of associated entity keys)
      - Entity nodes (id: unique key, with base text, label, and occurrences)
      - Edges between document and entity nodes with occurrence counts as weight.
      - Name variation maps for better cross-lingual name matching
    """

    def __init__(self, storage_dir: Optional[str] = None) -> None:
        if storage_dir is None:
            project_root = get_llamasearch_dir()
            storage_dir = os.path.join(project_root, "index", "knowledge_graph")
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)
        self.graph_path = os.path.join(self.storage_dir, "graph.json")
        self.entity_indices_path = os.path.join(self.storage_dir, "entity_indices.json")
        self.name_variants_path = os.path.join(self.storage_dir, "name_variants.json")
        
        # Ensure directories for saving graph and indices exist
        os.makedirs(os.path.dirname(self.graph_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.entity_indices_path), exist_ok=True)
        
        # Initialize NER pipeline using a multilingual model.
        self.ner_pipeline = pipeline("ner", model="Babelscape/wikineural-multilingual-ner", aggregation_strategy="simple")
        self.nlp = self._create_transformer_nlp()
        self.entities: List[Dict[str, Any]] = []  # Candidate entity nodes
        self.graph: Dict[str, Any] = {"nodes": [], "edges": []}
        self.relations: Dict[str, Any] = {}
        
        # Enhanced name handling
        self.name_components: Dict[str, List[str]] = {}  # Maps name parts to full names
        self.phonetic_map: Dict[str, List[str]] = {}  # Maps phonetic codes to names
        
        self._load_or_init_graph()
        self._build_name_indices()

    def _create_transformer_nlp(self):
        """
        Returns a function that processes text and returns a TransformerDoc.
        Converts each result from the NER pipeline to a dictionary with string keys.
        """
        def nlp_func(text: str) -> TransformerDoc:
            try:
                ner_results = self.ner_pipeline(text)
                if ner_results is None:
                    ner_results = []
                
                # Ensure ner_results is a list before checking its length
                if not isinstance(ner_results, list):
                    ner_results = list(ner_results)

                # Try with capitalized version if all lowercase and no entities found
                if len(ner_results) == 0 and text.islower():
                    # Capitalize first letter of each word
                    capitalized = ' '.join(w.capitalize() for w in text.split())
                    if capitalized != text:
                        cap_results = self.ner_pipeline(capitalized)
                        if cap_results:
                            ner_results = cap_results
                
                # Convert each result to a dict with string keys
                processed_results = []
                for r in ner_results:
                    if isinstance(r, dict):
                        new_r = {str(key): value for key, value in r.items()}
                    else:
                        new_r = {}  # Handle cases where r is not a dictionary
                    processed_results.append(new_r)
                
                ents = []
                for res in processed_results:
                    entity_group = str(res["entity_group"]) if "entity_group" in res else ""
                    entity_fallback = str(res["entity"]) if "entity" in res else ""
                    entity_label = (entity_group if entity_group else entity_fallback).replace("B-", "").replace("I-", "")
                    word = str(res["word"]) if "word" in res else ""
                    ents.append(TransformerEntity(text=word, label=entity_label))
                return TransformerDoc(text=text, ents=ents)
            except Exception as e:
                logger.error(f"Error in transformer NER: {e}")
                return TransformerDoc(text=text, ents=[])
        
        return nlp_func

    def _load_or_init_graph(self) -> None:
        """
        Loads graph and entity indices from disk if available; otherwise, initializes empty structures.
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
            logger.info(f"No entity indices file found at {self.entity_indices_path}; initializing empty entities.")
            self.entities = []
        
        # Load name variants if available
        if os.path.exists(self.name_variants_path):
            try:
                with open(self.name_variants_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.name_components = data.get("name_components", {})
                    self.phonetic_map = data.get("phonetic_map", {})
                logger.info(f"Loaded name variant indices from {self.name_variants_path}")
            except Exception as e:
                logger.error(f"Error loading name variants: {e}")
                self.name_components = {}
                self.phonetic_map = {}

    def _build_name_indices(self) -> None:
        """
        Build additional indices for enhanced name matching:
        1. Name component index: Maps individual name parts to full names
        2. Phonetic index: Maps phonetic representations to names
        """
        # Skip if no entities loaded
        if not self.entities:
            return
            
        self.name_components = {}
        self.phonetic_map = {}
        
        # Process all person entities
        for entity in self.entities:
            if entity.get("label") == "PER":
                full_name = entity.get("base", "")
                if not full_name:
                    continue
                    
                # Add phonetic mapping
                phonetic = get_phonetic_representation(full_name)
                if phonetic:
                    if phonetic not in self.phonetic_map:
                        self.phonetic_map[phonetic] = []
                    if full_name not in self.phonetic_map[phonetic]:
                        self.phonetic_map[phonetic].append(full_name)
                
                # Add name components for multi-part names
                name_parts = full_name.split()
                if len(name_parts) > 1:
                    for part in name_parts:
                        if len(part) > 2:  # Skip very short parts
                            if part.lower() not in self.name_components:
                                self.name_components[part.lower()] = []
                            if full_name not in self.name_components[part.lower()]:
                                self.name_components[part.lower()].append(full_name)
        
        logger.info(f"Built name indices: {len(self.name_components)} name components, "
                   f"{len(self.phonetic_map)} phonetic entries")

    def _save_graph(self) -> None:
        """
        Saves the graph structure to disk.
        """
        os.makedirs(os.path.dirname(self.graph_path), exist_ok=True)
        with open(self.graph_path, "w", encoding="utf-8") as f:
            json.dump(self.graph, f, ensure_ascii=False, indent=2)

    def save_knowledge_graph(self) -> None:
        """
        Saves the knowledge graph, entity indices, and name variant maps to disk.
        """
        try:
            self._save_graph()
            
            # Save entity indices
            os.makedirs(os.path.dirname(self.entity_indices_path), exist_ok=True)
            with open(self.entity_indices_path, "w", encoding="utf-8") as f:
                json.dump({"entities": self.entities}, f, ensure_ascii=False, indent=2)
            
            # Save name variant maps
            with open(self.name_variants_path, "w", encoding="utf-8") as f:
                json.dump({
                    "name_components": self.name_components,
                    "phonetic_map": self.phonetic_map
                }, f, ensure_ascii=False, indent=2)
                
            logger.info("Knowledge graph and all indices saved successfully")
        except Exception as e:
            logger.error(f"Error saving knowledge graph: {e}")

    def _extract_entities_from_chunk(self, text: str) -> List[Dict[str, Any]]:
        """
        Extracts entities from a text chunk using transformer-based NER.
        Returns a list of dictionaries with keys 'text' and 'label'.
        """
        entities: List[Dict[str, Any]] = []
        if not text or len(text) < 5:
            return entities
        try:
            doc = self.nlp(text)
            
            # Process standard capitalized text
            for ent in doc.ents:
                if ent.text.strip():
                    entities.append({"text": ent.text.strip(), "label": ent.label_})
            
            # If few entities found and text might contain lowercase names, try another approach
            if len(entities) < 2:
                # Try to find potential person names in lowercase text
                words = text.split()
                for i in range(len(words)-1):
                    # Check for consecutive words that might be names
                    if len(words[i]) > 2 and not words[i].isupper() and not words[i].startswith(('http', 'www')) and not words[i].isdigit():
                        if len(words[i+1]) > 2 and not words[i+1].isupper() and not words[i+1].isdigit():
                            # Potential full name found
                            potential_name = f"{words[i]} {words[i+1]}"
                            
                            # Check if it matches known name patterns
                            if potential_name.lower() in [e.get("text", "").lower() for e in entities]:
                                continue  # Already captured
                                
                            # Check if it matches known name components
                            for part in potential_name.lower().split():
                                if part in self.name_components:
                                    entities.append({
                                        "text": potential_name,
                                        "label": "PER"
                                    })
                                    break
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
        return entities

    def _disambiguate_and_update(self, ent: Dict[str, Any], source: str, context: str) -> str:
        """
        Disambiguates an extracted entity (with keys 'text' and 'label') using context.
        If an existing candidate with similar context is found, updates its occurrences; otherwise, creates a new candidate.
        Enhanced for better handling of personal names.
        
        Returns the unique key for the entity.
        """
        base = ent["text"]
        label = ent["label"]
        threshold = 0.5  # Base threshold
        
        # Adjust threshold based on entity type
        if label == "PER":
            # Lower threshold for person names to improve recall
            threshold = 0.4
            
            # Check for name variations using phonetic matching
            phonetic = get_phonetic_representation(base)
            if phonetic and phonetic in self.phonetic_map:
                potential_matches = self.phonetic_map[phonetic]
                logger.debug(f"Found phonetic matches for '{base}': {potential_matches}")
        
        # Get candidate entities with same or similar base name
        # For person entities, also consider name component matches
        if label == "PER":
            # Get direct base matches
            candidates = [cand for cand in self.entities 
                         if cand.get("label", "") == "PER" and 
                         name_similarity(cand.get("base", ""), base) >= 0.7]
            
            # For single-word names, also check name component index
            if len(base.split()) == 1 and base.lower() in self.name_components:
                full_names = self.name_components[base.lower()]
                for full_name in full_names:
                    component_candidates = [cand for cand in self.entities 
                                          if cand.get("base", "") == full_name]
                    candidates.extend(component_candidates)
        else:
            candidates = [cand for cand in self.entities 
                         if cand.get("label", "") == label and 
                         cand.get("base", "").lower() == base.lower()]
        
        # Call enhanced disambiguation 
        candidate = disambiguate_entity(base, context, candidates, threshold)
        
        if candidate is not None:
            candidate["occurrences"].append({"source": source, "context": context})
            return candidate["key"]
        else:
            new_key = base
            count = 1
            existing_keys = {cand["key"] for cand in self.entities}
            while new_key in existing_keys:
                count += 1
                new_key = f"{base}_{count}"
            
            new_candidate = {
                "key": new_key,
                "base": base,
                "label": label,
                "occurrences": [{"source": source, "context": context}]
            }
            self.entities.append(new_candidate)
            
            # Update name indices for new person entity
            if label == "PER":
                # Add to phonetic map
                phonetic = get_phonetic_representation(base)
                if phonetic:
                    if phonetic not in self.phonetic_map:
                        self.phonetic_map[phonetic] = []
                    self.phonetic_map[phonetic].append(base)
                
                # Add to name component index if multi-part name
                name_parts = base.split()
                if len(name_parts) > 1:
                    for part in name_parts:
                        if len(part) > 2:  # Skip very short parts
                            part_lower = part.lower()
                            if part_lower not in self.name_components:
                                self.name_components[part_lower] = []
                            self.name_components[part_lower].append(base)
            
            return new_key

    def build_from_text(self, text: str, source: str) -> None:
        """
        Updates the knowledge graph with information from a text chunk.
        Extracts entities using the unified NER model, disambiguates them based on context,
        and updates the graph by creating document nodes, entity nodes, and weighted edges.
        Enhanced with better name handling.
        """
        # Process with original capitalization
        doc = self.nlp(text)
        
        # Also process lowercase version to catch potential names missed by NER
        if text.islower():
            # Try with first letters capitalized
            capitalized_text = ' '.join(word.capitalize() for word in text.split())
            capitalized_doc = self.nlp(capitalized_text)
            
            # Merge entities from both runs
            all_entities = []
            seen_texts = set()
            
            for ent in doc.ents:
                all_entities.append(ent)
                seen_texts.add(ent.text.lower())
                
            for ent in capitalized_doc.ents:
                if ent.text.lower() not in seen_texts:
                    all_entities.append(ent)
                    seen_texts.add(ent.text.lower())
        else:
            all_entities = doc.ents
        
        # Now process entities found
        for ent in all_entities:
            context = get_context_for_entity(doc, ent.text)
            entity_key = self._disambiguate_and_update({"text": ent.text, "label": ent.label_}, source, context)
            doc_node_id = f"doc:{source}"
            
            if not any(node.get("id") == doc_node_id for node in self.graph["nodes"]):
                self.graph["nodes"].append({"id": doc_node_id, "source": source, "entities": []})
                
            if not any(node.get("id") == entity_key for node in self.graph["nodes"]):
                self.graph["nodes"].append({"id": entity_key, "base": ent.text, "label": ent.label_})
                
            existing_edge = next(
                (e for e in self.graph["edges"]
                 if (e["source"] == doc_node_id and e["target"] == entity_key) or
                    (e["source"] == entity_key and e["target"] == doc_node_id)),
                None
            )
            
            if existing_edge:
                existing_edge["weight"] += 1
            else:
                self.graph["edges"].append({"source": doc_node_id, "target": entity_key, "weight": 1})
                
            for node in self.graph["nodes"]:
                if node.get("id") == doc_node_id:
                    if entity_key not in node.get("entities", []):
                        node.setdefault("entities", []).append(entity_key)
                    break

    def add_document(self, source: str, text: str) -> None:
        """
        Public method to add a document to the knowledge graph.
        'source' is an identifier (e.g. file path or document ID) and 'text' is the document content.
        """
        self.build_from_text(text, source)
        self.save_knowledge_graph()
        
    def search(self, query_entities: List[str], limit: int = 10) -> Dict[str, Dict[str, float]]:
        """
        Search the knowledge graph for documents related to the query entities.
        Enhanced with better support for name queries.
        
        Args:
            query_entities: List of entity strings to search for
            limit: Maximum number of results to return per entity
            
        Returns:
            Dict mapping entity strings to dicts of {document_source: relevance_score}
        """
        results = {}
        
        # First, normalize query entities (no lower() to preserve proper names)
        normalized_queries = query_entities.copy()
        
        # Check if we have name components that might match
        expanded_queries = []
        for query in normalized_queries:
            query_lower = query.lower()
            
            # Check name component index
            if query_lower in self.name_components:
                for full_name in self.name_components[query_lower]:
                    expanded_queries.append(full_name)
            
            # Check phonetic matches
            phonetic = get_phonetic_representation(query)
            if phonetic and phonetic in self.phonetic_map:
                for matching_name in self.phonetic_map[phonetic]:
                    if matching_name not in expanded_queries:
                        expanded_queries.append(matching_name)
        
        # Add expanded queries to search list
        if expanded_queries:
            logger.info(f"Expanded queries: {query_entities} → {expanded_queries}")
            normalized_queries.extend(expanded_queries)
        
        # Find entities in the graph that match or partially match the query entities
        for entity in self.entities:
            entity_base = entity.get("base", "")
            entity_key = entity.get("key", "")
            
            # For each query, check similarity with this entity
            for query in normalized_queries:
                # Use specialized name matching for person entities
                similarity = 0.0
                if entity.get("label", "") == "PER":
                    similarity = name_similarity(query, entity_base)
                    match = similarity >= 0.7  # Threshold for name match
                else:
                    # For non-person entities use simpler matching
                    exact_match = query.lower() == entity_base.lower()
                    partial_match = query.lower() in entity_base.lower() or entity_base.lower() in query.lower()
                    match = exact_match or partial_match
                
                if match:
                    # Get all documents where this entity appears
                    entity_docs = {}
                    
                    # Look through occurrences
                    for occurrence in entity.get("occurrences", []):
                        source = occurrence.get("source", "")
                        if source:
                            # Set score based on match quality
                            if entity.get("label", "") == "PER":
                                score = min(1.0, similarity)
                            else:
                                score = 1.0 if query.lower() == entity_base.lower() else 0.7
                            entity_docs[source] = score
                    
                    # Find all nodes in the graph connected to this entity
                    for edge in self.graph.get("edges", []):
                        if edge.get("source") == entity_key or edge.get("target") == entity_key:
                            # Find the document node
                            doc_node = edge.get("source") if edge.get("source").startswith("doc:") else edge.get("target") if edge.get("target").startswith("doc:") else None
                            if doc_node:
                                # Extract source from doc node ID (format is "doc:source")
                                source = doc_node[4:]  # Remove "doc:" prefix
                                if source:
                                    # Score based on edge weight
                                    weight = edge.get("weight", 1)
                                    base_score = min(1.0, weight / 10.0)  # Normalize weight to 0-1 range
                                    
                                    if entity.get("label", "") == "PER":
                                        score = base_score * similarity
                                    else:
                                        score = base_score * (1.0 if query.lower() == entity_base.lower() else 0.7)
                                        
                                    if source in entity_docs:
                                        entity_docs[source] = max(entity_docs[source], score)
                                    else:
                                        entity_docs[source] = score
                    
                    # Add entity's documents to results
                    if entity_docs:
                        display_name = entity_base
                        if display_name not in results:
                            results[display_name] = entity_docs
                        else:
                            # Merge documents for this entity
                            for source, score in entity_docs.items():
                                if source in results[display_name]:
                                    results[display_name][source] = max(results[display_name][source], score)
                                else:
                                    results[display_name][source] = score
        
        # If no direct entity matches, try to find related entities
        if not results:
            # Look for documents where query terms appear in context
            for entity in self.entities:
                entity_base = entity.get("base", "")
                # For each query entity, check if it appears in any of this entity's contexts
                for query in query_entities:
                    query_lower = query.lower()
                    for occurrence in entity.get("occurrences", []):
                        context = occurrence.get("context", "").lower()
                        if query_lower in context:
                            # Query appears in this entity's context, add to results
                            source = occurrence.get("source", "")
                            if source:
                                if entity_base not in results:
                                    results[entity_base] = {}
                                results[entity_base][source] = 0.5  # Lower score for context match
        
        # Limit results per entity
        for entity, docs in results.items():
            # Sort documents by score
            sorted_docs = sorted(docs.items(), key=lambda x: x[1], reverse=True)
            # Keep only top 'limit' results
            results[entity] = {k: v for k, v in sorted_docs[:limit]}
        
        return results
