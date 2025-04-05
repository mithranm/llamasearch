# llamasearch/core/knowledge_graph.py

import logging
import spacy
import sys
import subprocess
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

def jaccard_similarity(text1: str, text2: str) -> float:
    """Compute the Jaccard similarity between two texts."""
    tokens1 = set(text1.lower().split())
    tokens2 = set(text2.lower().split())
    if not tokens1 or not tokens2:
        return 0.0
    return len(tokens1.intersection(tokens2)) / len(tokens1.union(tokens2))


class KnowledgeGraph:
    """
    Builds a knowledge graph from text using spaCy NER.
    For each recognized entity, it stores the entity's label,
    a list of occurrences (each with a sentence and context file),
    and co-mention information.
    
    This version uses a disambiguation method so that if two entities
    share the same surface form (e.g. "Sarah") but appear in different contexts,
    they will be stored separately.
    """

    def __init__(self) -> None:
        self.entities: Dict[str, Dict[str, Any]] = {}
        self.nlp: Optional[spacy.language.Language] = None
        self._init_spacy()

    def _init_spacy(self) -> None:
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("Loaded spaCy model successfully")
        except OSError:
            logger.warning("SpaCy model not found. Attempting to install...")
            # Try to install model and reload
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

    def _disambiguate(self, base_name: str, sent_text: str, threshold: float = 0.5) -> str:
        """
        Given an entity's base name and a sentence, determine whether this occurrence
        matches an existing entity instance (using Jaccard similarity on the occurrence sentence).
        If a match is found (similarity >= threshold), return its key;
        otherwise, generate a new disambiguated key.
        """
        candidates = [k for k in self.entities if k.startswith(base_name)]
        for cand in candidates:
            for occ in self.entities[cand]["occurrences"]:
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

    def build_from_text(self, text: str, context_file: str = "") -> None:
        """
        Build (or update) the knowledge graph from the provided text.
        Each sentence is processed using spaCy; for each entity occurrence,
        we disambiguate based on its context (sentence) and then record the occurrence.
        Co-mentions are then added between entities in the same sentence.
        """
        if self.nlp is None:
            raise RuntimeError("SpaCy NLP model is not loaded.")
        doc = self.nlp(text)
        for sent in doc.sents:
            sent_text: str = sent.text.strip()
            disamb_entities: List[str] = []
            # Process each recognized entity in the sentence
            for ent in sent.ents:
                ent_text = ent.text.strip()
                if not ent_text:
                    continue
                # Use the disambiguation routine
                key = self._disambiguate(ent_text, sent_text)
                if key not in self.entities:
                    self.entities[key] = {
                        "label": ent.label_,
                        "occurrences": [],
                        "co_mentions": {}
                    }
                self.entities[key]["occurrences"].append({
                    "sentence": sent_text,
                    "context_file": context_file
                })
                disamb_entities.append(key)
            # Record co-mentions for disambiguated entity keys
            for i in range(len(disamb_entities)):
                for j in range(i + 1, len(disamb_entities)):
                    e1 = disamb_entities[i]
                    e2 = disamb_entities[j]
                    if e2 not in self.entities[e1]["co_mentions"]:
                        self.entities[e1]["co_mentions"][e2] = []
                    self.entities[e1]["co_mentions"][e2].append({
                        "sentence": sent_text,
                        "context_file": context_file
                    })
                    if e1 not in self.entities[e2]["co_mentions"]:
                        self.entities[e2]["co_mentions"][e1] = []
                    self.entities[e2]["co_mentions"][e1].append({
                        "sentence": sent_text,
                        "context_file": context_file
                    })

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
