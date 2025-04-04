# llamasearch/core/knowledge_graph.py

import logging
import spacy
import sys
import site
import os
import subprocess

logger = logging.getLogger(__name__)


class KnowledgeGraph:
    """
    Builds a knowledge graph from text using spaCy NER.
    For each recognized entity, store label, occurrences, co-mentions, etc.
    No hardcoding of specific fields or staff names.
    """

    def __init__(self):
        self.entities = {}
        self._init_spacy()

    def _init_spacy(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("Loaded spaCy model successfully")
        except OSError:
            logger.warning("SpaCy model not found. Attempting to install...")

            # Get the user site-packages directory for installation
            user_site = site.getusersitepackages()

            try:
                # Install the model with python -m command to ensure proper installation
                subprocess.run(
                    [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
                    check=True,
                )

                logger.info("SpaCy model installed. Attempting to load...")

                # Try loading again after installation
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                    logger.info("Successfully loaded spaCy model after installation")
                except OSError:
                    # If still can't load, try to import directly as a module
                    logger.warning(
                        "Still can't load model. Trying alternative methods..."
                    )

                    # Method 1: Try to import as a module and use load()
                    try:
                        import en_core_web_sm

                        self.nlp = en_core_web_sm.load()
                        logger.info("Loaded spaCy model via direct module import")
                    except (ImportError, AttributeError):
                        # Method 2: Try to locate the model in user site-packages
                        try:
                            # Look for the model in the user site-packages
                            model_path = os.path.join(user_site, "en_core_web_sm")
                            if os.path.exists(model_path):
                                self.nlp = spacy.load(model_path)
                                logger.info(
                                    f"Loaded spaCy model from path: {model_path}"
                                )
                            else:
                                # Try to find the model elsewhere in sys.path
                                for path in sys.path:
                                    potential_path = os.path.join(
                                        path, "en_core_web_sm"
                                    )
                                    if os.path.exists(potential_path):
                                        self.nlp = spacy.load(potential_path)
                                        logger.info(
                                            f"Loaded spaCy model from path: {potential_path}"
                                        )
                                        break
                                else:
                                    # If we couldn't find the model, use a blank model as fallback
                                    logger.warning(
                                        "Using blank spaCy model as fallback"
                                    )
                                    self.nlp = spacy.blank("en")
                        except Exception as e:
                            logger.error(f"Error loading model from paths: {e}")
                            logger.warning("Using blank spaCy model as fallback")
                            self.nlp = spacy.blank("en")
            except Exception as e:
                logger.error(f"Error during model installation: {e}")
                # Create a minimal fallback for basic functionality
                logger.warning("Using blank spaCy model as fallback")
                self.nlp = spacy.blank("en")

    def build_from_text(self, text: str, context_file: str = ""):
        doc = self.nlp(text)
        for sent in doc.sents:
            sent_text = sent.text.strip()
            entlist = []
            for ent in sent.ents:
                ent_str = ent.text.strip()
                ent_label = ent.label_
                if not ent_str:
                    continue
                if ent_str not in self.entities:
                    self.entities[ent_str] = {
                        "label": ent_label,
                        "occurrences": [],
                        "co_mentions": {},
                    }
                # record occurrence
                self.entities[ent_str]["occurrences"].append(
                    {"sentence": sent_text, "context_file": context_file}
                )
                entlist.append(ent_str)
            # co-occurrences
            for i in range(len(entlist)):
                for j in range(i + 1, len(entlist)):
                    e1 = entlist[i]
                    e2 = entlist[j]
                    if e2 not in self.entities[e1]["co_mentions"]:
                        self.entities[e1]["co_mentions"][e2] = []
                    self.entities[e1]["co_mentions"][e2].append(
                        {"sentence": sent_text, "context_file": context_file}
                    )
                    if e1 not in self.entities[e2]["co_mentions"]:
                        self.entities[e2]["co_mentions"][e1] = []
                    self.entities[e2]["co_mentions"][e1].append(
                        {"sentence": sent_text, "context_file": context_file}
                    )

    def get_entity_info(self, ent_str: str):
        return self.entities.get(ent_str)

    def get_all_entities(self):
        return list(self.entities.keys())

    def get_co_mentions(self, ent_str: str):
        e = self.entities.get(ent_str)
        if not e:
            return {}
        return e["co_mentions"]
