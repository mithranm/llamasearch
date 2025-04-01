# llamasearch/core/knowledge_graph.py

import logging
import spacy

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
        except OSError:
            import subprocess
            subprocess.run(["python","-m","spacy","download","en_core_web_sm"], check=True)
            self.nlp = spacy.load("en_core_web_sm")

    def build_from_text(self, text:str, context_file:str=None):
        doc= self.nlp(text)
        for sent in doc.sents:
            sent_text= sent.text.strip()
            entlist=[]
            for ent in sent.ents:
                ent_str=ent.text.strip()
                ent_label= ent.label_
                if not ent_str:
                    continue
                if ent_str not in self.entities:
                    self.entities[ent_str]={
                        "label": ent_label,
                        "occurrences": [],
                        "co_mentions": {}
                    }
                # record occurrence
                self.entities[ent_str]["occurrences"].append({
                    "sentence": sent_text,
                    "context_file": context_file
                })
                entlist.append(ent_str)
            # co-occurrences
            for i in range(len(entlist)):
                for j in range(i+1, len(entlist)):
                    e1=entlist[i]
                    e2=entlist[j]
                    if e2 not in self.entities[e1]["co_mentions"]:
                        self.entities[e1]["co_mentions"][e2]=[]
                    self.entities[e1]["co_mentions"][e2].append({
                        "sentence": sent_text,
                        "context_file": context_file
                    })
                    if e1 not in self.entities[e2]["co_mentions"]:
                        self.entities[e2]["co_mentions"][e1]=[]
                    self.entities[e2]["co_mentions"][e1].append({
                        "sentence": sent_text,
                        "context_file": context_file
                    })

    def get_entity_info(self, ent_str:str):
        return self.entities.get(ent_str)

    def get_all_entities(self):
        return list(self.entities.keys())

    def get_co_mentions(self, ent_str:str):
        e= self.entities.get(ent_str)
        if not e:
            return {}
        return e["co_mentions"]
