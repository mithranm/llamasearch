"""
Enhanced chunking module with proper multilingual named entity recognition support
using transformer-based models to handle diverse entities across languages.
"""

import os
import re
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Generator, Optional, Tuple

from tqdm import tqdm
from bs4 import BeautifulSoup

# Import transformers library for NER
from transformers.pipelines import pipeline
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForTokenClassification

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

###############################################
# Entity Recognition Components
###############################################

class EntityRecognizer:
    """
    Multilingual named entity recognition using transformer models.
    This class provides advanced NER capabilities to identify entities
    across multiple languages, even when not properly capitalized.
    """
    
    # Default model that works well for multilingual NER with better performance for
    # English, romanized Chinese, and Russian names
    DEFAULT_MODEL = "Babelscape/wikineural-multilingual-ner"
    
    def __init__(self, model_name: Optional[str] = None, device: str = "cpu", use_fast: bool = True):
        """
        Initialize the entity recognizer with a multilingual transformer model.
        
        Args:
            model_name: Hugging Face model name for NER. 
                        Defaults to "Babelscape/wikineural-multilingual-ner"
            device: Device to run inference on ('cpu', 'cuda:0', etc.)
            use_fast: Whether to use the fast tokenizer if available
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self.device = device
        self.use_fast = use_fast
        self.loaded = False
        self.model = None
        self.tokenizer = None
        self.ner_pipeline = None
        self._load_model()
        
        # Name variation handling
        self.name_phonetic_map = {}  # Maps phonetic representations to name variations
        
    def _load_model(self):
        """Load the NER model and tokenizer"""
        try:
            logger.info(f"Loading NER model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, use_fast=self.use_fast
            )
            self.model = AutoModelForTokenClassification.from_pretrained(self.model_name)
            self.ner_pipeline = pipeline(
                "ner", 
                model=self.model, 
                tokenizer=self.tokenizer,
                device=self.device,
                aggregation_strategy="simple", # Merges adjacent tokens of same entity
            )
            self.loaded = True
            logger.info(f"Successfully loaded NER model {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load NER model: {e}")
            self.loaded = False

    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract named entities from text using transformer-based model.
        Enhanced to better handle non-English names.
        
        Args:
            text: Text to extract entities from
            
        Returns:
            List of entity dictionaries with keys: entity_text, entity_type, start, end
        """
        try:
            # Get entities from the NER pipeline
            if self.ner_pipeline is None:
                raise ValueError("NER pipeline is not initialized.")
            
            # Process text as is first
            results = self.ner_pipeline(text)
            entities = []
            
            # Process non-capitalized text separately to catch lowercase names
            # that the model might miss due to capitalization expectations
            if not any(c.isupper() for c in text):
                # Try capitalizing first letters of words for better NER
                cap_text = ' '.join(word.capitalize() if len(word) > 2 else word 
                                  for word in text.split())
                cap_results = self.ner_pipeline(cap_text)
                
                # Merge results, avoiding duplicates
                results = self._merge_ner_results(results, cap_results)
            
            # Convert to a consistent format
            if results is not None:
                for item in results:
                    if isinstance(item, dict):
                        entity_type = item.get("entity_group", "")
                        # Boost confidence for person names
                        score_boost = 0.1 if entity_type == "PER" else 0
                        
                        entities.append({
                            "entity_text": item.get("word"),
                            "entity_type": entity_type,
                            "start": item.get("start"),
                            "end": item.get("end"),
                            "score": item.get("score", 0.5) + score_boost
                        })
            
            # Process for potential name variations
            for entity in entities:
                if entity["entity_type"] == "PER" and " " in entity["entity_text"]:
                    # Extract name components for multi-part names
                    self._add_name_components(entity["entity_text"], entities)
            
            return entities
        except Exception as e:
            logger.error(f"Error in entity extraction: {e}")
            # Fall back to regex-based extraction on error
            return self._extract_entities_regex(text)
    
    def _merge_ner_results(self, results1, results2):
        """Merge two sets of NER results, avoiding duplicates"""
        if not results1:
            return results2
        if not results2:
            return results1
            
        # Create a map of start positions to avoid duplicates
        pos_map = {r.get("start"): r for r in results1}
        
        # Add results from results2 that don't overlap with results1
        for r in results2:
            start = r.get("start")
            end = r.get("end")
            
            # Check if this overlaps with any existing result
            overlap = False
            for s, existing in pos_map.items():
                e = existing.get("end")
                if (start <= s < end) or (start < e <= end) or (s <= start < e):
                    overlap = True
                    # If new result has higher score or is a PER type, prefer it
                    if (r.get("score", 0) > existing.get("score", 0) or 
                        r.get("entity_group") == "PER" and existing.get("entity_group") != "PER"):
                        pos_map[s] = r
                    break
            
            if not overlap:
                results1.append(r)
                
        return results1
    
    def _add_name_components(self, full_name: str, entities_list: List[Dict[str, Any]]):
        """Add individual name components as separate entities with reference to full name"""
        name_parts = full_name.split()
        if len(name_parts) < 2:
            return
            
        for part in name_parts:
            if len(part) > 2:  # Skip very short parts
                # Add as separate entity with reference to full name
                entities_list.append({
                    "entity_text": part,
                    "entity_type": "PER_COMPONENT",
                    "start": -1,  # Not from original text
                    "end": -1,
                    "score": 0.6,  # Lower confidence for components
                    "parent_entity": full_name
                })
    
    def _extract_entities_regex(self, text: str) -> List[Dict[str, Any]]:
        """
        Backup method using regex patterns when transformer model is unavailable.
        Not as accurate as transformer-based NER but provides basic functionality.
        Enhanced to better handle non-English names.
        
        Args:
            text: Text to extract entities from
            
        Returns:
            List of entity dictionaries with keys: entity_text, entity_type, start, end
        """
        entities = []
        
        # Pattern for person names - more flexible to catch non-English names
        # Matches capitalized words possibly followed by more capitalized words
        person_pattern = re.compile(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b')
        
        # Additional pattern for potential names with different capitalization patterns
        # This helps with names that don't follow Western capitalization conventions
        alternate_name_pattern = re.compile(r'\b[A-Za-z][a-z]+(?:\s+[A-Za-z][a-z]+){1,3}\b')
        
        # Find all matches 
        for match in person_pattern.finditer(text):
            entity_text = match.group(0)
            # Skip common words that start with capital letters but aren't likely names
            if entity_text.lower() in {'the', 'a', 'an', 'this', 'that', 'these', 'those', 
                                      'it', 'there', 'their', 'monday', 'tuesday', 'wednesday', 
                                      'thursday', 'friday', 'saturday', 'sunday'}:
                continue
                
            entities.append({
                "entity_text": entity_text,
                "entity_type": "PER",  # Assume person by default
                "start": match.start(),
                "end": match.end(),
                "score": 0.5  # Lower confidence for regex detection
            })
        
        # Process alternate pattern only if we didn't find standard capitalized names
        if len(entities) == 0:
            for match in alternate_name_pattern.finditer(text):
                entity_text = match.group(0)
                
                # Skip common words
                if entity_text.lower() in {'the', 'a', 'an', 'this', 'that', 'these', 'those',
                                         'it', 'there', 'their', 'monday', 'tuesday', 'wednesday',
                                         'thursday', 'friday', 'saturday', 'sunday', 'january',
                                         'february', 'march', 'april', 'may', 'june', 'july',
                                         'august', 'september', 'october', 'november', 'december'}:
                    continue
                
                # Check if not all lowercase - more likely to be a name
                if not entity_text.islower() or len(entity_text.split()) > 1:
                    entities.append({
                        "entity_text": entity_text,
                        "entity_type": "PER",
                        "start": match.start(),
                        "end": match.end(),
                        "score": 0.4  # Even lower confidence
                    })
                # For all lowercase text, only add if multiple words (more likely to be a name)
                elif len(entity_text.split()) > 1:
                    entities.append({
                        "entity_text": entity_text,
                        "entity_type": "PER",
                        "start": match.start(),
                        "end": match.end(),
                        "score": 0.3  # Lowest confidence
                    })
            
        return entities

###############################################
# Abstract Base and Common Utility Methods
###############################################

class BaseChunker(ABC):
    """
    Abstract base class for document chunking.
    Defines common functionality and interface for chunkers.
    Enhanced with transformer-based entity recognition.
    """

    def __init__(
        self,
        min_chunk_size: int = 250,
        max_chunk_size: int = 500,
        overlap_size: int = 100,
        batch_size: int = 5,
        debug_output: bool = False,
        num_workers: int = 1,
        ner_model: Optional[str] = None,
        device: str = "cpu",
    ):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.batch_size = batch_size
        self.overlap_size = overlap_size
        self.debug_output = debug_output
        self.num_workers = num_workers
        self.device = device
        
        if self.overlap_size >= self.max_chunk_size:
            raise ValueError("overlap_size must be less than chunk_size")
        if self.max_chunk_size <= 0:
            raise ValueError("chunk_size must be non-zero and positive")
        if self.min_chunk_size is not None:
            if self.max_chunk_size < self.min_chunk_size:
                raise ValueError("chunk_size must be greater than min_chunk_size")
            if self.min_chunk_size < 0:
                raise ValueError("min_chunk_size must be non-negative")
        else:
            logger.warning("min_chunk_size is not set, using default value of 100")
            self.min_chunk_size = 100
        
        # Initialize entity recognizer using transformer model
        self.entity_recognizer = EntityRecognizer(ner_model, device=device)
        
        # Precompile common regex patterns for noise detection
        self.nav_patterns = re.compile(r'(menu|navigation|navbar|breadcrumb|footer|header)', re.IGNORECASE)
        self.noise_patterns = re.compile(r'(advertisement|sponsored)', re.IGNORECASE)
        self.duplicate_patterns = re.compile(r'(duplicate content|copied content)', re.IGNORECASE)
        
        # Log chunking configuration
        logger.info(f"Initialized chunker with max size={self.max_chunk_size}, "
                f"min_size={self.min_chunk_size}, overlap={self.overlap_size}, "
                f"NER model={ner_model or EntityRecognizer.DEFAULT_MODEL}")

    @abstractmethod
    def _convert_to_html(self, text: str) -> str:
        """
        Convert input text to HTML.
        For Markdown files, this uses a Markdown converter.
        For HTML files, this returns the text unchanged.
        """
        pass

    @abstractmethod
    def _get_supported_extensions(self) -> List[str]:
        """
        Return a list of file extensions supported by this chunker.
        """
        pass

    @abstractmethod
    def chunk_document(self, text: str) -> Generator[Dict[str, Any], None, None]:
        """
        Process input text and yield chunk dictionaries.
        Each yielded dict should include at least:
          - 'chunk': the text chunk,
          - 'metadata': additional info (may be empty),
          - 'embedding_text': text to use for embeddings.
        """
        pass

    # --- Utility Methods ---

    def _clean_text(self, text: str) -> str:
        """Clean text by removing noise, boilerplate, and normalizing whitespace."""
        text = self.duplicate_patterns.sub('', text)
        text = self.noise_patterns.sub('', text)
        text = re.sub(r'https?://\S+|www\.\S+|[\w\.-]+@[\w\.-]+', '', text)
        text = re.sub(r'[\r\n\t]+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\s*([,.!?;])\s*', r'\1 ', text)
        return text.strip()

    def _split_text(self, text: str) -> Generator[Dict[str, Any], None, None]:
        """
        Entity-aware text splitter with support for multilingual NER.
        Preserves entities within chunks where possible and tracks entity information.
        """
        if not text:
            return
        
        # Extract entities using the transformer model
        entities = self.entity_recognizer.extract_entities(text)
        entity_texts = [e["entity_text"] for e in entities]
        
        # Find section headers (e.g., "By the numbers:")
        section_headers = []
        section_pattern = re.compile(r'\*\*([^*]+):\*\*')
        for match in section_pattern.finditer(text):
            section_headers.append((match.group(1), match.start(), match.end()))
        
        # Split text into paragraphs
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
        
        # Initialize processing
        buffer = ""
        current_entities = []
        
        for para in paragraphs:
            # Check if this paragraph starts a new section
            is_section_header = any(header[0] in para for header in section_headers)
            
            # Check if this paragraph contains entities
            para_entities = [e for e in entities if e["entity_text"] in para]
            contains_entities = len(para_entities) > 0
            
            # Potential new buffer if we add this paragraph
            if buffer:
                candidate = buffer + "\n" + para
            else:
                candidate = para
                
            # If paragraph contains important entities or starts a section and buffer is already large enough
            if (is_section_header or contains_entities) and len(buffer) >= self.min_chunk_size:
                yield {
                    "chunk": buffer, 
                    "metadata": {"entities": [e["entity_text"] for e in current_entities]}, 
                    "embedding_text": buffer
                }
                buffer = para
                current_entities = para_entities
                continue
                
            # If adding would exceed max size
            if len(candidate) > self.max_chunk_size:
                # Yield current buffer if it's not empty
                if buffer and len(buffer) >= self.min_chunk_size:
                    yield {
                        "chunk": buffer, 
                        "metadata": {"entities": [e["entity_text"] for e in current_entities]}, 
                        "embedding_text": buffer
                    }
                
                # Handle long paragraphs
                if len(para) > self.max_chunk_size:
                    # If contains entities, try to keep them intact
                    if contains_entities:
                        # If paragraph is only slightly over limit, keep it whole to preserve entities
                        if len(para) <= self.max_chunk_size * 1.2:
                            yield {
                                "chunk": para, 
                                "metadata": {"entities": [e["entity_text"] for e in para_entities]}, 
                                "embedding_text": para
                            }
                        else:
                            # Smart split around entity boundaries
                            chunks = self._smart_split_with_entities(para, para_entities)
                            for chunk, chunk_entities in chunks:
                                if len(chunk) >= self.min_chunk_size:
                                    yield {
                                        "chunk": chunk, 
                                        "metadata": {"entities": [e["entity_text"] for e in chunk_entities]}, 
                                        "embedding_text": chunk
                                    }
                    else:
                        # For paragraphs without entities, use sliding window
                        start = 0
                        while start < len(para):
                            end = start + self.max_chunk_size
                            chunk = para[start:end].strip()
                            if chunk and len(chunk) >= self.min_chunk_size:
                                # Check for entities in this chunk
                                chunk_entities = [e for e in entities 
                                                if e["start"] >= start and e["end"] <= end]
                                yield {
                                    "chunk": chunk, 
                                    "metadata": {"entities": [e["entity_text"] for e in chunk_entities]}, 
                                    "embedding_text": chunk
                                }
                            start = end - self.overlap_size
                
                # Reset buffer after processing long paragraph
                buffer = ""
                current_entities = []
            else:
                # If we can add to buffer without exceeding limits
                buffer = candidate
                current_entities.extend(para_entities)
        
        # Don't forget any remaining content in buffer
        if buffer and len(buffer) >= self.min_chunk_size:
            yield {
                "chunk": buffer, 
                "metadata": {"entities": [e["entity_text"] for e in current_entities]}, 
                "embedding_text": buffer
            }

    def _smart_split_with_entities(
        self, text: str, entities: List[Dict[str, Any]]
    ) -> List[Tuple[str, List[Dict[str, Any]]]]:
        """
        Split a long text while preserving entity boundaries where possible.
        
        Args:
            text: The text to split
            entities: List of entity dictionaries with start/end positions
            
        Returns:
            List of (chunk_text, chunk_entities) tuples
        """
        if not entities:
            # If no entities, just do regular chunking
            chunks = []
            start = 0
            while start < len(text):
                end = start + self.max_chunk_size
                chunk = text[start:end].strip()
                if chunk:
                    chunks.append((chunk, []))
                start = end - self.overlap_size
            return chunks
            
        # Sort entities by position
        sorted_entities = sorted(entities, key=lambda x: x["start"])
        
        chunks = []
        start = 0
        while start < len(text):
            # Default end position based on chunk size
            proposed_end = start + self.max_chunk_size
            
            # Find entities that would be split by the proposed boundary
            crossing_entities = [e for e in sorted_entities 
                               if e["start"] < proposed_end and e["end"] > proposed_end]
            
            # Adjust end position to avoid splitting entities
            if crossing_entities:
                # If entities cross boundary, move end to before the first crossing entity
                end = min(e["start"] for e in crossing_entities)
                # If this would create a tiny chunk, move end to after all crossing entities
                if end - start < self.min_chunk_size:
                    end = max(e["end"] for e in crossing_entities)
            else:
                # If no entities cross, use proposed end
                end = proposed_end
                
            # Ensure we're making progress even if there are constraints
            if end <= start:
                # If stuck, just use max_chunk_size
                end = start + self.max_chunk_size
                
            # Extract the chunk and its entities
            chunk = text[start:end].strip()
            chunk_entities = [e for e in sorted_entities 
                             if e["start"] >= start and e["end"] <= end]
            
            if chunk:
                chunks.append((chunk, chunk_entities))
                
            # Move start position for next chunk with overlap
            start = end - self.overlap_size
            
        return chunks

    # --- File Processing Methods ---

    def process_file(self, file_path: str, show_progress: bool = True) -> Generator[Dict[str, Any], None, None]:
        """
        Read a file and yield chunk dictionaries produced by chunk_document().
        If a metadata comment is present at the top, parse it and attach its "source" field to each chunk's metadata.
        """
        ext = Path(file_path).suffix.lower()
        if ext not in self._get_supported_extensions():
            raise ValueError(f"Unsupported file type: {ext}")
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            return

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        metadata = {}
        metadata_pattern = r"^<!--\s*METADATA:\s*(\{.*?\})\s*-->"
        m = re.match(metadata_pattern, text, re.DOTALL)
        if m:
            try:
                metadata = json.loads(m.group(1))
            except Exception as e:
                logger.warning(f"Error parsing metadata in {file_path}: {e}")
            text = text[m.end():].strip()

        logger.info(f"Processing {file_path} ({len(text)} chars)")
        chunk_count = 0
        pbar = tqdm(total=max(1, len(text) // (self.max_chunk_size // 2)), 
                   desc=f"Chunking {os.path.basename(file_path)}", 
                   unit="chunk") if show_progress else None
                   
        for chunk in self.chunk_document(text):
            if "metadata" not in chunk:
                chunk["metadata"] = {}
            chunk["metadata"]["source"] = metadata.get("source", file_path)
            yield chunk
            chunk_count += 1
            if pbar:
                pbar.update(1)
        if pbar:
            pbar.close()
        logger.info(f"Finished chunking {file_path}: {chunk_count} chunks.")

    def process_file_in_batches(self, file_path: str, show_progress: bool = True) -> Generator[List[Dict[str, Any]], None, None]:
        """
        Process a file and yield batches (lists) of chunk dictionaries.
        """
        batch_size = self.batch_size
        all_chunks = list(self.process_file(file_path, show_progress=show_progress))
        out_batch: List[Dict[str, Any]] = []
        for chunk in all_chunks:
            out_batch.append(chunk)
            if len(out_batch) >= batch_size:
                yield out_batch
                out_batch = []
        if out_batch:
            yield out_batch

    def process_files_in_parallel(self, file_paths: List[str], show_progress: bool = True) -> Dict[str, List[Dict[str, Any]]]:
        """
        Process multiple files in parallel.
        Returns a dict mapping file paths to lists of chunk dictionaries.
        """
        valid = [fp for fp in file_paths if Path(fp).suffix.lower() in self._get_supported_extensions() and os.path.exists(fp)]
        if not valid:
            logger.warning(f"No valid files to process in parallel for {self.__class__.__name__}")
            return {}
        results: Dict[str, List[Dict[str, Any]]] = {}
        if len(valid) == 1 or self.num_workers < 2:
            for fp in valid:
                results[fp] = list(self.process_file(fp, show_progress=show_progress))
            return results
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {executor.submit(self.process_file, fp, False): fp for fp in valid}
            pbar = tqdm(total=len(valid), desc="Parallel chunking", unit="file") if show_progress else None
            for future in futures:
                fp = futures[future]
                try:
                    chunks = list(future.result())
                    results[fp] = chunks
                except Exception as e:
                    logger.error(f"Error processing {fp}: {e}")
                    results[fp] = []
                if pbar:
                    pbar.update(1)
                    pbar.set_postfix({"chunks": len(results[fp])})
            if pbar:
                pbar.close()
        return results

# --- Concrete Chunker Implementations ---

class MarkdownChunker(BaseChunker):
    """
    Chunker for Markdown files with enhanced multilingual NER support.
    Converts Markdown to HTML, extracts visible text, then splits it.
    """
    def _convert_to_html(self, text: str) -> str:
        import markdown
        return markdown.markdown(text, extensions=["fenced_code", "tables", "attr_list"])

    def _get_supported_extensions(self) -> List[str]:
        return [".md"]

    def chunk_document(self, text: str) -> Generator[Dict[str, Any], None, None]:
        html = self._convert_to_html(text)
        soup = BeautifulSoup(html, "html.parser")
        plain_text = soup.get_text(separator="\n")
        plain_text = self._clean_text(plain_text)
        yield from self._split_text(plain_text)

class HtmlChunker(BaseChunker):
    """
    Chunker for HTML files with enhanced multilingual NER support.
    Uses the DOM structure to extract coherent text blocks from the main content area.
    """
    # Extended selectors to capture main content across diverse websites.
    MAIN_SELECTORS = ["article", "main", "#content", ".content", ".main-content", "#main", "#primary", ".mw-parser-output"]

    def _convert_to_html(self, text: str) -> str:
        # Input text is already HTML
        return text

    def _get_supported_extensions(self) -> List[str]:
        return [".html", ".htm"]

    def _extract_plain_text(self, html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        # Remove boilerplate tags that are likely not part of the main content
        for tag in soup.find_all(["script", "style", "nav", "header", "footer", "aside"]):
            tag.decompose()
        # Try using one of the extended main selectors
        for selector in self.MAIN_SELECTORS:
            container = soup.select_one(selector)
            if container:
                text = container.get_text(separator="\n", strip=True)
                if len(text) >= self.min_chunk_size:
                    if "\n" not in text:
                        text = re.sub(r'\. ', '.\n', text)
                    return text
        # Fallback: get all visible text
        text = soup.get_text(separator="\n", strip=True)
        if "\n" not in text:
            text = re.sub(r'\. ', '.\n', text)
        return text

    def chunk_document(self, text: str) -> Generator[Dict[str, Any], None, None]:
        html = self._convert_to_html(text)
        plain_text = self._extract_plain_text(html)
        plain_text = self._clean_text(plain_text)
        yield from self._split_text(plain_text)

# --- Module-level Helper Function ---

def process_directory(markdown_chunker: MarkdownChunker, html_chunker: HtmlChunker, directory_path: str, debug: bool = False) -> Dict[str, List[Dict[str, Any]]]:
    """
    Process all supported files in a directory.
    Returns a dict mapping file paths to lists of chunk dictionaries.
    This version is case-insensitive: it accepts .md, .html, or .htm in any case.
    """
    from pathlib import Path
    p = Path(directory_path)
    logger.info(f"Processing directory: {directory_path}")
    markdown_files = [x for x in p.glob("**/*") if x.suffix.lower() in {".md"}]
    html_files = [x for x in p.glob("**/*") if x.suffix.lower() in {".html", ".htm"}]
    markdown_file_paths = [str(x) for x in markdown_files]
    html_file_paths = [str(x) for x in html_files]
    if not markdown_file_paths and not html_file_paths:
        logger.warning(f"No supported files found in {directory_path}")
        return {}
    markdown_results = markdown_chunker.process_files_in_parallel(file_paths=markdown_file_paths, show_progress=not debug)
    html_results = html_chunker.process_files_in_parallel(file_paths=html_file_paths, show_progress=not debug)
    results = {}
    results.update(markdown_results)
    results.update(html_results)
    if debug:
        logger.debug(f"Processed {len(results)} files in {directory_path}")
    else:
        logger.info(f"Processed {len(results)} files in {directory_path}")
    return results