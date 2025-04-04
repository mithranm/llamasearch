# llamasearch/core/enhanced_chunker.py

import os
import re
import json
import logging
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Generator, Tuple, Optional
from pathlib import Path
from tqdm import tqdm
import markdown
from bs4 import BeautifulSoup, Tag
from bs4.element import NavigableString, PageElement

from .resource_manager import get_resource_manager

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_BATCH_SIZE = 100  # Default batch size for chunking

class BaseChunker(ABC):
    """
    Abstract base class for document chunking.
    Defines common functionality and interface for different chunker implementations.
    """
    
    def _is_navigation_element(self, tag: Tag) -> bool:
        """Check if a tag is likely a navigation element."""
        if not isinstance(tag, Tag):
            return False
            
        # Check tag name
        if tag.name in ['nav', 'header', 'footer']:
            return True
            
        # Check classes and IDs
        classes = tag.get('class', None)
        tag_id = tag.get('id', None)
        
        # Handle class attribute
        class_str = ''
        if classes:
            if isinstance(classes, (list, tuple)):
                class_str = ' '.join(str(c) for c in classes)
            else:
                class_str = str(classes)
                
        # Handle ID attribute
        id_str = ''
        if tag_id:
            if isinstance(tag_id, (list, tuple)):
                id_str = ' '.join(str(id) for id in tag_id)
            else:
                id_str = str(tag_id)
            
        # Check if class matches navigation patterns
        if class_str and self.nav_patterns.search(class_str):
            return True
            
        # Check if ID matches navigation patterns
        if id_str and self.nav_patterns.search(id_str):
            return True
            
        # Check role attribute
        role = tag.get('role', None)
        if role and str(role) in ['navigation', 'menubar', 'toolbar']:
            return True
            
        return False

    def _similarity_ratio(self, text1: str, text2: str) -> float:
        """Calculate similarity ratio between two texts using character-based comparison."""
        # Normalize texts
        text1_norm = ' '.join(re.findall(r'\w+', text1.lower()))
        text2_norm = ' '.join(re.findall(r'\w+', text2.lower()))
        
        # Get lengths
        len1, len2 = len(text1_norm), len(text2_norm)
        if len1 == 0 or len2 == 0:
            return 0.0
            
        # Use character-based similarity
        common = sum(1 for i in range(min(len1, len2)) if text1_norm[i] == text2_norm[i])
        return 2 * common / (len1 + len2)

    def _clean_text(self, text: str) -> str:
        """Clean text content by removing noise, boilerplate, and normalizing whitespace."""
        # Remove duplicate/noisy content
        text = self.duplicate_patterns.sub('', text)
        text = self.noise_patterns.sub('', text)

        # Remove common boilerplate text
        text = re.sub(r'(?i)(click here|learn more about|read more about|find out more about|explore|discover)', '', text)
        text = re.sub(r'(?i)(for more information|contact us|get in touch|call us|email us)', '', text)
        
        # Remove URLs and email addresses
        text = re.sub(r'https?://\S+|www\.\S+|[\w\.-]+@[\w\.-]+', '', text)
        
        # Remove special characters and normalize whitespace
        text = re.sub(r'[\r\n\t]+', ' ', text)  # Replace newlines and tabs with space
        text = re.sub(r'\s+', ' ', text)  # Normalize multiple spaces
        text = re.sub(r'\s*([,.!?;])\s*', r'\1 ', text)  # Normalize punctuation spacing
        
        return text.strip()

    def __init__(
        self,
        chunk_size: int = 150,
        text_embedding_size: int = 512,
        min_chunk_size: int = 50,
        max_chunks: int = 5000,
        batch_size: Optional[int] = None,
        ignore_link_urls: bool = True,
        code_context_window: int = 2,
        include_section_headers: bool = True,
        always_create_chunks: bool = True,
        overlap_size: int = 50,
        semantic_headers_only: bool = True,
        min_section_length: int = 100,
        num_workers: Optional[int] = None,
        auto_optimize: bool = True,
        debug_output: bool = False,
    ):
        self.chunk_size = chunk_size
        self.text_embedding_size = text_embedding_size
        self.min_chunk_size = min_chunk_size
        self.max_chunks = max_chunks
        self.batch_size = batch_size or DEFAULT_CHUNK_BATCH_SIZE
        self.ignore_link_urls = ignore_link_urls
        self.code_context_window = code_context_window
        self.include_section_headers = include_section_headers
        self.always_create_chunks = always_create_chunks
        self.overlap_size = overlap_size
        self.semantic_headers_only = semantic_headers_only
        self.min_section_length = min_section_length
        self.debug_output = debug_output
        self.debug_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'debug')
        if self.debug_output:
            os.makedirs(self.debug_dir, exist_ok=True)

        # Common patterns to filter out navigation, metadata, and noise
        self.nav_patterns = re.compile(
            r'(menu|navigation|navbar|breadcrumb|footer|header|copyright|social|share|related|popular|trending|categories|tags|search)',
            re.IGNORECASE
        )
        self.metadata_patterns = re.compile(
            r'(posted on|published|last updated|author|comments|views|likes|share this|follow us|subscribe)',
            re.IGNORECASE
        )
        self.duplicate_patterns = re.compile(
            r'(this content appears in multiple locations|similar content|duplicate content|copied content)',
            re.IGNORECASE
        )
        self.noise_patterns = re.compile(
            r'(advertisement|sponsored content|loading|please wait|cookies|privacy policy)',
            re.IGNORECASE
        )

        # Setup resource management and workers
        self.resource_manager = get_resource_manager(auto_optimize=auto_optimize)
        if num_workers is None and auto_optimize:
            ex = self.resource_manager.get_executor("io")
            if isinstance(ex, ThreadPoolExecutor):
                self.num_workers = ex._max_workers
            else:
                self.num_workers = 4
        else:
            self.num_workers = num_workers or 4

        # Initialize NLP if needed
        self.nlp = None
        if self.semantic_headers_only:
            try:
                import spacy
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("Using spaCy for semantic analysis.")
            except (ImportError, OSError):
                logger.warning("spaCy not available. Overlap logic will be simpler.")

        logger.info(
            f"Initialized {self.__class__.__name__} with chunk_size={chunk_size}, "
            f"overlap_size={overlap_size}, workers={self.num_workers}"
        )

    @abstractmethod
    def _convert_to_html(self, text: str) -> str:
        """Convert input text to HTML for processing."""
        pass

    @abstractmethod
    def _get_supported_extensions(self) -> List[str]:
        """Return list of file extensions this chunker supports."""
        pass

    def _remove_navigation_elements(self, soup: BeautifulSoup) -> None:
        """Remove common navigation and metadata elements from HTML."""
        for tag in soup.find_all(True):  # True means match any tag
            if not isinstance(tag, Tag):
                continue

            # Skip tags without attributes
            if not tag.attrs:
                continue

            # Check all attribute values
            should_remove = False
            for attr_name, attr_value in tag.attrs.items():
                # Handle class lists specially
                if attr_name == 'class':
                    if any(cls for cls in attr_value if self.nav_patterns.search(str(cls))):
                        should_remove = True
                        break
                # Handle other attributes
                elif self.nav_patterns.search(str(attr_value)):
                    should_remove = True
                    break

            if should_remove:
                tag.decompose()

    def _extract_code_blocks(self, html: str) -> Tuple[str, List[Dict[str, Any]]]:
        """Extract code blocks from HTML content."""
        soup = BeautifulSoup(html, 'html.parser')
        code_blocks: List[Dict[str, Any]] = []
        
        for pre in soup.find_all(['pre', 'code']):
            if not isinstance(pre, Tag):
                continue
                
            code = {
                'type': 'code',
                'content': pre.get_text(),
                'language': 'text',
                'context': []  # Initialize as empty list
            }
            
            # Get language from class if available
            if isinstance(pre, Tag):  # Ensure we're working with a Tag
                classes = pre.get('class')
                if classes is not None and len(classes) > 0:
                    code['language'] = str(classes[0])
            
            # Get surrounding context
            context_items: List[str] = []
            current: Optional[PageElement] = pre
            for _ in range(self.code_context_window):
                if not current or not current.previous_sibling:
                    break
                current = current.previous_sibling
                if isinstance(current, Tag):
                    context_items.append(current.get_text().strip())
            
            if context_items:
                code['context'] = context_items
            
            code_blocks.append(code)
            pre.decompose()
            
        return str(soup), code_blocks

    def _is_semantic_header(self, tag: Tag) -> bool:
        """Check if a tag represents a semantic header."""
        if not tag.name or not tag.name.startswith('h'):
            return False
            
        text = tag.get_text().strip()
        if not text:
            return False
            
        if not self.semantic_headers_only:
            return True
            
        if self.nlp:
            doc = self.nlp(text)
            return any(token.pos_ in ['NOUN', 'PROPN', 'VERB'] for token in doc)
            
        return True  # Fallback if no NLP

    def _extract_sections(self, html: str) -> List[Dict[str, Any]]:
        """Extract sections from HTML using semantic hints or headers."""
        soup = BeautifulSoup(html, 'html.parser')
        self._remove_navigation_elements(soup)
        
        sections: List[Dict[str, Any]] = []
        current_section = {'title': '', 'content': '', 'level': 0}
        
        for elem in soup.descendants:
            if isinstance(elem, NavigableString):
                if current_section['content'] or str(elem).strip():
                    current_section['content'] += str(elem)
            elif isinstance(elem, Tag):
                if elem.name and elem.name.startswith('h') and self._is_semantic_header(elem):
                    if current_section['content'].strip():
                        sections.append(dict(current_section))
                    current_section = {
                        'title': elem.get_text().strip(),
                        'content': '',
                        'level': int(elem.name[1])
                    }
                elif elem.name in ['p', 'div', 'section']:
                    current_section['content'] += elem.get_text() + '\n'
                    
        if current_section['content'].strip():
            sections.append(dict(current_section))
            
        return sections

    def _split_with_semantic_overlap(self, text: str) -> List[Tuple[str, bool, bool]]:
        """Split text into chunks with semantic overlap."""
        if self.nlp:
            return self._split_with_nlp(text)
        return self._split_basic_overlap(text)

    def _split_with_nlp(self, text: str) -> List[Tuple[str, bool, bool]]:
        """Use NLP to create semantically meaningful chunks."""
        if not self.nlp:
            return self._split_basic_overlap(text)
            
        doc = self.nlp(text)
        sentences = list(doc.sents)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for i, sent in enumerate(sentences):
            sent_text = sent.text.strip()
            sent_len = len(sent_text)
            
            if current_length + sent_len > self.chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append((
                    chunk_text,
                    len(chunks) == 0,
                    False
                ))
                current_chunk = []
                current_length = 0
                
            current_chunk.append(sent_text)
            current_length += sent_len
            
        if current_chunk:
            chunks.append((
                ' '.join(current_chunk),
                len(chunks) == 0,
                True
            ))
            
        return chunks

    def _split_basic_overlap(self, text: str) -> List[Tuple[str, bool, bool]]:
        """Basic text splitting with overlap."""
        if len(text) <= self.chunk_size:
            return [(text, True, True)]
            
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            if end < len(text):
                # Try to break at sentence or paragraph
                for marker in ['.\n\n', '.\n', '. ', '\n\n', '\n', ' ']:
                    break_point = text.rfind(marker, start, end)
                    if break_point != -1:
                        end = break_point + len(marker)
                        break
                        
            chunk = text[start:end].strip()
            if len(chunk) >= self.min_chunk_size:
                chunks.append((
                    chunk,
                    start == 0,
                    end >= len(text)
                ))
                
            start = end - self.overlap_size
            
        return chunks

    def process_file_in_batches(
        self, file_path: str,
        batch_size: Optional[int] = None,
        show_progress: bool = True
    ) -> Generator[List[Dict[str, Any]], None, None]:
        """Process a file and yield batches of chunks."""
        if batch_size is None:
            batch_size = self.batch_size
            
        all_chunks = self.process_file(file_path, show_progress=show_progress)
        out_batch: List[Dict[str, Any]] = []
        
        for chunk_dict in all_chunks:
            out_batch.append(chunk_dict)
            if len(out_batch) >= batch_size:
                yield out_batch
                out_batch = []
                
        if out_batch:
            yield out_batch

    def process_file(
        self,
        file_path: str,
        show_progress: bool = True
    ) -> Generator[Dict[str, Any], None, None]:
        """Process a single file into chunks."""
        ext = Path(file_path).suffix.lower()
        if ext not in self._get_supported_extensions():
            raise ValueError(f"Unsupported file type: {ext}")

        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        fsize = len(text)

        logger.info(f"Processing {file_path} with size={fsize} chars")
        chunk_count = 0
        pbar = None
        if show_progress:
            est_chunks = min(fsize // (self.chunk_size // 2), self.max_chunks)
            pbar = tqdm(total=est_chunks, desc=f"Chunks for {os.path.basename(file_path)}", unit="chunk")

        for chunk_dict in self.chunk_document(text):
            yield chunk_dict
            chunk_count += 1
            if pbar:
                pbar.update(1)
                
        if pbar:
            pbar.close()

        logger.info(f"Finished chunking {file_path}: {chunk_count} chunks")

    def process_files_in_parallel(
        self,
        file_paths: List[str],
        show_progress: bool = True
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Process multiple files in parallel."""
        valid = [
            fp for fp in file_paths
            if Path(fp).suffix.lower() in self._get_supported_extensions()
            and os.path.exists(fp)
        ]
        
        if not valid:
            logger.warning("No valid files to process in parallel.")
            return {}
            
        results: Dict[str, List[Dict[str, Any]]] = {}

        if len(valid) == 1 or self.num_workers < 2:
            for path in valid:
                chunk_list = list(self.process_file(path, show_progress=show_progress))
                results[path] = chunk_list
            return results

        with ThreadPoolExecutor(max_workers=self.num_workers) as ex:
            pb = None
            if show_progress:
                pb = tqdm(total=len(valid), desc="Parallel chunking", unit="file")
                
            future_map = {
                ex.submit(self.process_file, p, False): p
                for p in valid
            }
            
            for fut in as_completed(future_map):
                fpath = future_map[fut]
                try:
                    chunks = list(fut.result())
                    results[fpath] = chunks
                except Exception as e:
                    logger.error(f"Error chunking {fpath}: {e}")
                    results[fpath] = []
                if pb:
                    pb.update(1)
                    pb.set_postfix({"chunks": len(results[fpath])})
                    
            if pb:
                pb.close()

        return results

    @abstractmethod
    def chunk_document(self, text: str) -> Generator[Dict[str, Any], None, None]:
        """Convert document text into chunks."""
        pass


class MarkdownChunker(BaseChunker):
    """Chunker implementation for Markdown files."""

    def _convert_to_html(self, text: str) -> str:
        return markdown.markdown(text)

    def _get_supported_extensions(self) -> List[str]:
        return [".md"]

    def chunk_document(self, text: str) -> Generator[Dict[str, Any], None, None]:
        """Convert markdown to HTML, extract sections, and create chunks."""
        html = self._convert_to_html(text)
        html, code_blocks = self._extract_code_blocks(html)
        sections = self._extract_sections(html)

        for section in sections:
            content = section['content'].strip()
            if not content and not self.always_create_chunks:
                continue

            if len(content) <= self.chunk_size:
                if len(content) >= self.min_chunk_size:
                    yield {
                        'chunk': content,
                        'metadata': {'title': section['title'], 'level': section['level']},
                        'embedding_text': content
                    }
                continue

            # Split long sections with overlap
            chunks = self._split_with_semantic_overlap(content)
            for chunk_text, is_first, is_last in chunks:
                if len(chunk_text) < self.min_chunk_size:
                    continue

                metadata = {
                    'title': section['title'],
                    'level': section['level'],
                    'is_first': is_first,
                    'is_last': is_last
                }

                yield {
                    'chunk': chunk_text,
                    'metadata': metadata,
                    'embedding_text': chunk_text
                }

        # Handle code blocks separately
        for code in code_blocks:
            if not code['content'].strip():
                continue

            metadata = {
                'type': 'code',
                'language': code['language'],
                'context': code['context']
            }

            yield {
                'chunk': code['content'],
                'metadata': metadata,
                'embedding_text': code['content']
            }


class HtmlChunker(BaseChunker):
    """Chunker implementation for HTML files."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Additional patterns for web-specific content
        self.duplicate_patterns = re.compile(
            r'(search|subscribe|sign up|newsletter|follow us|share this|read more|learn more|view all|show more)',
            re.IGNORECASE
        )
        self.noise_patterns = re.compile(
            r'(cookie|privacy|terms|copyright|sitemap|accessibility|\d+\s*shares?|\d+\s*views?)',
            re.IGNORECASE
        )
        
    def _similarity_ratio(self, text1: str, text2: str) -> float:
        """Calculate similarity ratio between two texts using character-based comparison."""
        # Normalize texts
        text1_norm = ' '.join(re.findall(r'\w+', text1.lower()))
        text2_norm = ' '.join(re.findall(r'\w+', text2.lower()))
        
        # Get lengths
        len1, len2 = len(text1_norm), len(text2_norm)
        if len1 == 0 or len2 == 0:
            return 0.0
            
        # Use character-based similarity
        common = sum(1 for i in range(min(len1, len2)) if text1_norm[i] == text2_norm[i])
        return 2 * common / (len1 + len2)

    def _convert_to_html(self, text: str) -> str:
        return text  # Already HTML

    def _get_supported_extensions(self) -> List[str]:
        return [".html", ".htm"]



    def chunk_document(self, text: str) -> Generator[Dict[str, Any], None, None]:
        """Process HTML directly, extract sections, and create chunks with improved deduplication."""
        html, code_blocks = self._extract_code_blocks(text)

        # Parse HTML and do initial cleanup
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove script, style, meta tags and other non-content elements
        for tag in soup.find_all(['script', 'style', 'meta', 'link', 'noscript', 'nav', 'footer', 'header']):
            tag.decompose()

        # Extract main content
        main_content = soup.find(['main', 'article', 'div', 'body'])
        if not main_content:
            return

        # Track all seen content for deduplication
        seen_content = set()
        seen_content_normalized = set()

        def normalize_text(text: str) -> str:
            """Normalize text for fuzzy matching."""
            return ' '.join(re.findall(r'\w+', text.lower()))

        def is_unique_content(content: str, min_length: int = 0) -> bool:
            """Check if content is unique and meets minimum length."""
            if not content or (min_length > 0 and len(content) < min_length):
                return False

            # Clean and normalize content
            content_clean = self._clean_text(content)
            if len(content_clean) < min_length:
                return False

            content_norm = normalize_text(content_clean)
            
            # Check for exact or near duplicates
            if content_clean in seen_content or content_norm in seen_content_normalized:
                return False

            # Check similarity with existing content
            for existing_norm in seen_content_normalized:
                if self._similarity_ratio(content_norm, existing_norm) > 0.7:
                    return False

            seen_content.add(content_clean)
            seen_content_normalized.add(content_norm)
            return True

        def process_chunk(chunk_text: str, metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            """Process and validate a single chunk."""
            if not is_unique_content(chunk_text, self.min_chunk_size):
                return None

            chunk_clean = self._clean_text(chunk_text)
            return {
                'chunk': chunk_clean,
                'metadata': metadata,
                'embedding_text': chunk_clean
            }

        # Process sections
        sections = self._extract_sections(str(main_content))
        for section in sections:
            content = section['content']
            if not is_unique_content(content, self.min_chunk_size):
                continue

            content_clean = self._clean_text(content)
            if len(content_clean) <= self.chunk_size:
                chunk = process_chunk(content_clean, {
                    'title': section['title'],
                    'level': section['level']
                })
                if chunk:
                    yield chunk
                continue

            # Split long sections with overlap
            chunks = self._split_with_semantic_overlap(content_clean)
            prev_chunk = None
            for chunk_text, is_first, is_last in chunks:
                if prev_chunk and self._similarity_ratio(chunk_text, prev_chunk) > 0.7:
                    continue

                chunk = process_chunk(chunk_text, {
                    'title': section['title'],
                    'level': section['level'],
                    'is_first': is_first,
                    'is_last': is_last
                })
                if chunk:
                    yield chunk
                    prev_chunk = chunk_text

        # Handle code blocks separately
        for code in code_blocks:
            if not code['content'].strip():
                continue

            chunk = process_chunk(code['content'], {
                'type': 'code',
                'language': code['language'],
                'context': code['context']
            })
            if chunk:
                yield chunk


# For backward compatibility
class EnhancedChunker(MarkdownChunker):
    """
    Enhanced markdown chunker with parallel processing capabilities.

    We also provide a method 'process_file_in_batches' that yields
    lists (batches) of chunk dicts, so large files can be processed
    and consumed in increments by something like VectorDB.
    """

    def __init__(
        self,
        chunk_size: int = 150,
        text_embedding_size: int = 512,
        min_chunk_size: int = 50,
        max_chunks: int = 5000,
        batch_size: Optional[int] = None,
        ignore_link_urls: bool = True,
        code_context_window: int = 2,
        include_section_headers: bool = True,
        always_create_chunks: bool = True,
        overlap_size: int = 50,
        semantic_headers_only: bool = True,
        min_section_length: int = 100,
        num_workers: Optional[int] = None,
        auto_optimize: bool = True,
        debug_output: bool = False,
    ):
        self.chunk_size = chunk_size
        self.text_embedding_size = text_embedding_size
        self.min_chunk_size = min_chunk_size
        self.max_chunks = max_chunks
        self.batch_size = batch_size or DEFAULT_CHUNK_BATCH_SIZE
        self.ignore_link_urls = ignore_link_urls
        self.code_context_window = code_context_window
        self.include_section_headers = include_section_headers
        self.always_create_chunks = always_create_chunks
        
        self.overlap_size = overlap_size
        self.semantic_headers_only = semantic_headers_only
        self.min_section_length = min_section_length
        self.debug_output = debug_output
        self.debug_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'debug')
        if self.debug_output:
            os.makedirs(self.debug_dir, exist_ok=True)

        self.nav_patterns = re.compile(
            r'(menu|navigation|navbar|breadcrumb|footer|header|copyright|social|share|related|popular|trending|categories|tags|search)',
            re.IGNORECASE
        )

        # Common web page metadata patterns to filter out
        self.metadata_patterns = re.compile(
            r'(posted on|published|last updated|author|comments|views|likes|share this|follow us|subscribe)',
            re.IGNORECASE
        )

        self.resource_manager = get_resource_manager(auto_optimize=auto_optimize)
        if num_workers is None and auto_optimize:
            # get a typed executor so we can read max_workers
            ex = self.resource_manager.get_executor("io")
            if isinstance(ex, ThreadPoolExecutor):
                self.num_workers = ex._max_workers  # or ex.max_workers if you prefer
            else:
                self.num_workers = 4
        else:
            self.num_workers = num_workers or 4

        self.nlp = None
        if self.semantic_headers_only:
            try:
                import spacy
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("Using spaCy for semantic analysis in EnhancedChunker.")
            except (ImportError, OSError):
                logger.warning("spaCy not available. Overlap logic will be simpler.")

        logger.info(
            f"EnhancedChunker initialized with chunk_size={chunk_size}, "
            f"overlap_size={overlap_size}, workers={self.num_workers}"
        )

    def process_file_in_batches(
        self, file_path: str, batch_size: Optional[int] = None, show_progress: bool = True
    ) -> Generator[List[Dict[str, Any]], None, None]:
        """
        Process a single file but yield results in lists (batches) of chunk dicts.
        This is the method your VectorDB calls to handle large docs in increments.
        """
        if batch_size is None:
            batch_size = self.batch_size
        all_chunks = self.process_file(file_path, show_progress=show_progress)
        out_batch: List[Dict[str, Any]] = []
        for cdict in all_chunks:
            out_batch.append(cdict)
            if len(out_batch) >= batch_size:
                yield out_batch
                out_batch = []
        if out_batch:
            yield out_batch

    def process_file(
        self, file_path: str, show_progress: bool = True
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Process a single file (markdown or HTML), chunking it into dicts with
        { "chunk":..., "metadata":..., "embedding_text":... }.
        Yields them one at a time.
        """
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in [".md", ".html", ".htm"]:
            raise ValueError("Only markdown and HTML files are supported.")

        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return  # yields nothing

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        fsize = len(text)

        logger.info(f"Processing {ext} file {file_path} with size={fsize} chars.")
        chunk_count = 0
        pbar = None
        if show_progress:
            est_chunks = min(fsize // (self.chunk_size // 2), self.max_chunks)
            pbar = tqdm(total=est_chunks, desc=f"Chunks for {os.path.basename(file_path)}", unit="chunk")

        # Choose appropriate chunker based on file extension
        if ext in [".html", ".htm"]:
            chunker = HtmlChunker(
                chunk_size=self.chunk_size,
                text_embedding_size=self.text_embedding_size,
                min_chunk_size=self.min_chunk_size,
                max_chunks=self.max_chunks,
                batch_size=self.batch_size,
                ignore_link_urls=self.ignore_link_urls,
                code_context_window=self.code_context_window,
                include_section_headers=self.include_section_headers,
                always_create_chunks=self.always_create_chunks,
                overlap_size=self.overlap_size,
                semantic_headers_only=self.semantic_headers_only,
                min_section_length=self.min_section_length,
                num_workers=self.num_workers,
                auto_optimize=True,
                debug_output=self.debug_output
            )
        else:  # .md by default
            chunker = MarkdownChunker(
                chunk_size=self.chunk_size,
                text_embedding_size=self.text_embedding_size,
                min_chunk_size=self.min_chunk_size,
                max_chunks=self.max_chunks,
                batch_size=self.batch_size,
                ignore_link_urls=self.ignore_link_urls,
                code_context_window=self.code_context_window,
                include_section_headers=self.include_section_headers,
                always_create_chunks=self.always_create_chunks,
                overlap_size=self.overlap_size,
                semantic_headers_only=self.semantic_headers_only,
                min_section_length=self.min_section_length,
                num_workers=self.num_workers,
                auto_optimize=True,
                debug_output=self.debug_output
            )

        # For debug output
        chunks = []

        # chunk_document yields chunk dicts
        for chunk_dict in chunker.chunk_document(text):
            chunks.append(chunk_dict)
            yield chunk_dict
            chunk_count += 1
            if pbar:
                pbar.update(1)

        # Save debug output if enabled
        if self.debug_output:
            debug_file = os.path.join(
                self.debug_dir,
                f"{os.path.splitext(os.path.basename(file_path))[0]}_chunks.json"
            )
            with open(debug_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "file": file_path,
                        "chunks": chunks,
                        "stats": {
                            "total_chunks": len(chunks),
                            "avg_chunk_size": sum(len(c["chunk"]) for c in chunks) / len(chunks) if chunks else 0,
                            "min_chunk_size": min((len(c["chunk"]) for c in chunks), default=0),
                            "max_chunk_size": max((len(c["chunk"]) for c in chunks), default=0)
                        }
                    },
                    f,
                    indent=2,
                    ensure_ascii=False
                )
        if pbar:
            pbar.close()

        logger.info(f"Finished chunking {file_path}, total {chunk_count} chunks.")

    def process_files_in_parallel(
        self, file_paths: List[str], show_progress: bool = True
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Process multiple files (markdown and HTML) in parallel, returning a dict."""
        valid = [fp for fp in file_paths if fp.lower().endswith((".md", ".html", ".htm")) and os.path.exists(fp)]
        if not valid:
            logger.warning("No valid markdown or HTML files to process in parallel.")
            return {}
        results: Dict[str, List[Dict[str, Any]]] = {}

        if len(valid) == 1 or self.num_workers < 2:
            # do single-thread
            for path in valid:
                chunk_list = list(self.process_file(path, show_progress=show_progress))
                results[path] = chunk_list
            return results

        with ThreadPoolExecutor(max_workers=self.num_workers) as ex:
            pb = None
            if show_progress:
                pb = tqdm(total=len(valid), desc="Parallel chunking", unit="file")
            future_map = {
                ex.submit(self.process_file, p, False): p for p in valid
            }
            for fut in as_completed(future_map):
                fpath = future_map[fut]
                try:
                    c = list(fut.result())
                    results[fpath] = c
                except Exception as e:
                    logger.error(f"Error chunking {fpath}: {e}")
                    results[fpath] = []
                if pb:
                    pb.update(1)
                    pb.set_postfix({"chunks": len(results[fpath])})
            if pb:
                pb.close()

        return results

    def chunk_document(self, text: str) -> Generator[Dict[str, Any], None, None]:
        """
        Convert markdown => HTML => extract code => remove code => remove nav => section => chunk
        Yields chunk dicts (one chunk at a time).
        """
        html = self._markdown_to_html(text)

        # 1) extract code blocks
        code_blocks = self._extract_code_blocks(html)
        for code_block in code_blocks:
            # potentially chunk the code block further if huge
            if not isinstance(code_block, dict):
                continue
            code_txt = str(code_block.get('content', ''))
            lines = code_txt.split("\n")
            current = []
            current_len = 0
            for line in lines:
                if (current_len + len(line) + 1) > self.chunk_size and current:
                    chunk_txt = "\n".join(current).strip()
                    yield {
                        "chunk": chunk_txt,
                        "metadata": {
                            "type": "code_block",
                            "language": str(code_block.get('language', '')),
                            "context": list(code_block.get('context', []))
                        },
                        "embedding_text": chunk_txt,
                    }
                    current = [line]
                    current_len = len(line)
                else:
                    current.append(line)
                    current_len += len(line) + 1
            if current:
                chunk_txt = "\n".join(current).strip()
                yield {
                    "chunk": chunk_txt,
                    "metadata": {
                        "type": "code_block",
                        "language": str(code_block.get('language', '')),
                        "context": list(code_block.get('context', []))
                    },
                    "embedding_text": chunk_txt,
                }

        # 2) remove those code blocks from soup so we don't process them again
        soup = BeautifulSoup(html, "html.parser")
        for pre in soup.find_all("pre"):
            if isinstance(pre, Tag):
                pre.decompose()

        # remove nav
        self._remove_navigation_elements(soup)

        # 3) extract sections
        sections = self._extract_sections(str(soup))

        # 4) chunk each section with overlap
        for idx, sec in enumerate(sections):
            content = sec["content"].strip()
            if not content:
                continue
            if len(content) <= self.chunk_size:
                yield {
                    "chunk": content,
                    "metadata": {
                        "type": sec.get("type", "text_chunk"),
                        "title": sec.get("title", ""),
                        "level": sec.get("level", 0),
                        "section_index": idx,
                        "is_complete_section": True
                    },
                    "embedding_text": content,
                }
                continue

            subchunks = self._split_with_semantic_overlap(content)
            for j, (subtxt, is_first, is_last) in enumerate(subchunks):
                yield {
                    "chunk": subtxt,
                    "metadata": {
                        "type": sec.get("type", "text_chunk"),
                        "title": sec.get("title", ""),
                        "level": sec.get("level", 0),
                        "section_index": idx,
                        "chunk_index": j,
                        "is_first_chunk": is_first,
                        "is_last_chunk": is_last,
                        "is_complete_section": False
                    },
                    "embedding_text": subtxt,
                }

    def _markdown_to_html(self, text: str) -> str:
        return markdown.markdown(
            text,
            extensions=["fenced_code", "tables", "attr_list"],
        )

    def _remove_navigation_elements(self, soup: BeautifulSoup):
        # remove nav-ish stuff
        for x in soup.find_all(class_=self.nav_patterns):
            if isinstance(x, Tag):
                x.decompose()
        for x in soup.find_all(id=self.nav_patterns):
            if isinstance(x, Tag):
                x.decompose()
        for nav in soup.find_all(["nav", "header", "footer"]):
            if isinstance(nav, Tag):
                nav.decompose()

    def _is_semantic_header(self, tag: Tag) -> bool:
        txt = tag.get_text().strip()
        
        # Filter out common non-semantic headers
        if any(p.search(txt.lower()) for p in [self.nav_patterns, self.metadata_patterns]):
            return False
            
        # Filter out date-only headers
        if re.match(r'^\d{4}(-\d{2}){0,2}$', txt):
            return False
            
        # Filter out very short headers that don't form proper phrases
        if len(txt.split()) < 2 and len(txt) < 10:
            return False
            
        # Must contain some letters and reasonable length
        return len(txt) >= 10 and any(c.isalpha() for c in txt)

    def _extract_sections(self, html: str) -> List[Dict[str, Any]]:
        """Extract sections using semantic structure and headers while avoiding duplicates."""
        soup = BeautifulSoup(html, "html.parser")
        sections: List[Dict[str, Any]] = []
        processed_content = set()

        def normalize_content(content: str) -> str:
            """Normalize content for deduplication."""
            return ' '.join(re.findall(r'\w+', content.lower()))

        def add_section(title: str, content: str, level: int, section_type: str = 'content') -> bool:
            """Add a section if its content is unique and meaningful."""
            content = content.strip()
            if not content or len(content) < self.min_section_length:
                return False

            # Clean and normalize content
            content_clean = self._clean_text(content)
            content_norm = normalize_content(content_clean)
            
            # Skip if too similar to existing sections
            for existing_norm in processed_content:
                if self._similarity_ratio(content_norm, existing_norm) > 0.7:
                    return False
                    
            processed_content.add(content_norm)
            sections.append({
                'content': content_clean,
                'level': level,
                'title': title,
                'type': section_type
            })
            return True

        # First try sections marked by comments
        content = str(soup)
        section_matches = re.finditer(
            r'<!--\s*SECTION_START\s+level="(\d+)"\s+title="([^"]*?)"\s*-->\s*(.*?)\s*<!--\s*SECTION_END\s*-->',
            content,
            re.DOTALL
        )
        
        found_sections = False
        for match in section_matches:
            found_sections = True
            level = int(match.group(1))
            title = match.group(2)
            section_content = match.group(3).strip()
            
            if add_section(title, section_content, level, 'semantic_section'):
                found_sections = True

        if found_sections:
            return sections

        # Check for main content markers
        main_content_match = re.search(
            r'<!--\s*MAIN_CONTENT_START\s*-->\s*(.*?)\s*<!--\s*MAIN_CONTENT_END\s*-->',
            content,
            re.DOTALL
        )
        
        if main_content_match:
            main_content = main_content_match.group(1)
            soup = BeautifulSoup(main_content, "html.parser")

        # Try semantic article/section tags
        semantic_tags = soup.find_all(['article', 'section'])
        for tag in semantic_tags:
            if not isinstance(tag, Tag):
                continue
                
            if self._is_navigation_element(tag):
                continue
                
            # Get title from aria-label, title attribute, or header
            title = ''
            if hasattr(tag, 'get'):
                title = str(tag.get('aria-label') or tag.get('title') or '').strip()
                
            if not title and isinstance(tag, Tag):
                header = tag.find(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
                if isinstance(header, Tag):
                    title = header.get_text().strip()
                    
            if not title:
                title = 'Content Section'
                
            add_section(title, str(tag), 1, 'semantic_tag')

        # Try headers for remaining content
        hdrs = soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])
        if self.semantic_headers_only and self.nlp:
            hdrs = [h for h in hdrs if isinstance(h, Tag) and self._is_semantic_header(h)]

        if hdrs:
            # Sort headers by DOM order
            hdrs = sorted(hdrs, key=lambda x: (getattr(x, 'sourceline', 0), getattr(x, 'sourcepos', 0)))
            
            for i, h in enumerate(hdrs):
                if not isinstance(h, Tag):
                    continue
                    
                if self._is_navigation_element(h):
                    continue

                try:
                    level = int(h.name[1])
                    title = h.get_text().strip()
                    next_header = hdrs[i + 1] if i + 1 < len(hdrs) else None
                    
                    # Get content between this header and next header
                    content_parts = []
                    current = h.next_sibling
                    while current and (not next_header or current != next_header):
                        if isinstance(current, Tag):
                            if current.name not in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6'] and \
                               not self._is_navigation_element(current):
                                content_parts.append(str(current))
                        elif isinstance(current, NavigableString):
                            content_parts.append(str(current))
                    
                    content = ''.join(content_parts)
                    add_section(title, content, level, 'header_section')
                except Exception:
                    continue

        # If no sections found, use content blocks
        if not sections:
            content_blocks = self._extract_content_blocks(soup)
            for block in content_blocks:
                add_section(
                    block['title'],
                    block['content'],
                    block['level'],
                    block['type']
                )

        return sections

    def _extract_content_blocks(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """More selective fallback approach for content block extraction."""
        out: List[Dict[str, Any]] = []
        
        # Find substantial text blocks
        paragraphs = []
        for p in soup.find_all(['p', 'div', 'section']):
            if isinstance(p, Tag):
                txt = p.get_text().strip()
                # Filter out likely metadata and very short paragraphs
                if len(txt) >= self.min_section_length and not self.metadata_patterns.search(txt.lower()):
                    paragraphs.append(txt)
                    
        if paragraphs:
            # Group related paragraphs together
            current_section = []
            current_length = 0
            sections = []
            
            for p in paragraphs:
                if current_length + len(p) > self.chunk_size * 2:
                    if current_section:
                        sections.append('\n\n'.join(current_section))
                    current_section = [p]
                    current_length = len(p)
                else:
                    current_section.append(p)
                    current_length += len(p)
                    
            if current_section:
                sections.append('\n\n'.join(current_section))
                
            for i, section in enumerate(sections):
                out.append({
                    "content": section,
                    "level": 0,
                    "title": f"ContentBlock{i+1}",
                    "type": "content_block"
                })
        else:
            # Last fallback - get all text but with higher minimum length
            full = soup.get_text().strip()
            if len(full) >= self.min_section_length * 2:
                out.append({
                    "content": full,
                    "level": 0,
                    "title": "FullContent",
                    "type": "full_content"
                })
                
        return out

    def _split_with_semantic_overlap(self, text: str) -> List[Tuple[str, bool, bool]]:
        """
        We can do an NLP-based or fallback approach. 
        Return list of (chunk_text, is_first, is_last).
        """
        if self.nlp:
            try:
                return self._split_with_nlp(text)
            except Exception:
                logger.warning("NLP chunking failed; falling back.")
        return self._split_basic_overlap(text)

    def _split_with_nlp(self, text: str) -> List[Tuple[str, bool, bool]]:
        if not self.nlp:
            return self._split_basic_overlap(text)
        doc = self.nlp(text)
        sents = list(doc.sents)
        chunks: List[str] = []
        cur: List[str] = []
        cur_len = 0
        for sent in sents:
            st = sent.text.strip()
            if (cur_len + len(st)) > self.chunk_size and cur:
                chunk_txt = " ".join(cur)
                chunks.append(chunk_txt)
                # overlap
                overlap_count = max(1, min(3, len(cur)//5))
                cur = cur[-overlap_count:] + [st]
                cur_len = sum(len(x) for x in cur)
            else:
                cur.append(st)
                cur_len += len(st) + 1
        if cur:
            chunk_txt = " ".join(cur)
            chunks.append(chunk_txt)

        result: List[Tuple[str, bool, bool]] = []
        for i, c in enumerate(chunks):
            result.append((c, i == 0, i == len(chunks)-1))
        return result

    def _split_basic_overlap(self, text: str) -> List[Tuple[str, bool, bool]]:
        # More aggressive chunking with paragraph awareness
        paragraphs = text.split('\n\n')
        out: List[str] = []
        current: List[str] = []
        current_len = 0
        
        for p in paragraphs:
            p = p.strip()
            if not p:
                continue
                
            # If adding this paragraph would exceed chunk size
            if current_len + len(p) > self.chunk_size and current:
                chunk_txt = '\n\n'.join(current)
                out.append(chunk_txt)
                
                # Keep last paragraph for overlap if it's not too big
                if len(current[-1]) < self.overlap_size:
                    current = [current[-1], p]
                    current_len = len(current[-1]) + len(p)
                else:
                    current = [p]
                    current_len = len(p)
            else:
                current.append(p)
                current_len += len(p)
                
        if current:
            chunk_txt = '\n\n'.join(current)
            out.append(chunk_txt)
            
        return [(c, i == 0, i == len(out)-1) for i, c in enumerate(out)]


def save_debug_info(file_path: str, chunks: List[Dict[str, Any]]) -> None:
    """Save debug information about chunks to a JSON file."""
    debug_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'debug')
    os.makedirs(debug_dir, exist_ok=True)
    
    debug_file = os.path.join(
        debug_dir,
        f"{os.path.splitext(os.path.basename(file_path))[0]}_chunks.json"
    )
    
    with open(debug_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "file": file_path,
                "chunks": chunks,
                "stats": {
                    "total_chunks": len(chunks),
                    "avg_chunk_size": sum(len(c["chunk"]) for c in chunks) / len(chunks) if chunks else 0,
                    "min_chunk_size": min((len(c["chunk"]) for c in chunks), default=0),
                    "max_chunk_size": max((len(c["chunk"]) for c in chunks), default=0)
                }
            },
            f,
            indent=2,
            ensure_ascii=False
        )

def process_directory(directory_path: str, recursive: bool = True, chunker: Optional[EnhancedChunker] = None, debug: bool = False) -> Dict[str, List[Dict[str, Any]]]:
    """
    Simple utility to chunk all markdown and HTML files in a directory in parallel.
    """
    if not chunker:
        chunker = EnhancedChunker(auto_optimize=True)
    p = Path(directory_path)
    if recursive:
        files = list(p.glob("**/*.md")) + list(p.glob("**/*.html")) + list(p.glob("**/*.htm"))
    else:
        files = list(p.glob("*.md")) + list(p.glob("*.html")) + list(p.glob("*.htm"))
    file_paths = [str(x) for x in files]
    if not file_paths:
        logger.warning(f"No markdown or HTML files in {directory_path}")
        return {}
    
    result = chunker.process_files_in_parallel(file_paths)
    
    if debug:
        for file_path, chunks in result.items():
            save_debug_info(file_path, chunks)
    
    return result