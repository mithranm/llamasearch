# llamasearch/core/chunker.py

import os
import re
import json
import logging
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Dict, Any, Generator, Optional

from tqdm import tqdm
import markdown
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_BATCH_SIZE = 100  # Default batch size for processing chunks

###############################################
# Abstract Base and Common Utility Methods
###############################################

class BaseChunker(ABC):
    """
    Abstract base class for document chunking.
    Defines common functionality and interface for chunkers.
    """

    def __init__(
        self,
        chunk_size: int = 150,
        min_chunk_size: int = 50,
        overlap_size: int = 50,
        debug_output: bool = False,
        num_workers: Optional[int] = None,
        **kwargs,
    ):
        self.chunk_size = chunk_size
        self.min_chunk_size = min_chunk_size
        self.overlap_size = overlap_size
        self.debug_output = debug_output
        self.num_workers = num_workers or (os.cpu_count() or 4)
        # Precompile common regex patterns
        self.nav_patterns = re.compile(r'(menu|navigation|navbar|breadcrumb|footer|header)', re.IGNORECASE)
        self.noise_patterns = re.compile(r'(advertisement|sponsored)', re.IGNORECASE)
        self.duplicate_patterns = re.compile(r'(duplicate content|copied content)', re.IGNORECASE)

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
        Adaptive splitter: first attempt to split by paragraph breaks.
        If resulting segments are still too long, apply a sliding window.
        """
        paragraphs = [p.strip() for p in text.split("\n") if len(p.strip()) >= self.min_chunk_size]
        for para in paragraphs:
            if len(para) <= self.chunk_size:
                yield {"chunk": para, "metadata": {}, "embedding_text": para}
            else:
                start = 0
                while start < len(para):
                    end = start + self.chunk_size
                    chunk = para[start:end].strip()
                    if chunk and len(chunk) >= self.min_chunk_size:
                        yield {"chunk": chunk, "metadata": {}, "embedding_text": chunk}
                    start = end - self.overlap_size

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
        pbar = tqdm(total=max(1, len(text) // (self.chunk_size // 2)), desc=f"Chunking {os.path.basename(file_path)}", unit="chunk") if show_progress else None
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

    def process_file_in_batches(self, file_path: str, batch_size: Optional[int] = None, show_progress: bool = True) -> Generator[List[Dict[str, Any]], None, None]:
        """
        Process a file and yield batches (lists) of chunk dictionaries.
        """
        batch_size = batch_size or DEFAULT_CHUNK_BATCH_SIZE
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
            logger.warning("No valid files to process in parallel.")
            return {}
        results: Dict[str, List[Dict[str, Any]]] = {}
        if len(valid) == 1 or self.num_workers < 2:
            for fp in valid:
                results[fp] = list(self.process_file(fp, show_progress=show_progress))
            return results
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
    Chunker for Markdown files.
    Converts Markdown to HTML, extracts visible text, then splits it.
    """
    def _convert_to_html(self, text: str) -> str:
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
    Chunker for HTML files.
    Uses the DOM structure to extract coherent text blocks from the main content area.
    """
    # Common selectors for main content on wiki pages
    MAIN_SELECTORS = ["main", "#WikiaArticle", "#content", ".mw-parser-output"]

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
        # Try using one of the main selectors if available
        for selector in self.MAIN_SELECTORS:
            container = soup.select_one(selector)
            if container:
                text = container.get_text(separator="\n", strip=True)
                if text:
                    return text
        # Fallback: get all visible text
        return soup.get_text(separator="\n", strip=True)

    def chunk_document(self, text: str) -> Generator[Dict[str, Any], None, None]:
        html = self._convert_to_html(text)
        plain_text = self._extract_plain_text(html)
        plain_text = self._clean_text(plain_text)
        yield from self._split_text(plain_text)

# --- Module-level Helper Function ---

def process_directory(
    directory_path: str,
    recursive: bool = True,
    chunker: Optional[BaseChunker] = None,
    debug: bool = False
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Process all supported files in a directory.
    Returns a dict mapping file paths to lists of chunk dictionaries.
    """
    if not chunker:
        chunker = MarkdownChunker()  # Default to Markdown; user can merge with HTML results
    p = Path(directory_path)
    if recursive:
        files = list(p.glob("**/*.md")) + list(p.glob("**/*.html")) + list(p.glob("**/*.htm"))
    else:
        files = list(p.glob("*.md")) + list(p.glob("*.html")) + list(p.glob("*.htm"))
    file_paths = [str(x) for x in files]
    if not file_paths:
        logger.warning(f"No supported files found in {directory_path}")
        return {}
    results = chunker.process_files_in_parallel(file_paths)
    if debug:
        debug_dir = os.path.join(os.path.dirname(directory_path), "debug")
        if os.path.exists(debug_dir):
            for file in os.listdir(debug_dir):
                file_path = os.path.join(debug_dir, file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    logger.error(f"Error deleting file {file_path}: {e}")
        os.makedirs(debug_dir, exist_ok=True)
        for fp, chunks in results.items():
            debug_file = os.path.join(debug_dir, f"{Path(fp).stem}_chunks.json")
            with open(debug_file, "w", encoding="utf-8") as f:
                json.dump({
                    "file": fp,
                    "chunks": chunks,
                    "stats": {
                        "total_chunks": len(chunks),
                        "avg_chunk_size": sum(len(c["chunk"]) for c in chunks) / len(chunks) if chunks else 0,
                        "min_chunk_size": min((len(c["chunk"]) for c in chunks), default=0),
                        "max_chunk_size": max((len(c["chunk"]) for c in chunks), default=0)
                    }
                }, f, indent=2, ensure_ascii=False)
    return results
