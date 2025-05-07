# src/llamasearch/core/chunker.py

import re
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter

from llamasearch.utils import setup_logging

logger = setup_logging(__name__, use_qt_handler=True)

DEFAULT_SEPARATORS = [
    "\n\n",
    "\n# ",
    "\n## ",
    "\n### ",
    "\n#### ",
    "\n- ",
    "\n* ",
    "\n+ ",
    "```\n",
    "\n---\n",
    "\n___\n",
    "\n***\n",
    "\n",
    ". ",
    "! ",
    "? ",
    "; ",
    ": ",
    ",",
    "\u3002",
    "\uff0e",
    "\u3001",
    " ",
    "",
]

DEFAULT_MIN_CHUNK_LENGTH = 30 # Lowered default for better testability with short texts
MARKDOWN_LIKE_EXTENSIONS = {".md", ".markdown", ".txt"}
HTML_EXTENSIONS = {".html", ".htm"}


def calculate_effective_length(text: str) -> int:
    """Calculates the length of the text, excluding the URL part of Markdown links."""
    text_without_urls = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    return len(text_without_urls)


def chunk_markdown_text(
    markdown_text: str,
    source: Optional[str] = None,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
    min_chunk_char_length: int = DEFAULT_MIN_CHUNK_LENGTH,
    separators: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Chunks text based on file type inferred from the source path.
    Uses BeautifulSoup for HTML, minimal processing for Markdown/Text.
    Filters based on effective length after link stripping.

    Args:
        markdown_text: The raw content (Markdown, Text, HTML) to chunk.
        source: Optional identifier for the source of the text (e.g., filename).
        chunk_size: The target maximum size of each chunk (in characters).
        chunk_overlap: The number of characters to overlap between chunks.
        min_chunk_char_length: The minimum *effective* character length (excluding link URLs) for a chunk to be kept.
        separators: List of separators to use for splitting. Defaults to a robust list.

    Returns:
        A list of dictionaries, where each dictionary represents a valid chunk
        with 'chunk' (the text) and 'metadata' (source, chunk_index, length).
    """
    if not markdown_text:
        logger.warning(f"Received empty input text from source: {source}")
        return []

    text_content_for_splitting = "" # This will be the content after processing and link stripping

    processing_mode = "unknown"
    try:
        file_ext = (
            Path(source).suffix.lower()
            if source and isinstance(source, str)
            else ".unknown"
        )
    except Exception:
        file_ext = ".unknown"

    if file_ext in MARKDOWN_LIKE_EXTENSIONS:
        processing_mode = "markdown/text"
        logger.debug(f"Processing source '{source}' as Markdown/Text.")
        # For markdown/text, the original text is used for chunking, but links are stripped for length calculation and splitter
        text_content_for_splitting = markdown_text
        text_content_for_splitting = re.sub(r"\r\n", "\n", text_content_for_splitting)
        text_content_for_splitting = re.sub(r"\n{3,}", "\n\n", text_content_for_splitting)
        # Keep original_text_for_chunking as is for MD/TXT

    elif file_ext in HTML_EXTENSIONS:
        processing_mode = "html"
        logger.debug(f"Processing source '{source}' as HTML with BeautifulSoup.")
        try:
            soup = BeautifulSoup(markdown_text, "html.parser")
            tags_to_remove = [
                "script", "style", "nav", "header", "footer",
                "aside", "form", "button", "meta", "link", "noscript",
            ]
            for tag_name in tags_to_remove:
                for tag in soup.find_all(tag_name):
                    tag.decompose()

            main_content_selectors = [
                "article", "main", "#content", ".content", ".main-content",
                "#main", "#primary", ".post-content", ".entry-content",
            ]
            main_body = None
            for selector in main_content_selectors:
                main_body = soup.select_one(selector)
                if main_body:
                    logger.debug(f"Found main content container using selector: '{selector}'")
                    break

            if not main_body:
                main_body = soup.body if soup.body else soup
                logger.debug("Using text from body tag or entire structure." if soup.body else "No body/main content, using entire structure.")

            # Get text for splitting (links will be stripped later from this)
            # And also get text for the actual chunk content (which should retain structure but be clean)
            # For HTML, we use get_text for the final chunk content.
            extracted_html_text = main_body.get_text(separator="\n", strip=True)
            extracted_html_text = re.sub(r"\n{3,}", "\n\n", extracted_html_text)
            extracted_html_text = re.sub(r"^\s+|\s+$", "", extracted_html_text, flags=re.MULTILINE)

            lines = extracted_html_text.split("\n")
            cleaned_lines = []
            for line in lines:
                line_stripped = line.strip()
                if re.match(r"^\[[^\]]+\]\([^)]+\)$", line_stripped) or \
                   re.match(r"^\[\*\*?[^*]+\*\*?\]\([^)]+\)$", line_stripped) or \
                   re.match(r"^uid\s+\[\s*unknown\s*\]", line_stripped) or \
                   line_stripped.lower() in [
                       "available for this page:", "[ back to top ▲ ]", "back to top ▲",
                       "[ back to top ]", "back to top",
                   ] or \
                   re.match(r"^(\[[a-z-]+\]\s+\[[^\]]+\]\([^)]+\)\s*)+$", line_stripped):
                    continue
                cleaned_lines.append(line)
            
            text_content_for_splitting = "\n".join(cleaned_lines)

        except Exception as bs_err:
            logger.error(
                f"BeautifulSoup parsing/extraction failed for HTML source {source}: {bs_err}. Falling back to raw text.",
                exc_info=True,
            )
            text_content_for_splitting = markdown_text # Fallback
            processing_mode = "html_fallback_raw"
    else:
        processing_mode = "unknown_fallback_raw"
        logger.warning(
            f"Unknown file type '{file_ext}' for source '{source}'. Processing as raw text."
        )
        text_content_for_splitting = markdown_text


    # --- Prepare text_content_for_splitting for the splitter ---
    # Strip markdown links only from the version used by the splitter's length function and splitting logic
    text_for_splitter_logic = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text_content_for_splitting)

    if not text_for_splitter_logic.strip():
        logger.warning(
            f"No text content remaining after {processing_mode} processing and link stripping for splitter logic: {source}"
        )
        return []

    logger.debug(
        f"Text content length for splitter logic after {processing_mode} processing and link stripping: {len(text_for_splitter_logic)} chars."
    )

    effective_separators = separators if separators is not None else DEFAULT_SEPARATORS

    if chunk_overlap >= chunk_size:
        logger.warning(
            f"Chunk overlap ({chunk_overlap}) >= chunk size ({chunk_size}) for source '{source}'. Reducing overlap."
        )
        chunk_overlap = max(0, chunk_size // 4)

    text_splitter = RecursiveCharacterTextSplitter(
        separators=effective_separators,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len, # This will operate on text_for_splitter_logic
        is_separator_regex=False,
        keep_separator=True, 
    )

    try:
        # The splitter operates on text_for_splitter_logic (links stripped)
        # but we will map these splits back to original_text_for_chunking
        # Langchain's splitter returns plain text chunks.
        split_texts_from_stripped = text_splitter.split_text(text_for_splitter_logic)
        
        # Reconstruct chunks from original_text_for_chunking based on the splits
        # This is a simplification; a robust way would involve character offsets.
        # For now, we'll assume the splitter gives us good boundaries,
        # and the resulting chunks will be from original_text_for_chunking but filtered.
        # The key is that the *decision* to split and the *length checks* use the stripped version.
        # The final chunk text can be different if original_text_for_chunking != text_for_splitter_logic (e.g. MD files)
        
        # For simplicity in this pass, we will return chunks from text_for_splitter_logic
        # This means MD links will be stripped in the final output.
        # If we want to keep MD links, the logic here needs to map indices from
        # text_for_splitter_logic back to original_text_for_chunking.
        # Current behavior: MD links WILL be stripped in output chunks. HTML content is already processed.
        split_texts = split_texts_from_stripped

        logger.debug(
            f"Initial split of '{processing_mode}' text for {source or 'unknown'} resulted in {len(split_texts)} potential chunks."
        )
    except Exception as e:
        logger.error(
            f"Error splitting processed text from source {source}: {e}", exc_info=True
        )
        return []

    valid_chunks_with_metadata = []
    
    for i, chunk_text in enumerate(split_texts):
        stripped_chunk = chunk_text.strip() 
        if not stripped_chunk: 
            logger.debug(f"Skipping chunk {i} from {source}: empty after strip.")
            continue

        # Effective length is calculated on the chunk text itself (which is already link-stripped if it came from text_for_splitter_logic)
        effective_len = calculate_effective_length(stripped_chunk) 
        raw_len = len(stripped_chunk)

        logger.debug(
            f"Chunk {i} candidate (EffectiveLen={effective_len}, RawLen={raw_len}): '{stripped_chunk[:80]}...'"
        )

        if effective_len >= min_chunk_char_length:
            first_line = stripped_chunk.split("\n", 1)[0].strip() 
            if len(first_line) > 1 and len(set(first_line)) <= 2: 
                if stripped_chunk.count(first_line) * len(first_line) > raw_len * 0.8:
                    logger.debug(
                        f"Skipping potentially low-quality repetitive chunk {i} from {source}: '{stripped_chunk[:50]}...'"
                    )
                    continue

            metadata = {
                "source": source if source else "unknown",
                "chunk_index_in_doc": i, 
                "length": raw_len,
                "effective_length": effective_len,
                "processing_mode": processing_mode,
            }
            valid_chunks_with_metadata.append(
                {"chunk": stripped_chunk, "metadata": metadata}
            )
        else:
            if raw_len > 0: 
                logger.debug(
                    f"Skipping chunk {i} from {source} due to effective length < {min_chunk_char_length} (EffLen={effective_len}, RawLen={raw_len}): '{stripped_chunk[:50]}...'"
                )

    logger.info(
        f"Split '{processing_mode}' text from {source or 'unknown'} into {len(valid_chunks_with_metadata)} valid chunks (Size: {chunk_size}, Overlap: {chunk_overlap}, MinEffLen: {min_chunk_char_length})"
    )

    if not valid_chunks_with_metadata and text_for_splitter_logic.strip():
        logger.warning(
            f"Chunking resulted in zero valid chunks for source: {source} ({processing_mode} processing). Original text length: {len(markdown_text)}, Processed text (for splitter) length: {len(text_for_splitter_logic)}. Check chunk parameters and content."
        )

    return valid_chunks_with_metadata


def chunk_document(
    markdown_text: str,
    source: Optional[str] = None,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
    min_chunk_char_length: int = DEFAULT_MIN_CHUNK_LENGTH,
) -> Generator[Dict[str, Any], None, None]:
    """Generator version for chunking (delegates to the list version)."""
    chunks = chunk_markdown_text(
        markdown_text, source, chunk_size, chunk_overlap, min_chunk_char_length
    )
    for chunk in chunks:
        yield chunk