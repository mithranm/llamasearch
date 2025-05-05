# src/llamasearch/core/chunker.py

import re
from typing import Any, Dict, Generator, List, Optional
from pathlib import Path  # Added for path operations

# --- Add BeautifulSoup import ---
from bs4 import BeautifulSoup
# ------------------------------

from langchain_text_splitters import RecursiveCharacterTextSplitter

from llamasearch.utils import setup_logging

logger = setup_logging(__name__, use_qt_handler=True)

# Define more robust separators, prioritizing semantic breaks
DEFAULT_SEPARATORS = [
    "\n\n",  # Paragraphs
    "\n# ",
    "\n## ",
    "\n### ",
    "\n#### ",  # Markdown headers
    "\n- ",
    "\n* ",
    "\n+ ",  # Markdown lists (unordered)
    "```\n",  # Code blocks start/end
    "\n---\n",
    "\n___\n",
    "\n***\n",  # Horizontal rules
    "\n",  # Lines
    ". ",
    "! ",
    "? ",  # Sentences
    "; ",
    ": ",
    ",",  # Clauses
    "\u3002",
    "\uff0e",
    "\u3001",  # CJK punctuation
    " ",  # Spaces (last resort)
    "",  # Characters (absolute last resort)
]

DEFAULT_MIN_CHUNK_LENGTH = 50  # Minimum effective characters for a chunk

# --- Define Extensions ---
MARKDOWN_LIKE_EXTENSIONS = {".md", ".markdown", ".txt"}
HTML_EXTENSIONS = {".html", ".htm"}
# --- End Define Extensions ---


def calculate_effective_length(text: str) -> int:
    """Calculates the length of the text, excluding the URL part of Markdown links."""
    # This function is now less critical for filtering if links are pre-stripped,
    # but we keep it for consistency in case the stripping fails or for other uses.
    text_without_urls = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    return len(text_without_urls)


def chunk_markdown_text(
    markdown_text: str,  # Raw input content
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

    # --- Determine file type and apply appropriate preprocessing ---
    text_content = ""
    processing_mode = "unknown"
    try:
        file_ext = (
            Path(source).suffix.lower()
            if source and isinstance(source, str)
            else ".unknown"
        )
    except Exception:  # Handle potential invalid source paths
        file_ext = ".unknown"

    if file_ext in MARKDOWN_LIKE_EXTENSIONS:
        processing_mode = "markdown/text"
        logger.debug(f"Processing source '{source}' as Markdown/Text.")
        # Minimal cleaning for Markdown/Text
        text_content = markdown_text
        text_content = re.sub(r"\r\n", "\n", text_content)  # Normalize line endings
        text_content = re.sub(
            r"\n{3,}", "\n\n", text_content
        )  # Reduce multiple newlines
        # Keep link stripping for consistency? Optional.
        # text_content = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text_content)

    elif file_ext in HTML_EXTENSIONS:
        processing_mode = "html"
        logger.debug(f"Processing source '{source}' as HTML with BeautifulSoup.")
        try:
            # Use 'html.parser' for robustness
            soup = BeautifulSoup(markdown_text, "html.parser")

            # Remove common non-content tags BEFORE getting text
            tags_to_remove = [
                "script",
                "style",
                "nav",
                "header",
                "footer",
                "aside",
                "form",
                "button",
                "meta",
                "link",
                "noscript",
            ]
            for tag_name in tags_to_remove:
                for tag in soup.find_all(tag_name):
                    tag.decompose()

            # Attempt to find common main content containers
            main_content_selectors = [
                "article",
                "main",
                "#content",
                ".content",
                ".main-content",
                "#main",
                "#primary",
                ".post-content",
                ".entry-content",
            ]
            main_body = None
            for selector in main_content_selectors:
                main_body = soup.select_one(selector)
                if main_body:
                    logger.debug(
                        f"Found main content container using selector: '{selector}'"
                    )
                    break

            if not main_body:
                main_body = soup.body if soup.body else soup
                if main_body is soup:
                    logger.debug(
                        "No specific main content container or body tag found, using entire parsed structure."
                    )
                else:
                    logger.debug("Using text from body tag.")

            text_content = main_body.get_text(separator="\n", strip=True)
            text_content = re.sub(
                r"\n{3,}", "\n\n", text_content
            )  # Reduce multiple newlines
            text_content = re.sub(
                r"^\s+|\s+$", "", text_content, flags=re.MULTILINE
            )  # Trim lines

            # Additional Cleanup Regex (Optional - applied after BS)
            lines = text_content.split("\n")
            cleaned_lines = []
            for line in lines:
                line_stripped = line.strip()
                if re.match(r"^\[[^\]]+\]\([^)]+\)$", line_stripped):
                    continue  # Skip link-only lines
                if re.match(r"^\[\*\*?[^*]+\*\*?\]\([^)]+\)$", line_stripped):
                    continue  # Skip bold link-only lines
                if re.match(r"^uid\s+\[\s*unknown\s*\]", line_stripped):
                    continue  # Skip UID lines
                if line_stripped.lower() in [
                    "available for this page:",
                    "[ back to top ▲ ]",
                    "back to top ▲",
                    "[ back to top ]",
                    "back to top",
                ]:
                    continue  # Footer/nav text
                if re.match(
                    r"^(\[[a-z-]+\]\s+\[[^\]]+\]\([^)]+\)\s*)+$", line_stripped
                ):
                    continue  # Language selectors
                cleaned_lines.append(line)
            text_content = "\n".join(cleaned_lines)
            text_content = re.sub(r"\n{3,}", "\n\n", text_content)
            text_content = re.sub(r"^\s+|\s+$", "", text_content, flags=re.MULTILINE)

        except Exception as bs_err:
            logger.error(
                f"BeautifulSoup parsing/extraction failed for HTML source {source}: {bs_err}. Falling back to raw text.",
                exc_info=True,
            )
            text_content = markdown_text  # Fallback to raw
            processing_mode = "html_fallback_raw"

    else:  # Unknown or unsupported extension
        processing_mode = "unknown_fallback_raw"
        logger.warning(
            f"Unknown file type '{file_ext}' for source '{source}'. Processing as raw text."
        )
        text_content = markdown_text  # Treat as raw text

    # --- Common processing after type-specific handling ---

    # Strip Markdown links [text](url) -> text (Applied to all types now for consistency)
    original_len = len(text_content)
    text_content = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text_content)
    stripped_len = len(text_content)
    if original_len != stripped_len:
        logger.debug(
            f"Stripped Markdown links from '{processing_mode}' content (length {original_len} -> {stripped_len})."
        )

    if not text_content.strip():
        logger.warning(
            f"No text content remaining after {processing_mode} processing and link stripping for source: {source}"
        )
        return []

    logger.debug(
        f"Text content length after {processing_mode} processing and link stripping: {len(text_content)} chars."
    )

    # --- Splitting Logic (Remains the same) ---
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
        length_function=len,  # Use standard len for splitting the processed text
        is_separator_regex=False,
        keep_separator=True,
    )

    try:
        split_texts = text_splitter.split_text(text_content)
        logger.debug(
            f"Initial split of '{processing_mode}' text for {source or 'unknown'} resulted in {len(split_texts)} potential chunks."
        )
    except Exception as e:
        logger.error(
            f"Error splitting processed text from source {source}: {e}", exc_info=True
        )
        return []

    valid_chunks_with_metadata = []
    original_indices_processed = set()

    for i, chunk_text in enumerate(split_texts):
        stripped_chunk = chunk_text.strip()
        # Effective length is calculated on the chunk *after* link stripping was done above
        effective_len = calculate_effective_length(stripped_chunk)
        raw_len = len(stripped_chunk)

        logger.debug(
            f"Chunk {i} candidate (EffectiveLen={effective_len}, RawLen={raw_len}): '{stripped_chunk[:80]}...'"
        )

        if effective_len >= min_chunk_char_length:
            first_line = stripped_chunk.split("\n", 1)[0]
            # Filter chunks that are mostly repetitive lines (like separators)
            if len(first_line) > 1 and len(set(first_line.strip())) <= 2:
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
                "processing_mode": processing_mode,  # Add how it was processed
            }
            valid_chunks_with_metadata.append(
                {"chunk": stripped_chunk, "metadata": metadata}
            )
            original_indices_processed.add(i)
        else:
            if raw_len > 0:
                logger.debug(
                    f"Skipping chunk {i} from {source} due to effective length < {min_chunk_char_length} (EffLen={effective_len}, RawLen={raw_len}): '{stripped_chunk[:50]}...'"
                )

    logger.info(
        f"Split '{processing_mode}' text from {source or 'unknown'} into {len(valid_chunks_with_metadata)} valid chunks (Size: {chunk_size}, Overlap: {chunk_overlap}, MinEffLen: {min_chunk_char_length})"
    )

    if not valid_chunks_with_metadata and text_content.strip():
        logger.warning(
            f"Chunking resulted in zero valid chunks for source: {source} ({processing_mode} processing). Original text length: {len(markdown_text)}, Processed text length: {len(text_content)}. Check chunk parameters and content."
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
