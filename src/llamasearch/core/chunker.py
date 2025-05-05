# src/killeraiagent/utils/markdown_chunker.py

import re
import logging
from pathlib import Path
from typing import List, Dict, Generator, Optional, Any
import markdown
from bs4 import BeautifulSoup, Tag

logger = logging.getLogger(__name__)

class MarkdownChunker:
    """
    Chunks Markdown text based on structural elements (headings, paragraphs, code blocks)
    and size constraints, suitable for RAG document preparation.
    """
    def __init__(
        self,
        min_chunk_size: int = 150,   # Smaller min size might be okay for doc chunks
        max_chunk_size: int = 1000,  # Allow larger chunks for code blocks etc.
        overlap_percent: float = 0.1, # Overlap based on percentage of max_chunk_size
        combine_under_min_size: bool = True, # Try to combine small consecutive sections
    ):
        if not (0 <= overlap_percent < 1):
            raise ValueError("overlap_percent must be between 0 and 1 (exclusive of 1)")
        if min_chunk_size <= 0 or max_chunk_size <= 0:
            raise ValueError("Chunk sizes must be positive")
        if min_chunk_size > max_chunk_size:
            raise ValueError("min_chunk_size cannot exceed max_chunk_size")

        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.overlap_size = int(max_chunk_size * overlap_percent)
        self.combine_under_min_size = combine_under_min_size

        # Tags indicating potential section breaks or distinct content blocks
        self.section_break_tags = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'hr', 'pre', 'table']
        self.content_tags = ['p', 'li', 'code'] # Tags with primary text content

    def _clean_text(self, text: str) -> str:
        """ Basic whitespace normalization. """
        text = re.sub(r'\s+', ' ', text) # Normalize whitespace
        return text.strip()

    def _extract_sections(self, markdown_text: str) -> List[str]:
        """ Extracts text sections based on HTML structure after Markdown conversion. """
        sections = []
        current_section = []

        try:
            # Use 'extra' for tables, fenced code, etc.
            html = markdown.markdown(markdown_text, extensions=['extra', 'fenced_code', 'tables'])
            soup = BeautifulSoup(html, "html.parser")
        except Exception as e:
            logger.warning(f"Failed to parse Markdown to HTML: {e}. Falling back to paragraph splitting.")
            # Fallback: split by paragraphs if HTML parsing fails
            paragraphs = re.split(r'\n{2,}', markdown_text)
            sections = [self._clean_text(p) for p in paragraphs if self._clean_text(p)]
            # Skip the HTML processing loop if fallback occurred
            soup = None # Indicate fallback

        # Iterate through all direct children of the body if HTML parsing succeeded
        if soup:
            for element in soup.find_all(True, recursive=False):
                # Ensure it's a Tag object before accessing .name
                if not isinstance(element, Tag):
                    continue

                tag_name = element.name.lower()
                text = self._clean_text(element.get_text())

                if not text:
                    continue
                # Duplicate check removed below

                # If it's a section break tag, finalize the current section (if any)
                # and start a new section with this element.
                if tag_name in self.section_break_tags:
                    if current_section:
                        sections.append(" ".join(current_section))
                        current_section = []
                    sections.append(text) # Add the break tag's content as its own section

                # If it's a primary content tag, add it to the current section
                elif tag_name in self.content_tags or tag_name == 'div': # Include divs as they might contain mixed content
                    current_section.append(text)

                # Handle other tags (like ul, ol directly under body) - extract their text
                elif not current_section and text: # If buffer is empty, start with this text
                    current_section.append(text)

        # Add any remaining content in the buffer
        if current_section:
            sections.append(" ".join(current_section))

        # Combine small consecutive sections if enabled
        if self.combine_under_min_size:
            combined_sections = []
            buffer = []
            buffer_len = 0
            for sec in sections:
                sec_len = len(sec)
                # If buffer is empty OR adding section doesn't exceed max AND buffer is currently too small
                if not buffer or \
                   (buffer_len + sec_len + 1 < self.max_chunk_size and buffer_len < self.min_chunk_size):
                    buffer.append(sec)
                    buffer_len += sec_len + (1 if buffer else 0)
                else:
                    # Yield the buffer (if long enough) and start new buffer
                    if buffer_len >= self.min_chunk_size:
                        combined_sections.append(" ".join(buffer))
                    elif buffer: # If buffer too small but not empty, add it as is
                        combined_sections.append(" ".join(buffer))

                    buffer = [sec]
                    buffer_len = sec_len
            # Add last buffer
            if buffer_len >= self.min_chunk_size:
                 combined_sections.append(" ".join(buffer))
            elif buffer: # Add even if small
                 combined_sections.append(" ".join(buffer))
            sections = combined_sections


        return [sec for sec in sections if sec] # Filter out any empty strings


    def chunk_document(self, markdown_text: str, source: Optional[str] = None) -> Generator[Dict[str, Any], None, None]:
        """
        Chunks a Markdown document into sections, then splits large sections.
        """
        if not markdown_text:
            return

        sections = self._extract_sections(markdown_text)
        chunk_id_counter = 0

        for section in sections:
            section_len = len(section)

            # If the section is within the desired size range
            if section_len <= self.max_chunk_size and section_len >= self.min_chunk_size:
                yield {
                    "chunk": section,
                    "metadata": {"source": source or "unknown", "chunk_id": chunk_id_counter},
                    "embedding_text": section
                }
                chunk_id_counter += 1
            # If the section is smaller than the minimum chunk size (and wasn't combined)
            elif section_len < self.min_chunk_size:
                 # Yield small sections as they are - combining happened earlier if enabled
                 yield {
                     "chunk": section,
                     "metadata": {"source": source or "unknown", "chunk_id": chunk_id_counter},
                     "embedding_text": section
                 }
                 chunk_id_counter += 1
            # If the section is larger than the maximum chunk size, split it
            else:
                start = 0
                while start < section_len:
                    end = min(start + self.max_chunk_size, section_len)

                    # Try to find a natural break point (like sentence end) near the end
                    # Look backwards from 'end' for '.', '!', '?', '\n'
                    best_break = end
                    possible_breaks = [m.start() + 1 for m in re.finditer(r'[.!?\n]\s+', section[max(start, end - self.overlap_size * 2):end])]
                    if possible_breaks:
                        natural_break = max(possible_breaks) # Furthest break point within window
                        if natural_break > start + self.min_chunk_size: # Ensure chunk isn't too small
                            best_break = natural_break

                    chunk_text = section[start:best_break].strip()

                    if len(chunk_text) >= self.min_chunk_size or start == 0: # Ensure first chunk is yielded even if small
                         yield {
                             "chunk": chunk_text,
                             "metadata": {"source": source or "unknown", "chunk_id": chunk_id_counter},
                             "embedding_text": chunk_text
                         }
                         chunk_id_counter += 1
                    elif chunk_text: # If we created a tiny chunk due to splitting logic, log it
                        logger.debug(f"Skipping very small chunk ({len(chunk_text)} chars) created during split of large section in {source}")


                    # Determine next start position with overlap
                    # Ensure overlap doesn't make us go backwards significantly
                    next_start = max(start + self.min_chunk_size, best_break - self.overlap_size)
                    if next_start >= section_len: # Prevent infinite loop if overlap pushes us past the end
                        break
                    start = next_start


    def process_file(self, file_path: str) -> Generator[Dict[str, Any], None, None]:
        """ Reads a Markdown file and yields chunks. """
        path = Path(file_path)
        if not path.is_file() or path.suffix.lower() != ".md":
            logger.warning(f"Skipping non-markdown file: {file_path}")
            return

        try:
            markdown_text = path.read_text(encoding="utf-8")
            yield from self.chunk_document(markdown_text, source=str(path))
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")