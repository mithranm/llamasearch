import re
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

import markdown
from bs4 import BeautifulSoup, Tag

from llamasearch.utils import setup_logging

logger = setup_logging(__name__, use_qt_handler=True)


class MarkdownChunker:
    """
    Chunks Markdown text based on structural elements (headings, paragraphs, code blocks)
    and size constraints, suitable for RAG document preparation.
    """

    def __init__(
        self,
        min_chunk_size: int = 150,  # Smaller min size might be okay for doc chunks
        max_chunk_size: int = 1000,  # Allow larger chunks for code blocks etc.
        overlap_percent: float = 0.1,  # Overlap based on percentage of max_chunk_size
        combine_under_min_size: bool = True,  # Try to combine small consecutive sections
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
        self.section_break_tags = [
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "hr",
            "pre",
            "table",
        ]
        self.content_tags = ["p", "li", "code"]  # Tags with primary text content

    def _clean_text(self, text: str) -> str:
        """Basic whitespace normalization."""
        text = re.sub(r"\s+", " ", text)  # Normalize whitespace
        return text.strip()

    def _extract_sections(self, markdown_text: str) -> List[str]:
        """Extracts text sections based on HTML structure after Markdown conversion."""
        sections = []
        current_section = []

        try:
            # Use 'extra' for tables, fenced code, etc.
            html = markdown.markdown(
                markdown_text, extensions=["extra", "fenced_code", "tables"]
            )
            soup = BeautifulSoup(html, "html.parser")
        except Exception as e:
            logger.warning(
                f"Failed to parse Markdown to HTML: {e}. Falling back to paragraph splitting."
            )
            # Fallback: split by paragraphs if HTML parsing fails
            paragraphs = re.split(r"\n{2,}", markdown_text)
            sections = [self._clean_text(p) for p in paragraphs if self._clean_text(p)]
            soup = None  # Indicate fallback

        if soup:
            for element in soup.find_all(True, recursive=False):
                if not isinstance(element, Tag):
                    continue

                tag_name = element.name.lower()
                text = self._clean_text(element.get_text())

                if not text:
                    continue

                if tag_name in self.section_break_tags:
                    if current_section:
                        sections.append(" ".join(current_section))
                        current_section = []
                    sections.append(text)  # Add break tag's content as its own section

                elif tag_name in self.content_tags or tag_name == "div":
                    current_section.append(text)

                elif not current_section and text:
                    current_section.append(text)

        if current_section:
            sections.append(" ".join(current_section))

        if self.combine_under_min_size:
            combined_sections = []
            buffer = []
            buffer_len = 0
            for sec in sections:
                sec_len = len(sec)
                if not buffer or (
                    buffer_len + sec_len + 1 < self.max_chunk_size
                    and buffer_len < self.min_chunk_size
                ):
                    buffer.append(sec)
                    buffer_len += sec_len + (1 if buffer else 0)
                else:
                    if buffer_len >= self.min_chunk_size:
                        combined_sections.append(" ".join(buffer))
                    elif buffer:
                        combined_sections.append(" ".join(buffer))

                    buffer = [sec]
                    buffer_len = sec_len
            if buffer_len >= self.min_chunk_size:
                combined_sections.append(" ".join(buffer))
            elif buffer:
                combined_sections.append(" ".join(buffer))
            sections = combined_sections

        return [sec for sec in sections if sec]  # Filter out empty strings

    def chunk_document(
        self, markdown_text: str, source: Optional[str] = None
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Chunks a Markdown document into sections, then splits large sections.
        Yields dicts containing chunk text and basic metadata.
        """
        if not markdown_text:
            return

        sections = self._extract_sections(markdown_text)
        chunk_id_counter = 0  # Index within this specific document

        for section_index, section in enumerate(sections):
            section_len = len(section)

            if (
                section_len <= self.max_chunk_size
                and section_len >= self.min_chunk_size
            ):
                yield {
                    "chunk": section,
                    "metadata": {
                        "source": source or "unknown",
                        "chunk_index_in_doc": chunk_id_counter,
                    },
                }
                chunk_id_counter += 1
            elif section_len < self.min_chunk_size:
                yield {
                    "chunk": section,
                    "metadata": {
                        "source": source or "unknown",
                        "chunk_index_in_doc": chunk_id_counter,
                    },
                }
                chunk_id_counter += 1
            else:  # Section is larger than max_chunk_size, split it
                start = 0
                sub_chunk_index = 0
                while start < section_len:
                    end = min(start + self.max_chunk_size, section_len)
                    best_break = end
                    # Look backwards for sentence breaks near the end
                    possible_breaks = [
                        m.start() + 1
                        for m in re.finditer(
                            r"[.!?\n]\s+",
                            section[max(start, end - self.overlap_size * 2) : end],
                        )
                    ]
                    if possible_breaks:
                        natural_break = max(possible_breaks) + max(
                            start, end - self.overlap_size * 2
                        )  # Adjust index relative to section start
                        if natural_break > start + self.min_chunk_size:
                            best_break = natural_break

                    chunk_text = section[start:best_break].strip()

                    if len(chunk_text) >= self.min_chunk_size or start == 0:
                        yield {
                            "chunk": chunk_text,
                            "metadata": {
                                "source": source or "unknown",
                                "chunk_index_in_doc": chunk_id_counter,
                                "split_from_section_index": section_index,
                                "sub_chunk_index": sub_chunk_index,
                            },
                        }
                        chunk_id_counter += 1
                        sub_chunk_index += 1
                    elif chunk_text:
                        logger.debug(
                            f"Skipping very small chunk ({len(chunk_text)} chars) created during split of large section in {source}"
                        )

                    # Determine next start position with overlap
                    next_start = max(
                        start + self.min_chunk_size, best_break - self.overlap_size
                    )  # Adjust next start considering overlap
                    if next_start >= section_len:
                        break
                    # Ensure we make progress, avoid infinite loop if overlap logic fails
                    if next_start <= start:
                        logger.warning(
                            f"Chunk splitting failed to advance in {source}, section {section_index}. Moving past problematic break."
                        )
                        next_start = best_break  # Force progress past the break point
                    start = next_start

    def process_file(self, file_path: str) -> Generator[Dict[str, Any], None, None]:
        """Reads a Markdown file and yields chunks."""
        path = Path(file_path)
        if not path.is_file() or path.suffix.lower() not in [".md", ".markdown"]:
            logger.warning(f"Chunker skipping non-markdown file: {file_path}")
            return

        try:
            markdown_text = path.read_text(encoding="utf-8")
            yield from self.chunk_document(markdown_text, source=str(path))
        except Exception as e:
            logger.error(f"Error processing file {file_path} with chunker: {e}")
