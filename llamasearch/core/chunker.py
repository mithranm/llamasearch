# llamasearch/core/chunker.py

import os
import re
import gc
import logging
from typing import List, Dict, Any, Generator, Tuple, Optional
import markdown
from bs4 import BeautifulSoup, Tag

logger = logging.getLogger(__name__)


class MarkdownChunker:
    """
    Enhanced markdown chunker that creates smaller, more focused chunks for better retrieval.
    Optimized for memory-constrained environments and improved multilingual support.
    Incorporates code block extraction, paragraph splitting, optional sentence-based splits, etc.
    """

    def __init__(
        self,
        chunk_size: int = 150,
        text_embedding_size: int = 512,
        min_chunk_size: int = 50,
        max_chunks: int = 5000,
        batch_size: int = 1,
        ignore_link_urls: bool = True,
        code_context_window: int = 2,
        include_section_headers: bool = True,
        always_create_chunks: bool = True,
    ):
        self.chunk_size = chunk_size
        self.text_embedding_size = text_embedding_size
        self.min_chunk_size = min_chunk_size
        self.max_chunks = max_chunks
        self.batch_size = batch_size
        self.ignore_link_urls = ignore_link_urls
        self.code_context_window = code_context_window
        self.include_section_headers = include_section_headers
        self.always_create_chunks = always_create_chunks

        logger.info(
            f"Initialized MarkdownChunker with text_embedding_size={text_embedding_size} tokens"
        )

    def process_file_in_batches(
        self, file_path: str, batch_size: Optional[int] = None
    ) -> Generator[List[Dict[str, Any]], None, None]:
        """
        Process a .md file in small batches, each containing chunk dicts
        { "chunk":..., "metadata":..., "embedding_text":... }
        """
        if not file_path.lower().endswith(".md"):
            raise ValueError("Only markdown (.md) files are supported")

        if batch_size is None:
            batch_size = self.batch_size

        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return

        logger.info(f"Processing file: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        logger.info(f"File size: {len(text)} characters")

        all_chunks = list(self.chunk_document(text))
        logger.info(f"File chunking produced {len(all_chunks)} chunks total")

        batch = []
        total_chunks = 0
        for cdict in all_chunks:
            batch.append(cdict)
            if len(batch) >= batch_size:
                yield batch
                total_chunks += len(batch)
                batch = []
                gc.collect()

        if batch:
            yield batch
            total_chunks += len(batch)

        logger.info(
            f"File processing complete: {total_chunks} total chunks, {batch_size} batch size"
        )

    def chunk_document(self, text: str) -> Generator[Dict[str, Any], None, None]:
        """
        The main pipeline that returns chunk dictionaries for embedding.
        Includes code block detection, paragraph splitting, partial splits, etc.
        """
        # 1) Clean text, convert to HTML
        html = self._markdown_to_html(text)
        # 2) Extract code blocks with context
        code_blocks = self._extract_code_blocks(html)
        # yield chunk dicts for each code block
        for code_text, context_text in code_blocks:
            # If code is huge, split into smaller pieces
            lines = code_text.split("\n")
            current_code = ""
            for line in lines:
                if len(current_code) + len(line) + 1 > self.chunk_size and current_code:
                    yield {
                        "chunk": current_code.strip(),
                        "metadata": {"type": "code_block"},
                        "embedding_text": current_code.strip(),
                    }
                    current_code = line
                else:
                    if current_code:
                        current_code += "\n" + line
                    else:
                        current_code = line
            if current_code.strip():
                yield {
                    "chunk": current_code.strip(),
                    "metadata": {"type": "code_block"},
                    "embedding_text": current_code.strip(),
                }

        # 3) remove code blocks from HTML so we don't process them again
        soup = BeautifulSoup(html, "html.parser")
        for pre in soup.find_all("pre"):
            pre.decompose()

        # 4) Extract textual sections
        sections = self._extract_sections(str(soup))
        # For each section, we might have to split if it exceeds chunk_size
        for sec in sections:
            content = sec["content"].strip()
            if not content:
                continue
            if len(content) <= self.chunk_size:
                yield {
                    "chunk": content,
                    "metadata": {
                        "type": "text_chunk",
                        "title": sec.get("title", ""),
                        "level": sec.get("level", 0),
                    },
                    "embedding_text": content,
                }
            else:
                # We do partial splitting by sentences if needed
                splitted = re.split(r"(?<=[.!?])\s+", content)
                current_sent = ""
                for s in splitted:
                    if (
                        len(current_sent) + len(s) + 1 > self.chunk_size
                        and current_sent
                    ):
                        yield {
                            "chunk": current_sent.strip(),
                            "metadata": {
                                "type": "text_chunk",
                                "title": sec.get("title", ""),
                                "level": sec.get("level", 0),
                            },
                            "embedding_text": current_sent.strip(),
                        }
                        current_sent = s
                    else:
                        if current_sent:
                            current_sent += " " + s
                        else:
                            current_sent = s
                if current_sent.strip():
                    yield {
                        "chunk": current_sent.strip(),
                        "metadata": {
                            "type": "text_chunk",
                            "title": sec.get("title", ""),
                            "level": sec.get("level", 0),
                        },
                        "embedding_text": current_sent.strip(),
                    }

    def _markdown_to_html(self, text: str) -> str:
        """
        Convert markdown to HTML using python-markdown.
        """
        return markdown.markdown(
            text,
            extensions=[
                "markdown.extensions.fenced_code",
                "markdown.extensions.tables",
                "markdown.extensions.attr_list",
            ],
        )

    def _extract_code_blocks(self, html: str) -> List[Tuple[str, str]]:
        """
        Extract code blocks from HTML content. Return list of (code_block, context).
        We'll do minimal context extraction around code blocks.
        """
        soup = BeautifulSoup(html, "html.parser")
        code_blocks = []
        pre_tags = soup.find_all("pre")
        for pre in pre_tags:
            # Check if it's a bs4.element.Tag object
            if isinstance(pre, Tag):
                # Use find method which returns a single Tag or None
                code_tag = pre.find("code")
                if code_tag is not None:
                    code_text = code_tag.get_text()
                    # Minimal context approach: preceding paragraph
                    context = ""
                    code_blocks.append((code_text, context))
        return code_blocks

    def _extract_sections(self, html: str) -> List[Dict[str, Any]]:
        """
        Extract text sections from HTML content.
        We'll do a naive approach: find all headers (h1..h6) and gather text until next header.
        If no headers, we treat paragraphs as sections.
        """
        soup = BeautifulSoup(html, "html.parser")
        sections = []
        headers = soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])
        if not headers:
            # fallback: paragraphs
            paras = soup.find_all("p")
            if paras:
                for i, p in enumerate(paras):
                    ptext = p.get_text().strip()
                    if ptext:
                        sections.append(
                            {
                                "content": ptext,
                                "level": 0,
                                "title": f"Paragraph {i+1}",
                            }
                        )
            else:
                # fallback if no paras
                rawtext = soup.get_text().strip()
                if rawtext:
                    sections.append(
                        {
                            "content": rawtext,
                            "level": 0,
                            "title": "Full text",
                        }
                    )
            return sections

        # we do a more advanced approach if headers exist
        all_headers = headers
        for i, hdr in enumerate(all_headers):
            # Check that hdr is a Tag
            if isinstance(hdr, Tag):
                level = int(hdr.name[1])  # h1->1, h2->2...
                title = hdr.get_text().strip()
                # find text until next header
                next_header = all_headers[i + 1] if i < len(all_headers) - 1 else None
                content_el = hdr.next_sibling
                content_list = []

                # Extract content until we hit the next header or run out of siblings
                while content_el and (next_header is None or content_el != next_header):
                    if isinstance(content_el, Tag):
                        content_list.append(content_el.get_text())
                    elif isinstance(content_el, str):
                        content_list.append(content_el.strip())

                    # Safely get the next sibling
                    if hasattr(content_el, "next_sibling"):
                        content_el = content_el.next_sibling
                    else:
                        # Break if we can't navigate further
                        break

                    if i < len(all_headers) - 1 and content_el == next_header:
                        break
                full_content = "\n".join(c.strip() for c in content_list if c.strip())
                sections.append(
                    {"content": full_content, "level": level, "title": title}
                )
        return sections
