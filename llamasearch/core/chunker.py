import os
import re
import gc
import logging
from typing import List, Dict, Any, Generator, Tuple, Optional
import markdown
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MarkdownChunker:
    """
    Enhanced markdown chunker that creates smaller, more focused chunks for better retrieval.
    Optimized for memory-constrained environments and improved multilingual support.
    """

    def __init__(
        self,
        chunk_size: int = 250,  # Smaller chunk size for better retrieval
        text_embedding_size: int = 512,  # Max tokens for text to be embedded
        min_chunk_size: int = 50,  # Minimum size for chunks
        max_chunks: int = 5000,
        batch_size: int = 1,  # Smaller batch size for memory efficiency
        ignore_link_urls: bool = True,
        code_context_window: int = 2,  # Paragraphs before/after code to consider
        include_section_headers: bool = True,
        always_create_chunks: bool = True,  # Always create chunks even if semantic chunking fails
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

        # Regular expressions for content detection
        self.header_pattern = re.compile(r"^(#{1,6})\s+(.*?)$", re.MULTILINE)
        self.code_block_pattern = re.compile(
            r"```(?:[a-zA-Z0-9_+-]+)?\n[\s\S]*?\n```", re.MULTILINE
        )
        self.list_item_pattern = re.compile(r"^\s*(?:\*|\-|\d+\.)\s+.*?$", re.MULTILINE)
        self.table_pattern = re.compile(
            r"^\|(?:.*?\|)+\s*$[\r\n]+\|(?:\s*:?[-]+:?\s*\|)+\s*$", re.MULTILINE
        )
        self.paragraph_pattern = re.compile(r"\n\s*\n", re.MULTILINE)
        self.link_pattern = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")

        # Pattern to remove navigation elements and non-essential content
        self.nav_pattern = re.compile(r"^\s*\*\s+\[.*?\].*?$", re.MULTILINE)

        # Initialize the markdown parser
        self.markdown_extensions = [
            "markdown.extensions.fenced_code",
            "markdown.extensions.tables",
            "markdown.extensions.attr_list",
        ]

        logger.info(
            f"Initialized MarkdownChunker with text_embedding_size={text_embedding_size} tokens"
        )

    def _clean_markdown_links(self, text: str) -> str:
        """
        Replace Markdown links with just their descriptive text.
        [Link Text](URL) becomes just "Link Text"
        """
        if self.ignore_link_urls:
            return self.link_pattern.sub(r"\1", text)
        return text

    def _clean_navigation_elements(self, text: str) -> str:
        """
        Remove navigation elements like menu items and non-essential content
        """
        # Remove navigation lines (lines that are just links)
        text = self.nav_pattern.sub("", text)

        # Remove duplicate empty lines
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text

    def _markdown_to_html(self, text: str) -> str:
        """
        Convert markdown to HTML using python-markdown
        """
        return markdown.markdown(text, extensions=self.markdown_extensions)

    def _extract_sections(self, html: str) -> List[Dict[str, Any]]:
        """
        Extract sections from HTML content using BeautifulSoup.
        Creates smaller, more focused sections for better retrieval.
        """
        soup = BeautifulSoup(html, "html.parser")
        sections = []

        # Extract headers and build section hierarchy
        headers = soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])

        if not headers:
            # If no headers, extract paragraphs as individual sections
            paragraphs = soup.find_all("p")

            if paragraphs:
                for i, para in enumerate(paragraphs):
                    para_text = para.get_text().strip()
                    if para_text:  # Only add non-empty paragraphs
                        sections.append(
                            {
                                "content": str(para),
                                "level": 0,
                                "title": f"Paragraph {i+1}",
                                "path": [f"Paragraph {i+1}"],
                            }
                        )
            else:
                # If no paragraphs either, treat the whole document as one section
                sections.append(
                    {"content": str(soup), "level": 0, "title": "", "path": []}
                )
            return sections

        # Process each header to create sections
        current_path = []
        last_level = 0

        for i, header in enumerate(headers):
            # Get header level (h1 = 1, h2 = 2, etc.)
            level = int(header.name[1])
            title = header.get_text().strip()

            # Skip empty headers
            if not title:
                continue

            # Update path based on header level
            if level <= last_level:
                # Pop levels that are greater than or equal to current level
                current_path = current_path[: -(last_level - level + 1)]

            # Add current header to path
            current_path.append(title)
            last_level = level

            # Get content until next header
            # content = ""
            element = header.next_sibling

            # Gather all elements until the next header
            elements = []
            while element and (i == len(headers) - 1 or element != headers[i + 1]):
                if hasattr(element, "name"):
                    elements.append(element)
                else:
                    # Only add text nodes if they contain non-whitespace content
                    if str(element).strip():
                        elements.append(element)

                if i < len(headers) - 1 and element.next_sibling == headers[i + 1]:
                    break

                element = element.next_sibling
                if not element:
                    break

            # Create smaller chunks from the elements
            # This creates more granular chunks for better retrieval
            current_chunk_elements = []
            current_chunk_size = 0
            max_chunk_size = self.chunk_size  # Characters per chunk

            for element in elements:
                element_text = str(element)
                element_size = len(element_text)

                # If adding this element would exceed the chunk size and we already have elements
                if (
                    current_chunk_elements
                    and current_chunk_size + element_size > max_chunk_size
                ):
                    # Create a section with the current chunk
                    chunk_content = "".join(str(e) for e in current_chunk_elements)
                    if chunk_content.strip():  # Only add non-empty chunks
                        chunk_title = f"{title} (part {len(sections) + 1 - sum(1 for s in sections if s['title'] == title)})"
                        sections.append(
                            {
                                "content": chunk_content,
                                "level": level,
                                "title": title,
                                "path": current_path.copy(),
                                "header": chunk_title,
                            }
                        )

                    # Reset for next chunk
                    current_chunk_elements = [element]
                    current_chunk_size = element_size
                else:
                    # Add to current chunk
                    current_chunk_elements.append(element)
                    current_chunk_size += element_size

            # Add the last chunk if it has content
            if current_chunk_elements:
                chunk_content = "".join(str(e) for e in current_chunk_elements)
                if chunk_content.strip():  # Only add non-empty chunks
                    chunk_title = f"{title} (part {len(sections) + 1 - sum(1 for s in sections if s['title'] == title)})"
                    sections.append(
                        {
                            "content": chunk_content,
                            "level": level,
                            "title": title,
                            "path": current_path.copy(),
                            "header": chunk_title,
                        }
                    )

        return sections

    def _extract_code_blocks(self, html: str) -> List[Tuple[str, str]]:
        """
        Extract code blocks from HTML content with improved context extraction.
        Returns list of (code_block, context_text) pairs with better semantic relevance.
        """
        soup = BeautifulSoup(html, "html.parser")
        code_blocks = []

        # Find all code blocks (pre > code)
        pre_blocks = soup.find_all("pre")

        for pre in pre_blocks:
            code = pre.find("code")
            if code:
                # Get the code block content
                code_content = code.get_text()

                # Skip very small code blocks (likely not meaningful code)
                if len(code_content.strip()) < 10:
                    continue

                # Get context (surrounding paragraphs and headers)
                context = ""

                # Find the closest header for additional context
                header = None
                element = pre
                headers = []
                while element and len(headers) < 2:  # Get up to 2 levels of headers
                    element = element.previous_sibling
                    if element and element.name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
                        headers.insert(0, element)

                # Add headers to context
                if headers and self.include_section_headers:
                    for header in headers:
                        section_header = f"Section: {header.get_text()}\n\n"
                        context = section_header + context

                # Extract relevant context around code block
                element = pre
                context_elements = []

                # Get previous elements for context
                prev_elements = []
                context_paragraphs_before = 0
                element = pre
                while element and context_paragraphs_before < self.code_context_window:
                    element = element.previous_sibling
                    if element and (
                        element.name
                        in [
                            "p",
                            "h1",
                            "h2",
                            "h3",
                            "h4",
                            "h5",
                            "h6",
                            "ul",
                            "ol",
                            "table",
                        ]
                        or (isinstance(element, str) and element.strip())
                    ):
                        prev_elements.append(element)
                        if element.name == "p":
                            context_paragraphs_before += 1

                # Get next elements for context
                next_elements = []
                context_paragraphs_after = 0
                element = pre
                while element and context_paragraphs_after < self.code_context_window:
                    element = element.next_sibling
                    if element and (
                        element.name
                        in [
                            "p",
                            "h1",
                            "h2",
                            "h3",
                            "h4",
                            "h5",
                            "h6",
                            "ul",
                            "ol",
                            "table",
                        ]
                        or (isinstance(element, str) and element.strip())
                    ):
                        next_elements.append(element)
                        if element.name == "p":
                            context_paragraphs_after += 1

                # Combine context elements with preference for elements closer to the code
                context_elements = list(reversed(prev_elements)) + next_elements

                for elem in context_elements:
                    if hasattr(elem, "get_text"):
                        # Headers get priority in the context
                        if elem.name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
                            context += f"HEADER: {elem.get_text().strip()}\n\n"
                        else:
                            context += elem.get_text().strip() + "\n\n"
                    elif isinstance(elem, str) and elem.strip():
                        context += elem.strip() + "\n\n"

                # Add code block and its context
                code_blocks.append(("```\n" + code_content + "\n```", context.strip()))

        return code_blocks

    def _extract_text_chunks(self, html: str) -> List[str]:
        """
        Extract text chunks from HTML content, excluding code blocks.
        Creates smaller, more focused chunks for better retrieval.
        """
        soup = BeautifulSoup(html, "html.parser")

        # Remove all code blocks to avoid duplication
        for pre in soup.find_all("pre"):
            pre.decompose()

        # Get sections with headers
        chunks = []

        # Find all headers
        headers = soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])

        if not headers:
            # If no headers, get all paragraphs and create chunks
            paragraphs = soup.find_all("p")

            # Create chunks from paragraphs
            current_chunk = ""
            current_chunk_size = 0

            for para in paragraphs:
                para_text = para.get_text().strip()
                if not para_text:
                    continue

                # If adding this paragraph would exceed the chunk size
                if (
                    current_chunk
                    and len(current_chunk) + len(para_text) > self.chunk_size
                ):
                    # Add current chunk and start a new one
                    chunks.append(current_chunk)
                    current_chunk = para_text
                else:
                    # Add to current chunk
                    if current_chunk:
                        current_chunk += "\n\n"
                    current_chunk += para_text

            # Add the last chunk if not empty
            if current_chunk:
                chunks.append(current_chunk)
        else:
            # Process each header and its content to create sections
            for i, header in enumerate(headers):
                header_text = header.get_text().strip()

                # Get content until next header
                content = header_text + "\n\n"
                element = header.next_sibling

                # Collect content elements
                content_elements = []
                while element and (i == len(headers) - 1 or element != headers[i + 1]):
                    if hasattr(element, "name") and element.name == "p":
                        content_elements.append(element.get_text().strip())

                    if i < len(headers) - 1 and element.next_sibling == headers[i + 1]:
                        break

                    element = element.next_sibling
                    if not element:
                        break

                # Create chunks from the content
                current_chunk = content
                current_chunk_size = len(content)

                for elem_text in content_elements:
                    # If adding this element would exceed the chunk size
                    if current_chunk_size + len(elem_text) > self.chunk_size:
                        # Add current chunk and start a new one with the header again
                        chunks.append(current_chunk)
                        current_chunk = header_text + "\n\n" + elem_text
                        current_chunk_size = len(current_chunk)
                    else:
                        # Add to current chunk
                        current_chunk += "\n\n" + elem_text
                        current_chunk_size += len(elem_text) + 2

                # Add the last chunk if not just the header
                if current_chunk and current_chunk != header_text + "\n\n":
                    chunks.append(current_chunk)

        return chunks

    def _estimate_token_count(self, text: str) -> int:
        """
        Estimate the number of tokens in a text.
        This is a simple heuristic based on word count.
        """
        # Simple approximation: split by whitespace and count words
        # Most embedding models have roughly 1.3 tokens per word
        words = text.split()
        return int(len(words) * 1.3) + 1  # Add 1 for safety

    def _truncate_to_token_limit(self, text: str) -> str:
        """
        Truncate text to approximately fit within the token limit.
        Preserves complete sentences when possible.
        """
        if self._estimate_token_count(text) <= self.text_embedding_size:
            return text

        # Split into sentences for more granular truncation
        sentences = re.split(r"(?<=[.!?])\s+", text)
        result = ""

        for sentence in sentences:
            # Check if adding this sentence would exceed the limit
            if (
                self._estimate_token_count(result + sentence + " ")
                > self.text_embedding_size
            ):
                break
            result += sentence + " "

        return result.strip()

    def _create_semantic_chunks(self, text: str) -> List[Dict[str, Any]]:
        """
        Create semantic chunks from markdown text using HTML parsing.
        Creates smaller, more focused chunks for better retrieval.
        """
        chunks = []

        # Clean text
        text = self._clean_navigation_elements(text)
        if self.ignore_link_urls:
            text = self._clean_markdown_links(text)

        # Convert markdown to HTML
        html = self._markdown_to_html(text)

        # Extract code blocks with context
        code_blocks = self._extract_code_blocks(html)

        # Create code-text pairs
        for code_block, context in code_blocks:
            # Truncate context text to fit token limit
            truncated_context = self._truncate_to_token_limit(context)

            # Create the pair
            code_text_pair = {
                "text_for_embedding": truncated_context,  # This will be embedded
                "code_block": code_block,  # This will be stored but not embedded
                "combined": f"{truncated_context}\n\n{code_block}",  # This will be returned for queries
                "metadata": {
                    "type": "code_text_pair",
                    "estimated_tokens": self._estimate_token_count(truncated_context),
                },
            }

            chunks.append(code_text_pair)

        # Extract text chunks (smaller than before)
        if not code_blocks or self.always_create_chunks:
            text_sections = self._extract_text_chunks(html)

            for section in text_sections:
                # Skip empty sections
                if not section.strip():
                    continue

                # Truncate to token limit
                truncated_section = self._truncate_to_token_limit(section)

                # Skip sections that are too small
                if len(truncated_section) < self.min_chunk_size:
                    continue

                chunks.append(
                    {
                        "text_for_embedding": truncated_section,
                        "combined": truncated_section,
                        "metadata": {
                            "type": "text_chunk",
                            "estimated_tokens": self._estimate_token_count(
                                truncated_section
                            ),
                        },
                    }
                )

        return chunks

    def _simple_chunk(self, text: str) -> Generator[Dict[str, Any], None, None]:
        """
        Simple chunking method as a fallback.
        Creates smaller, more granular chunks for better multilingual support.
        """
        # Clean links if needed
        if self.ignore_link_urls:
            text = self._clean_markdown_links(text)

        # Split into paragraphs
        paragraphs = text.split("\n\n")
        current_chunk = ""
        current_tokens = 0
        chunk_count = 0

        # If text is very short, just return it as a single chunk
        if len(text) < self.min_chunk_size:
            if text.strip():  # Only if it's not just whitespace
                chunk_count += 1
                yield {
                    "chunk": text,
                    "metadata": {
                        "type": "simple_text",
                        "estimated_tokens": self._estimate_token_count(text),
                    },
                    "embedding_text": text,
                }
                logger.info(
                    f"Created chunk {chunk_count}: {len(text)} chars (simple chunking - very short text)"
                )
            return

        # Find headers in the text
        # headers = self.header_pattern.findall(text)
        # has_headers = len(headers) > 0

        current_header = ""

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:  # Skip empty paragraphs
                continue

            # Check if this is a header
            header_match = self.header_pattern.match(paragraph)
            if header_match:
                # If we have content in the current chunk, yield it
                if current_chunk:
                    chunk_count += 1
                    if chunk_count <= self.max_chunks:
                        yield {
                            "chunk": current_chunk,
                            "metadata": {
                                "type": "simple_text",
                                "estimated_tokens": current_tokens,
                                "header": current_header,
                            },
                            "embedding_text": current_chunk,
                        }
                        logger.info(
                            f"Created chunk {chunk_count}: {len(current_chunk)} chars (simple chunking)"
                        )

                # Update current header
                current_header = header_match.group(2)

                # Start a new chunk with just the header
                current_chunk = paragraph
                current_tokens = self._estimate_token_count(paragraph)
                continue

            para_tokens = self._estimate_token_count(paragraph)

            # If this paragraph is too big on its own, split it by sentences
            if para_tokens > self.text_embedding_size:
                # If we have content in current chunk, yield it first
                if current_chunk:
                    chunk_count += 1
                    if chunk_count <= self.max_chunks:
                        yield {
                            "chunk": current_chunk,
                            "metadata": {
                                "type": "simple_text",
                                "estimated_tokens": current_tokens,
                                "header": current_header,
                            },
                            "embedding_text": current_chunk,
                        }
                        logger.info(
                            f"Created chunk {chunk_count}: {len(current_chunk)} chars (simple chunking)"
                        )
                    current_chunk = ""
                    current_tokens = 0

                # Split large paragraph into sentences
                sentences = re.split(r"(?<=[.!?])\s+", paragraph)
                temp_chunk = current_header + "\n\n" if current_header else ""
                temp_tokens = self._estimate_token_count(temp_chunk)

                for sentence in sentences:
                    sent_tokens = self._estimate_token_count(sentence)

                    # If adding this sentence exceeds limit, yield current and start new
                    if (
                        temp_tokens + sent_tokens > self.text_embedding_size
                        and temp_chunk
                    ):
                        chunk_count += 1
                        if chunk_count <= self.max_chunks:
                            yield {
                                "chunk": temp_chunk,
                                "metadata": {
                                    "type": "simple_text",
                                    "estimated_tokens": temp_tokens,
                                    "header": current_header,
                                },
                                "embedding_text": temp_chunk,
                            }
                            logger.info(
                                f"Created chunk {chunk_count}: {len(temp_chunk)} chars (simple chunking - sentence split)"
                            )

                        # Start new chunk with the header
                        temp_chunk = current_header + "\n\n" if current_header else ""
                        temp_tokens = self._estimate_token_count(temp_chunk)

                        # Add the sentence
                        temp_chunk += sentence
                        temp_tokens += sent_tokens
                    else:
                        # Add to current temp chunk
                        if temp_chunk and not temp_chunk.endswith("\n\n"):
                            temp_chunk += " "
                        temp_chunk += sentence
                        temp_tokens += sent_tokens

                # Don't forget the last sentence chunk
                if temp_chunk and temp_chunk != (
                    current_header + "\n\n" if current_header else ""
                ):
                    chunk_count += 1
                    if chunk_count <= self.max_chunks:
                        yield {
                            "chunk": temp_chunk,
                            "metadata": {
                                "type": "simple_text",
                                "estimated_tokens": temp_tokens,
                                "header": current_header,
                            },
                            "embedding_text": temp_chunk,
                        }
                        logger.info(
                            f"Created chunk {chunk_count}: {len(temp_chunk)} chars (simple chunking - last sentence)"
                        )

            # If adding this paragraph would exceed the token limit
            elif current_tokens + para_tokens > self.text_embedding_size:
                # Yield current chunk
                chunk_count += 1
                if chunk_count <= self.max_chunks:
                    yield {
                        "chunk": current_chunk,
                        "metadata": {
                            "type": "simple_text",
                            "estimated_tokens": current_tokens,
                            "header": current_header,
                        },
                        "embedding_text": current_chunk,  # Same for simple chunks
                    }
                    logger.info(
                        f"Created chunk {chunk_count}: {len(current_chunk)} chars (simple chunking)"
                    )

                # Start a new chunk
                current_chunk = (
                    current_header + "\n\n" if current_header else ""
                ) + paragraph
                current_tokens = self._estimate_token_count(current_chunk)
            else:
                # Add to current chunk
                if current_chunk:
                    current_chunk += "\n\n"
                current_chunk += paragraph
                current_tokens += para_tokens

            # Check if we've hit the chunk limit
            if chunk_count >= self.max_chunks:
                logger.warning(
                    f"Reached maximum chunk limit ({self.max_chunks}). Document chunking stopped."
                )
                return

        # Don't forget the last chunk
        if current_chunk:
            chunk_count += 1
            if chunk_count <= self.max_chunks:
                yield {
                    "chunk": current_chunk,
                    "metadata": {
                        "type": "simple_text",
                        "estimated_tokens": current_tokens,
                        "header": current_header,
                    },
                    "embedding_text": current_chunk,  # Same for simple chunks
                }
                logger.info(
                    f"Created chunk {chunk_count}: {len(current_chunk)} chars (simple chunking)"
                )

    def chunk_document(self, text: str) -> Generator[Dict[str, Any], None, None]:
        """
        Chunk document into smaller, more focused chunks for better retrieval.
        """
        if not text:
            return

        logger.info(f"Chunking document: {len(text)} characters")
        logger.info(
            f"Using semantic chunking with code-text pairing (embedding size: {self.text_embedding_size} tokens)"
        )
        logger.info(f"Maximum chunks limit: {self.max_chunks}")

        # Try semantic chunking first
        try:
            semantic_chunks = self._create_semantic_chunks(text)

            if semantic_chunks and len(semantic_chunks) > 0:
                # Semantic chunking succeeded
                chunk_count = 0

                for chunk in semantic_chunks:
                    chunk_count += 1
                    if chunk_count <= self.max_chunks:
                        chunk_dict = {
                            "chunk": chunk["combined"],
                            "metadata": chunk["metadata"],
                            "embedding_text": chunk["text_for_embedding"],
                        }
                        logger.info(
                            f"Created chunk {chunk_count}: {len(chunk['combined'])} chars (type: {chunk['metadata']['type']})"
                        )
                        yield chunk_dict
                    else:
                        logger.warning(
                            f"Reached maximum chunk limit ({self.max_chunks}). Document chunking stopped."
                        )
                        return

                logger.info(f"Semantic chunking complete: {chunk_count} chunks created")
            else:
                # No semantic chunks created, fall back to simple chunking
                logger.info(
                    "No semantic chunks created, falling back to simple chunking"
                )
                yield from self._simple_chunk(text)

        except Exception as e:
            logger.error(f"Error in semantic chunking: {str(e)}")
            logger.info("Falling back to simple chunking method")
            yield from self._simple_chunk(text)

    def process_file_in_batches(
        self, file_path: str, batch_size: Optional[int] = None
    ) -> Generator[List[Dict[str, Any]], None, None]:
        """
        Process a file in batches to conserve memory.
        """
        if batch_size is None:
            batch_size = self.batch_size

        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return

        logger.info(f"Processing file: {file_path}")
        total_batches = 0
        total_chunks = 0

        try:
            # Read file content
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()

            logger.info(f"File size: {len(text)} characters")

            # Process in batches
            batch = []

            for chunk_dict in self.chunk_document(text):
                batch.append(chunk_dict)
                total_chunks += 1

                if len(batch) >= batch_size:
                    total_batches += 1
                    logger.info(
                        f"Yielding batch {total_batches} with {len(batch)} chunks (total chunks: {total_chunks})"
                    )
                    yield batch

                    # Clear memory immediately
                    batch = []
                    gc.collect()

                # Check if we've hit the chunk limit
                if total_chunks >= self.max_chunks:
                    logger.warning(
                        f"Reached maximum chunk limit ({self.max_chunks}). Stopping batch processing."
                    )
                    if batch:  # Yield any remaining chunks in the batch
                        total_batches += 1
                        logger.info(
                            f"Yielding final batch {total_batches} with {len(batch)} chunks (total chunks: {total_chunks})"
                        )
                        yield batch
                    break

            # Yield any remaining chunks
            if batch:
                total_batches += 1
                logger.info(
                    f"Yielding final batch {total_batches} with {len(batch)} chunks (total chunks: {total_chunks})"
                )
                yield batch

            logger.info(
                f"File processing complete: {total_batches} batches, {total_chunks} total chunks"
            )

            # Clean up
            del text
            gc.collect()

        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            raise
