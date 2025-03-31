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
        chunk_size: int = 150,  # Smaller chunk size for better retrieval
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
        self.header_pattern = re.compile(r"^(#{1,6})\s+(.+?)$", re.MULTILINE)
        self.code_block_pattern = re.compile(
            r"```(?:[a-zA-Z0-9_+-]*)\n([\s\S]*?)\n```", re.MULTILINE
        )
        self.list_item_pattern = re.compile(r"^\s*(?:[*\-]|\d+\.)\s+(.+?)$", re.MULTILINE)
        self.table_pattern = re.compile(
            r"^\|(?:[^|]*\|)+\s*$(?:\r?\n\|(?:\s*:?[-]+:?\s*\|)+\s*$)?", re.MULTILINE
        )
        self.paragraph_pattern = re.compile(r"\n\s*\n", re.MULTILINE)
        self.link_pattern = re.compile(r"\[([^]]+)\]\(([^)]+)\)")
        self.name_pattern = re.compile(r'\b[A-Z][a-z]+(?: [A-Z][a-z]+)+\b')

        # Pattern to remove navigation elements and non-essential content
        self.nav_pattern = re.compile(r"^\s*\*\s+\[.*?\].*?$", re.MULTILINE)
        
        # Pattern to remove video player and media content
        self.video_player_pattern = re.compile(
            r'(?s)'  # Enable dot-all mode for the entire pattern
            r'Video Player.*?End of dialog window|'  # Video player content
            r'Beginning of dialog window.*?End of dialog window|'  # Dialog content
            r'Share\s*Settings.*?$|'  # Share settings
            r'!\[.*?\]\(.*?cdn.*?(?:jpg|png|gif).*?\).*?$',  # CDN media embeds
            re.MULTILINE
        )

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
            # Use a lambda function to handle the substitution
            return self.link_pattern.sub(lambda m: m.group(1), text)
        return text

    def _clean_navigation_elements(self, text: str) -> str:
        """
        Remove navigation elements, video players, and other non-essential content
        """
        # Remove navigation lines (lines that are just links)
        text = self.nav_pattern.sub("", text)
        
        # Remove video player and media content
        text = self.video_player_pattern.sub("", text)

        # Remove duplicate empty lines
        text = re.sub(r"\n{3,}", "\n\n", text)
        
        # Clean up any remaining empty lines around headers
        text = re.sub(r'\n+#', '\n#', text)
        text = re.sub(r'(#.*?)\n\n+', r'\1\n', text)

        return text.strip()

    def _clean_overlap_text(self, text: str) -> str:
        """
        Clean overlap text by:
        1. Removing markdown links and keeping only the link text
        2. Removing whitespace, newlines, and tabs
        Returns cleaned text that can be used for meaningful overlap calculations.
        """
        # First clean markdown links
        text = self._clean_markdown_links(text)
        
        # Then remove whitespace, newlines and tabs
        cleaned = re.sub(r'[\s\n\t]+', '', text)
        return cleaned

    def _markdown_to_html(self, text: str) -> str:
        """
        Convert markdown to HTML using python-markdown
        """
        return markdown.markdown(text, extensions=self.markdown_extensions)

    def _extract_sections(self, html: str) -> List[Dict[str, Any]]:
        """
        Extract sections from HTML content using BeautifulSoup.
        Creates smaller, more focused sections for better retrieval.
        Now enforces strict chunk size limits.
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
                        # Check if paragraph exceeds chunk size
                        if len(para_text) > self.chunk_size:
                            # Split into sentences
                            sentences = re.split(r'(?<=[.!?])\s+', para_text)
                            current_chunk = ""
                            
                            for sentence in sentences:
                                # If adding this sentence would exceed the chunk_size
                                if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                                    # Add the current chunk as a section
                                    sections.append({
                                        "content": current_chunk,
                                        "level": 0,
                                        "title": f"Paragraph {i+1}",
                                        "path": [f"Paragraph {i+1}"],
                                    })
                                    current_chunk = sentence
                                else:
                                    # Add to current chunk
                                    if current_chunk:
                                        current_chunk += " "
                                    current_chunk += sentence
                            
                            # Add the last chunk if not empty
                            if current_chunk:
                                sections.append({
                                    "content": current_chunk,
                                    "level": 0,
                                    "title": f"Paragraph {i+1}",
                                    "path": [f"Paragraph {i+1}"],
                                })
                        else:
                            # Paragraph fits within chunk size
                            sections.append({
                                "content": para_text,
                                "level": 0,
                                "title": f"Paragraph {i+1}",
                                "path": [f"Paragraph {i+1}"],
                            })
            else:
                # If no paragraphs either, split the content by character chunks
                full_text = soup.get_text().strip()
                if full_text:
                    # Split into chunks respecting sentence boundaries where possible
                    sentences = re.split(r'(?<=[.!?])\s+', full_text)
                    current_chunk = ""
                    chunk_count = 0
                    
                    for sentence in sentences:
                        # If this sentence alone exceeds chunk_size, split it
                        if len(sentence) > self.chunk_size:
                            # If we have a current chunk, add it first
                            if current_chunk:
                                chunk_count += 1
                                sections.append({
                                    "content": current_chunk,
                                    "level": 0,
                                    "title": f"Section {chunk_count}",
                                    "path": [f"Section {chunk_count}"],
                                })
                                current_chunk = ""
                            
                            # Then split this long sentence by character
                            for j in range(0, len(sentence), self.chunk_size):
                                chunk_count += 1
                                sentence_chunk = sentence[j:j+self.chunk_size]
                                sections.append({
                                    "content": sentence_chunk,
                                    "level": 0,
                                    "title": f"Section {chunk_count}",
                                    "path": [f"Section {chunk_count}"],
                                })
                        # If adding this sentence would exceed chunk_size
                        elif len(current_chunk) + len(sentence) > self.chunk_size:
                            # Add current chunk and start a new one
                            chunk_count += 1
                            sections.append({
                                "content": current_chunk,
                                "level": 0,
                                "title": f"Section {chunk_count}",
                                "path": [f"Section {chunk_count}"],
                            })
                            current_chunk = sentence
                        else:
                            # Add sentence to current chunk
                            if current_chunk:
                                current_chunk += " "
                            current_chunk += sentence
                    
                    # Add the last chunk if not empty
                    if current_chunk:
                        chunk_count += 1
                        sections.append({
                            "content": current_chunk,
                            "level": 0,
                            "title": f"Section {chunk_count}",
                            "path": [f"Section {chunk_count}"],
                        })
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

                # If this single element is larger than max_chunk_size
                if element_size > max_chunk_size:
                    # Process current chunk first if it has content
                    if current_chunk_elements:
                        chunk_content = "".join(str(e) for e in current_chunk_elements)
                        if chunk_content.strip():
                            chunk_title = f"{title} (part {len(sections) + 1 - sum(1 for s in sections if s['title'] == title)})"
                            sections.append({
                                "content": chunk_content,
                                "level": level,
                                "title": title,
                                "path": current_path.copy(),
                                "header": chunk_title,
                            })
                        current_chunk_elements = []
                        current_chunk_size = 0
                    
                    # Split this large element into smaller pieces
                    element_text = element.get_text() if hasattr(element, "get_text") else str(element)
                    # Try to split on sentence boundaries
                    sentences = re.split(r'(?<=[.!?])\s+', element_text)
                    
                    current_sentence_chunk = ""
                    for sentence in sentences:
                        # If this single sentence is too large
                        if len(sentence) > max_chunk_size:
                            # Add current sentence chunk if it exists
                            if current_sentence_chunk:
                                chunk_title = f"{title} (part {len(sections) + 1 - sum(1 for s in sections if s['title'] == title)})"
                                sections.append({
                                    "content": current_sentence_chunk,
                                    "level": level,
                                    "title": title,
                                    "path": current_path.copy(),
                                    "header": chunk_title,
                                })
                                current_sentence_chunk = ""
                            
                            # Split the long sentence by character
                            for j in range(0, len(sentence), max_chunk_size):
                                sub_chunk = sentence[j:j+max_chunk_size]
                                chunk_title = f"{title} (part {len(sections) + 1 - sum(1 for s in sections if s['title'] == title)})"
                                sections.append({
                                    "content": sub_chunk,
                                    "level": level,
                                    "title": title,
                                    "path": current_path.copy(),
                                    "header": chunk_title,
                                })
                        # If adding this sentence would exceed chunk size
                        elif len(current_sentence_chunk) + len(sentence) > max_chunk_size:
                            # Add current sentence chunk
                            chunk_title = f"{title} (part {len(sections) + 1 - sum(1 for s in sections if s['title'] == title)})"
                            sections.append({
                                "content": current_sentence_chunk,
                                "level": level,
                                "title": title,
                                "path": current_path.copy(),
                                "header": chunk_title,
                            })
                            current_sentence_chunk = sentence
                        else:
                            # Add to current sentence chunk
                            if current_sentence_chunk:
                                current_sentence_chunk += " "
                            current_sentence_chunk += sentence
                    
                    # Add the last sentence chunk if it exists
                    if current_sentence_chunk:
                        chunk_title = f"{title} (part {len(sections) + 1 - sum(1 for s in sections if s['title'] == title)})"
                        sections.append({
                            "content": current_sentence_chunk,
                            "level": level,
                            "title": title,
                            "path": current_path.copy(),
                            "header": chunk_title,
                        })
                # If adding this element would exceed the chunk size and we already have elements
                elif current_chunk_elements and current_chunk_size + element_size > max_chunk_size:
                    # Create a section with the current chunk
                    chunk_content = "".join(str(e) for e in current_chunk_elements)
                    if chunk_content.strip():  # Only add non-empty chunks
                        chunk_title = f"{title} (part {len(sections) + 1 - sum(1 for s in sections if s['title'] == title)})"
                        sections.append({
                            "content": chunk_content,
                            "level": level,
                            "title": title,
                            "path": current_path.copy(),
                            "header": chunk_title,
                        })

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
                    sections.append({
                        "content": chunk_content,
                        "level": level,
                        "title": title,
                        "path": current_path.copy(),
                        "header": chunk_title,
                    })

        # Final verification to ensure all chunks are under the maximum size
        final_sections = []
        for section in sections:
            content = section["content"]
            # If still too large, break it down further
            if len(content) > self.chunk_size:
                # Split content by sentences or by characters if needed
                sentences = re.split(r'(?<=[.!?])\s+', content)
                current_chunk = ""
                
                for sentence in sentences:
                    # If this sentence alone is too large
                    if len(sentence) > self.chunk_size:
                        # Add current chunk first if it exists
                        if current_chunk:
                            section_copy = section.copy()
                            section_copy["content"] = current_chunk
                            final_sections.append(section_copy)
                            current_chunk = ""
                        
                        # Then split this large sentence into character chunks
                        for j in range(0, len(sentence), self.chunk_size):
                            section_copy = section.copy()
                            section_copy["content"] = sentence[j:j+self.chunk_size]
                            final_sections.append(section_copy)
                    # If adding this sentence would exceed chunk size
                    elif len(current_chunk) + len(sentence) > self.chunk_size:
                        # Add current chunk and start a new one
                        section_copy = section.copy()
                        section_copy["content"] = current_chunk
                        final_sections.append(section_copy)
                        current_chunk = sentence
                    else:
                        # Add to current chunk
                        if current_chunk:
                            current_chunk += " "
                        current_chunk += sentence
                
                # Add the last chunk if it exists
                if current_chunk:
                    section_copy = section.copy()
                    section_copy["content"] = current_chunk
                    final_sections.append(section_copy)
            else:
                # Section is already within size limit
                final_sections.append(section)
                
        return final_sections

    def _extract_code_blocks(self, html: str) -> List[Tuple[str, str]]:
        """
        Extract code blocks from HTML content with improved context extraction.
        Returns list of (code_block, context_text) pairs with better semantic relevance.
        Now enforces strict chunk size limits.
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
                
                # Enforce size limits for code and context
                if len(code_content) > self.chunk_size:
                    # If code is too large, break it into smaller chunks
                    lines = code_content.split("\n")
                    current_chunk = ""
                    line_count = 0
                    
                    for line in lines:
                        # If adding this line would exceed chunk size
                        if len(current_chunk) + len(line) + 1 > self.chunk_size and current_chunk:
                            # Add current chunk with context
                            trimmed_context = context[:self.chunk_size] if len(context) > self.chunk_size else context
                            code_blocks.append((f"```\n{current_chunk}\n```", trimmed_context))
                            current_chunk = line
                        else:
                            # Add to current chunk
                            if current_chunk:
                                current_chunk += "\n"
                            current_chunk += line
                    
                    # Add the last chunk if not empty
                    if current_chunk:
                        trimmed_context = context[:self.chunk_size] if len(context) > self.chunk_size else context
                        code_blocks.append((f"```\n{current_chunk}\n```", trimmed_context))
                else:
                    # Code fits within chunk size, check context
                    trimmed_context = context[:self.chunk_size] if len(context) > self.chunk_size else context
                    code_blocks.append((f"```\n{code_content}\n```", trimmed_context))

        return code_blocks

    def _extract_text_chunks(self, html: str) -> List[str]:
        """
        Extract text chunks from HTML content, excluding code blocks.
        Creates smaller, more focused chunks for better retrieval.
        Now enforces strict chunk size limits.
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

                # If this paragraph is too large by itself
                if len(para_text) > self.chunk_size:
                    # Add current chunk first if it's not empty
                    if current_chunk:
                        chunks.append(current_chunk)
                        current_chunk = ""
                        current_chunk_size = 0
                    
                    # Split paragraph into sentences
                    sentences = re.split(r'(?<=[.!?])\s+', para_text)
                    current_sentence_chunk = ""
                    
                    for sentence in sentences:
                        # If this single sentence is too large
                        if len(sentence) > self.chunk_size:
                            # Add current sentence chunk first
                            if current_sentence_chunk:
                                chunks.append(current_sentence_chunk)
                                current_sentence_chunk = ""
                            
                            # Split the sentence into smaller chunks
                            for i in range(0, len(sentence), self.chunk_size):
                                sub_chunk = sentence[i:i+self.chunk_size]
                                chunks.append(sub_chunk)
                        # If adding this sentence would exceed chunk size
                        elif len(current_sentence_chunk) + len(sentence) > self.chunk_size:
                            chunks.append(current_sentence_chunk)
                            current_sentence_chunk = sentence
                        else:
                            # Add to current sentence chunk
                            if current_sentence_chunk:
                                current_sentence_chunk += " "
                            current_sentence_chunk += sentence
                    
                    # Add the last sentence chunk if not empty
                    if current_sentence_chunk:
                        chunks.append(current_sentence_chunk)
                # If adding this paragraph would exceed the chunk size
                elif current_chunk and current_chunk_size + len(para_text) > self.chunk_size:
                    # Add current chunk and start a new one
                    chunks.append(current_chunk)
                    current_chunk = para_text
                    current_chunk_size = len(para_text)
                else:
                    # Add to current chunk
                    if current_chunk:
                        current_chunk += "\n\n"
                    current_chunk += para_text
                    current_chunk_size += len(para_text) + 2  # +2 for the newlines

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
                    # If this element alone is too large
                    if len(elem_text) > self.chunk_size:
                        # Add current chunk first if it's not just the header
                        if current_chunk != content:
                            chunks.append(current_chunk)
                        
                        # Process the large element by sentences
                        sentences = re.split(r'(?<=[.!?])\s+', elem_text)
                        current_sentence_chunk = header_text + "\n\n"  # Start with header
                        
                        for sentence in sentences:
                            # If this single sentence is too large
                            if len(sentence) > self.chunk_size - len(header_text) - 2:
                                # Add current sentence chunk first
                                if current_sentence_chunk != header_text + "\n\n":
                                    chunks.append(current_sentence_chunk)
                                
                                # Split the sentence into smaller chunks, each with header
                                for j in range(0, len(sentence), self.chunk_size - len(header_text) - 2):
                                    sub_chunk = header_text + "\n\n" + sentence[j:j+self.chunk_size - len(header_text) - 2]
                                    chunks.append(sub_chunk)
                                
                                current_sentence_chunk = header_text + "\n\n"
                            # If adding this sentence would exceed chunk size
                            elif len(current_sentence_chunk) + len(sentence) > self.chunk_size:
                                chunks.append(current_sentence_chunk)
                                current_sentence_chunk = header_text + "\n\n" + sentence
                            else:
                                # Add to current sentence chunk
                                if current_sentence_chunk != header_text + "\n\n":
                                    current_sentence_chunk += " "
                                else:
                                    current_sentence_chunk += ""
                                current_sentence_chunk += sentence
                        
                        # Add the last sentence chunk if not just the header
                        if current_sentence_chunk != header_text + "\n\n":
                            chunks.append(current_sentence_chunk)
                        
                        # Reset to just the header for the next element
                        current_chunk = content
                        current_chunk_size = len(content)
                    # If adding this element would exceed the chunk size
                    elif current_chunk_size + len(elem_text) > self.chunk_size:
                        # Add current chunk and start a new one with the header again
                        chunks.append(current_chunk)
                        current_chunk = header_text + "\n\n" + elem_text
                        current_chunk_size = len(current_chunk)
                    else:
                        # Add to current chunk
                        current_chunk += "\n\n" + elem_text
                        current_chunk_size += len(elem_text) + 2

                # Add the last chunk if not just the header
                if current_chunk != content:
                    chunks.append(current_chunk)

        return chunks

    def _extract_markdown_paragraphs(self, text: str) -> List[str]:
        """
        Extract paragraphs directly from markdown text before any HTML conversion.
        This preserves the original markdown structure and semantic continuity.
        
        Strategy:
        1. First identify block-level elements (headers, code blocks, lists)
        2. Then extract paragraphs while preserving their markdown formatting
        3. Keep track of nesting level to maintain context
        """
        # Store positions of block-level elements
        block_positions = []
        
        # Find all block-level elements
        for pattern in [
            self.header_pattern,
            self.code_block_pattern,
            self.list_item_pattern,
            self.table_pattern
        ]:
            for match in pattern.finditer(text):
                block_positions.append((match.start(), match.end()))
        
        # Sort block positions
        block_positions.sort()
        
        # Extract paragraphs between block elements
        paragraphs = []
        last_end = 0
        
        for start, end in block_positions:
            # Check if there's paragraph content before this block
            if last_end < start:
                potential_para = text[last_end:start].strip()
                if potential_para:
                    # Only split on actual paragraph breaks (double newline)
                    # This preserves single newlines within paragraphs
                    para_splits = re.split(r'\n\s*\n', potential_para)
                    paragraphs.extend(p.strip() for p in para_splits if p.strip())
            
            # Add the block itself as a chunk
            block_text = text[start:end].strip()
            
            # If block is too large, split it
            if len(block_text) > self.chunk_size:
                # Is it a code block?
                if block_text.startswith("```") and block_text.endswith("```"):
                    # Extract language if present
                    first_line_end = block_text.find("\n")
                    language = block_text[3:first_line_end].strip() if first_line_end > 3 else ""
                    
                    # Split the code content
                    code_content = block_text[first_line_end+1:-3] if first_line_end > 0 else block_text[3:-3]
                    code_lines = code_content.split("\n")
                    
                    current_code_chunk = "```" + language + "\n"
                    for line in code_lines:
                        if len(current_code_chunk) + len(line) + 5 > self.chunk_size:  # +5 for newline and closing ```
                            current_code_chunk += "\n```"
                            paragraphs.append(current_code_chunk)
                            current_code_chunk = "```" + language + "\n" + line
                        else:
                            current_code_chunk += line + "\n"
                    
                    if current_code_chunk != "```" + language + "\n":
                        current_code_chunk += "```"
                        paragraphs.append(current_code_chunk)
                else:
                    # Regular block - split by sentences
                    sentences = re.split(r'(?<=[.!?])\s+', block_text)
                    current_block_chunk = ""
                    
                    for sentence in sentences:
                        # If this sentence alone is too large
                        if len(sentence) > self.chunk_size:
                            # Add current chunk first
                            if current_block_chunk:
                                paragraphs.append(current_block_chunk)
                            
                            # Split large sentence
                            for i in range(0, len(sentence), self.chunk_size):
                                paragraphs.append(sentence[i:i+self.chunk_size])
                            
                            current_block_chunk = ""
                        # If adding this sentence would exceed chunk size
                        elif len(current_block_chunk) + len(sentence) + 1 > self.chunk_size:
                            paragraphs.append(current_block_chunk)
                            current_block_chunk = sentence
                        else:
                            # Add to current chunk
                            if current_block_chunk:
                                current_block_chunk += " "
                            current_block_chunk += sentence
                    
                    # Add the last chunk
                    if current_block_chunk:
                        paragraphs.append(current_block_chunk)
            else:
                # Block fits within chunk size
                paragraphs.append(block_text)
            
            last_end = end
        
        # Don't forget text after the last block
        if last_end < len(text):
            potential_para = text[last_end:].strip()
            if potential_para:
                para_splits = re.split(r'\n\s*\n', potential_para)
                paragraphs.extend(p.strip() for p in para_splits if p.strip())
        
        # Final check to ensure all paragraphs are within size limit
        final_paragraphs = []
        for para in paragraphs:
            if len(para) <= self.chunk_size:
                final_paragraphs.append(para)
            else:
                # Split by sentences
                sentences = re.split(r'(?<=[.!?])\s+', para)
                current_para = ""
                
                for sentence in sentences:
                    # If this sentence alone is too large
                    if len(sentence) > self.chunk_size:
                        # Add current paragraph first
                        if current_para:
                            final_paragraphs.append(current_para)
                        
                        # Split sentence into character chunks
                        for i in range(0, len(sentence), self.chunk_size):
                            final_paragraphs.append(sentence[i:i+self.chunk_size])
                        
                        current_para = ""
                    # If adding this sentence would exceed chunk size
                    elif len(current_para) + len(sentence) + 1 > self.chunk_size:
                        final_paragraphs.append(current_para)
                        current_para = sentence
                    else:
                        # Add to current paragraph
                        if current_para:
                            current_para += " "
                        current_para += sentence
                
                # Add the last paragraph
                if current_para:
                    final_paragraphs.append(current_para)
        
        return final_paragraphs

    def chunk_text(self, text: str) -> List[str]:
        """
        Enhanced chunking that preserves markdown structure.
        Now enforces strict chunk size limits.
        """
        # Clean navigation and non-essential content first
        text = self._clean_navigation_elements(text)
        
        # Extract paragraphs at markdown level
        paragraphs = self._extract_markdown_paragraphs(text)
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for para in paragraphs:
            para_size = len(para)
            
            # If this paragraph alone exceeds chunk_size, it needs splitting
            if para_size > self.chunk_size:
                # First add the current chunk if it exists
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                    current_chunk = []
                    current_size = 0
                
                # Split the large paragraph (by sentences if possible)
                sentences = re.split(r'(?<=[.!?])\s+', para)
                
                if len(sentences) > 1:
                    # Paragraph has multiple sentences
                    current_sentence_group = ""
                    
                    for sentence in sentences:
                        # If this sentence alone is too big
                        if len(sentence) > self.chunk_size:
                            # Add current sentence group first
                            if current_sentence_group:
                                chunks.append(current_sentence_group)
                                current_sentence_group = ""
                            
                            # Split the sentence into character chunks
                            for i in range(0, len(sentence), self.chunk_size):
                                chunks.append(sentence[i:i+self.chunk_size])
                        # If adding this sentence would exceed chunk size
                        elif len(current_sentence_group) + len(sentence) + 1 > self.chunk_size:
                            chunks.append(current_sentence_group)
                            current_sentence_group = sentence
                        else:
                            # Add to current sentence group
                            if current_sentence_group:
                                current_sentence_group += " "
                            current_sentence_group += sentence
                    
                    # Add the last sentence group
                    if current_sentence_group:
                        chunks.append(current_sentence_group)
                else:
                    # Just one long sentence, split by characters
                    for i in range(0, para_size, self.chunk_size):
                        end = min(i + self.chunk_size, para_size)
                        chunks.append(para[i:end])
            # If adding this paragraph would exceed chunk_size
            elif current_size + para_size > self.chunk_size and current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = [para]
                current_size = para_size
            else:
                # Add to current chunk
                current_chunk.append(para)
                current_size += para_size
        
        # Add any remaining paragraphs
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
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
        Now enforces strict chunk size limits.
        """
        chunks = []

        # Clean text and convert to HTML
        text = self._clean_navigation_elements(text)
        if self.ignore_link_urls:
            text = self._clean_markdown_links(text)
        html = self._markdown_to_html(text)

        # Extract code blocks with context first
        code_blocks = self._extract_code_blocks(html)

        # Create code-text pairs
        for code_block, context in code_blocks:
            # Skip empty code blocks
            if not code_block.strip():
                continue

            # Truncate context text to fit token limit and chunk size
            truncated_context = self._truncate_to_token_limit(context)
            if len(truncated_context) > self.chunk_size:
                # Split context into smaller pieces if needed
                sentences = re.split(r'(?<=[.!?])\s+', truncated_context)
                current_chunk = ""
                
                for sentence in sentences:
                    # If adding this sentence would exceed chunk size
                    if len(current_chunk) + len(sentence) + 1 > self.chunk_size:
                        # Create chunk with this context portion and the code
                        code_text_pair = {
                            "text_for_embedding": current_chunk,
                            "code_block": code_block,
                            "combined": f"{current_chunk}\n\n{code_block}",
                            "metadata": {
                                "type": "code_text_pair",
                                "estimated_tokens": self._estimate_token_count(current_chunk),
                            },
                        }
                        chunks.append(code_text_pair)
                        current_chunk = sentence
                    else:
                        # Add to current chunk
                        if current_chunk:
                            current_chunk += " "
                        current_chunk += sentence
                
                # Add the last chunk
                if current_chunk:
                    code_text_pair = {
                        "text_for_embedding": current_chunk,
                        "code_block": code_block,
                        "combined": f"{current_chunk}\n\n{code_block}",
                        "metadata": {
                            "type": "code_text_pair",
                            "estimated_tokens": self._estimate_token_count(current_chunk),
                        },
                    }
                    chunks.append(code_text_pair)
            else:
                # Context fits within chunk size
                code_text_pair = {
                    "text_for_embedding": truncated_context,
                    "code_block": code_block,
                    "combined": f"{truncated_context}\n\n{code_block}",
                    "metadata": {
                        "type": "code_text_pair",
                        "estimated_tokens": self._estimate_token_count(truncated_context),
                    },
                }
                chunks.append(code_text_pair)

        # Extract text chunks (excluding code blocks)
        text_sections = self._extract_sections(html)

        for section in text_sections:
            content = section["content"]
            if not content.strip():
                continue

            # Extract text from HTML content if needed
            if isinstance(content, str) and ("<" in content or ">" in content):
                soup = BeautifulSoup(content, "html.parser")
                content = soup.get_text()

            # Check if content exceeds chunk size
            if len(content) > self.chunk_size:
                # Split into sentences
                sentences = re.split(r'(?<=[.!?])\s+', content)
                current_chunk = ""
                
                for sentence in sentences:
                    # If this sentence alone is too large
                    if len(sentence) > self.chunk_size:
                        # Add current chunk first
                        if current_chunk:
                            truncated_chunk = self._truncate_to_token_limit(current_chunk)
                            if len(truncated_chunk) >= self.min_chunk_size:
                                chunk = {
                                    "text_for_embedding": truncated_chunk,
                                    "combined": truncated_chunk,
                                    "metadata": {
                                        "type": "text_chunk",
                                        "estimated_tokens": self._estimate_token_count(truncated_chunk),
                                        "title": section.get("title", ""),
                                        "level": section.get("level", 0),
                                        "path": section.get("path", []),
                                    },
                                }
                                chunks.append(chunk)
                        
                        # Split sentence into smaller chunks
                        for i in range(0, len(sentence), self.chunk_size):
                            sentence_chunk = sentence[i:i+self.chunk_size]
                            if len(sentence_chunk) >= self.min_chunk_size:
                                chunk = {
                                    "text_for_embedding": sentence_chunk,
                                    "combined": sentence_chunk,
                                    "metadata": {
                                        "type": "text_chunk",
                                        "estimated_tokens": self._estimate_token_count(sentence_chunk),
                                        "title": section.get("title", ""),
                                        "level": section.get("level", 0),
                                        "path": section.get("path", []),
                                    },
                                }
                                chunks.append(chunk)
                        
                        current_chunk = ""
                    # If adding this sentence would exceed chunk size
                    elif len(current_chunk) + len(sentence) + 1 > self.chunk_size:
                        # Create chunk with current content
                        truncated_chunk = self._truncate_to_token_limit(current_chunk)
                        if len(truncated_chunk) >= self.min_chunk_size:
                            chunk = {
                                "text_for_embedding": truncated_chunk,
                                "combined": truncated_chunk,
                                "metadata": {
                                    "type": "text_chunk",
                                    "estimated_tokens": self._estimate_token_count(truncated_chunk),
                                    "title": section.get("title", ""),
                                    "level": section.get("level", 0),
                                    "path": section.get("path", []),
                                },
                            }
                            chunks.append(chunk)
                        
                        current_chunk = sentence
                    else:
                        # Add to current chunk
                        if current_chunk:
                            current_chunk += " "
                        current_chunk += sentence
                
                # Add the last chunk
                if current_chunk:
                    truncated_chunk = self._truncate_to_token_limit(current_chunk)
                    if len(truncated_chunk) >= self.min_chunk_size:
                        chunk = {
                            "text_for_embedding": truncated_chunk,
                            "combined": truncated_chunk,
                            "metadata": {
                                "type": "text_chunk",
                                "estimated_tokens": self._estimate_token_count(truncated_chunk),
                                "title": section.get("title", ""),
                                "level": section.get("level", 0),
                                "path": section.get("path", []),
                            },
                        }
                        chunks.append(chunk)
            else:
                # Truncate to token limit
                truncated_section = self._truncate_to_token_limit(content)

                # Skip sections that are too small
                if len(truncated_section) < self.min_chunk_size:
                    continue

                # Create chunk with metadata
                chunk = {
                    "text_for_embedding": truncated_section,
                    "combined": truncated_section,
                    "metadata": {
                        "type": "text_chunk",
                        "estimated_tokens": self._estimate_token_count(truncated_section),
                        "title": section.get("title", ""),
                        "level": section.get("level", 0),
                        "path": section.get("path", []),
                    },
                }

                chunks.append(chunk)

        return chunks

    def _simple_chunk(self, text: str) -> Generator[Dict[str, Any], None, None]:
        """
        Simple chunking method as a fallback.
        Creates smaller, more granular chunks for better multilingual support.
        Now enforces strict chunk size limits.
        """
        if not text:
            return

        # Clean links if needed
        if self.ignore_link_urls:
            text = self._clean_markdown_links(text)

        # Find all potential named entities to preserve
        named_entities = set(self.name_pattern.findall(text))
        
        # Split text into paragraphs
        paragraphs = [p.strip() for p in self.paragraph_pattern.split(text) if p.strip()]
        current_chunk = ""
        current_tokens = 0
        current_header = ""
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

        for paragraph in paragraphs:
            # Check if this is a header
            header_match = self.header_pattern.match(paragraph)
            if header_match:
                # If we have content in the current chunk, yield it
                if current_chunk and current_chunk.strip():
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

            # Check if paragraph contains named entities
            contains_named_entity = any(entity in paragraph for entity in named_entities)
            
            # Split long paragraphs into smaller chunks
            if len(paragraph) > self.chunk_size:
                # If we have content in the current chunk, yield it first
                if current_chunk and current_chunk.strip():
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

                # Split the long paragraph into sentences
                sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                current_chunk = ""
                current_tokens = 0
                
                # If paragraph has a single very long sentence
                if len(sentences) <= 1:
                    # Split by characters
                    for i in range(0, len(paragraph), self.chunk_size):
                        char_chunk = paragraph[i:i+self.chunk_size]
                        chunk_count += 1
                        if chunk_count <= self.max_chunks:
                            yield {
                                "chunk": char_chunk,
                                "metadata": {
                                    "type": "simple_text",
                                    "estimated_tokens": self._estimate_token_count(char_chunk),
                                    "header": current_header,
                                },
                                "embedding_text": char_chunk,
                            }
                            logger.info(
                                f"Created chunk {chunk_count}: {len(char_chunk)} chars (simple chunking - character split)"
                            )
                    continue
                
                for sentence in sentences:
                    sentence_tokens = self._estimate_token_count(sentence)
                    sentence_contains_entity = any(entity in sentence for entity in named_entities)
                    
                    # If this sentence is too large by itself
                    if len(sentence) > self.chunk_size:
                        # Add current chunk first if it exists
                        if current_chunk and current_chunk.strip():
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
                        
                        # Split the large sentence into character chunks
                        for i in range(0, len(sentence), self.chunk_size):
                            char_chunk = sentence[i:i+self.chunk_size]
                            chunk_count += 1
                            if chunk_count <= self.max_chunks:
                                yield {
                                    "chunk": char_chunk,
                                    "metadata": {
                                        "type": "simple_text",
                                        "estimated_tokens": self._estimate_token_count(char_chunk),
                                        "header": current_header,
                                    },
                                    "embedding_text": char_chunk,
                                }
                                logger.info(
                                    f"Created chunk {chunk_count}: {len(char_chunk)} chars (simple chunking - character split)"
                                )
                        
                        current_chunk = ""
                        current_tokens = 0
                        continue
                    
                    # If this sentence contains a named entity, make sure it gets its own chunk
                    if sentence_contains_entity and current_chunk:
                        # Yield current chunk before starting entity chunk
                        if current_chunk.strip():
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
                        # Start new chunk with the entity-containing sentence
                        current_chunk = (current_header + "\n\n" if current_header else "") + sentence
                        current_tokens = self._estimate_token_count(current_chunk)
                        continue
                    
                    # If adding this sentence would exceed the chunk size
                    if current_tokens + sentence_tokens > self.chunk_size:
                        # Yield current chunk if it exists
                        if current_chunk and current_chunk.strip():
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
                        
                        # Start new chunk with header and current sentence
                        current_chunk = (current_header + "\n\n" if current_header else "") + sentence
                        current_tokens = self._estimate_token_count(current_chunk)
                    else:
                        # Add sentence to current chunk
                        if current_chunk:
                            current_chunk += ". "
                        current_chunk += sentence
                        current_tokens += sentence_tokens
            else:
                para_tokens = self._estimate_token_count(paragraph)
                
                # If adding this paragraph would exceed the chunk size
                if current_tokens + para_tokens > self.chunk_size:
                    # Yield current chunk if it exists
                    if current_chunk and current_chunk.strip():
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

                    # Start new chunk with header and current paragraph
                    current_chunk = (current_header + "\n\n" if current_header else "") + paragraph
                    current_tokens = self._estimate_token_count(current_chunk)
                else:
                    # Add paragraph to current chunk
                    if current_chunk:
                        current_chunk += "\n\n"
                    current_chunk += paragraph
                    current_tokens += para_tokens

        # Don't forget the last chunk
        if current_chunk and current_chunk.strip():
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

    def chunk_document(self, text: str) -> Generator[Dict[str, Any], None, None]:
        """
        Chunk document into smaller, more focused chunks for better retrieval.
        Now enforces strict size limits.
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
                    # Verify chunk sizes once more
                    combined_text = chunk["combined"]
                    
                    # If combined text is too large, skip this chunk
                    if len(combined_text) > self.chunk_size:
                        logger.warning(f"Skipping oversized chunk: {len(combined_text)} chars")
                        continue
                        
                    chunk_count += 1
                    if chunk_count <= self.max_chunks:
                        chunk_dict = {
                            "chunk": combined_text,
                            "metadata": chunk["metadata"],
                            "embedding_text": chunk["text_for_embedding"],
                        }
                        logger.info(
                            f"Created chunk {chunk_count}: {len(combined_text)} chars (type: {chunk['metadata']['type']})"
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
        Only processes markdown (.md) files.
        """
        if not file_path.lower().endswith('.md'):
            raise ValueError("Only markdown (.md) files are supported")

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