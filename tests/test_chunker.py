# tests/test_chunker.py
import unittest
from unittest.mock import patch

from llamasearch.core.chunker import (calculate_effective_length,
                                      chunk_document, chunk_markdown_text)


class TestChunkerUtils(unittest.TestCase):

    def test_calculate_effective_length(self):
        self.assertEqual(calculate_effective_length("Hello world"), len("Hello world"))
        self.assertEqual(
            calculate_effective_length("[link](http://example.com) text"),
            len("link text"),
        )
        self.assertEqual(
            calculate_effective_length("No links here."), len("No links here.")
        )
        self.assertEqual(
            calculate_effective_length(
                "Multiple [links](...) and [another](...) here."
            ),
            len("Multiple links and another here."),
        )
        self.assertEqual(calculate_effective_length(""), 0)


class TestChunkMarkdownText(unittest.TestCase):

    @patch("llamasearch.core.chunker.logger")
    def test_empty_input(self, mock_logger):
        chunks = chunk_markdown_text("", source="empty_source.md")
        self.assertEqual(chunks, [])
        mock_logger.warning.assert_called_with(
            "Received empty input text from source: empty_source.md"
        )

    def test_simple_markdown_chunking(self):
        md_text = "## Header\n\nThis is a paragraph.\n\n- Item 1\n- Item 2\n\nAnother paragraph."
        # Lower min_chunk_char_length for this test's short content
        chunks = chunk_markdown_text(
            md_text,
            source="test.md",
            chunk_size=30,
            chunk_overlap=5,
            min_chunk_char_length=10,
        )
        self.assertTrue(len(chunks) > 1)
        for chunk_dict in chunks:
            self.assertIn("chunk", chunk_dict)
            self.assertIn("metadata", chunk_dict)
            self.assertGreaterEqual(chunk_dict["metadata"]["effective_length"], 10)
            self.assertEqual(chunk_dict["metadata"]["processing_mode"], "markdown/text")

    def test_html_chunking_basic_main_content(self):
        html_text = """
        <html><head><title>Test</title><style>body{color:red}</style><script>alert('hi')</script></head>
        <body>
          <nav><a>Home</a></nav>
          <header><h1>Site Title</h1></header>
          <main>
            <article>
              <h2>Article Title</h2>
              <p>This is the first paragraph of main content. It should be extracted.</p>
              <p>Second paragraph here, also important for context.</p>
            </article>
          </main>
          <footer><p>© 2024</p></footer>
        </body></html>
        """
        chunks = chunk_markdown_text(
            html_text,
            source="test.html",
            chunk_size=50,
            chunk_overlap=10,
            min_chunk_char_length=10,
        )  # Lowered min_chunk_char_length
        self.assertTrue(
            len(chunks) > 0,
            f"Expected chunks, got {len(chunks)}. Full text: {''.join(c['chunk'] for c in chunks)}",
        )
        full_extracted_text = "\n".join([c["chunk"] for c in chunks])

        self.assertIn("Article Title", full_extracted_text)
        self.assertIn("first paragraph of main content", full_extracted_text)
        self.assertIn("Second paragraph here", full_extracted_text)
        self.assertNotIn("Site Title", full_extracted_text)  # From header
        self.assertNotIn("Home", full_extracted_text)  # From nav
        self.assertNotIn("alert('hi')", full_extracted_text)  # From script
        self.assertNotIn("body{color:red}", full_extracted_text)  # From style
        self.assertNotIn("© 2024", full_extracted_text)  # From footer

        for chunk_dict in chunks:
            self.assertEqual(chunk_dict["metadata"]["processing_mode"], "html")

    def test_html_chunking_no_main_tag_uses_body(self):
        html_text = (
            "<body><p>Content directly in body.</p><div>More content.</div></body>"
        )
        # With min_chunk_char_length=10, "body." (len 5) gets filtered out.
        chunks = chunk_markdown_text(
            html_text,
            source="test.htm",
            chunk_size=20,
            chunk_overlap=5,
            min_chunk_char_length=10,
        )
        self.assertTrue(len(chunks) > 0)

        full_extracted_text = "\n".join([c["chunk"] for c in chunks])

        # "body." should be filtered out as its effective length (5) < min_chunk_char_length (10)
        # The original "Content directly in body." is split.
        # "Content directly in" (len 20) becomes one chunk (or part of one).
        # "body." (len 5) is processed, found too short, and dropped.
        # "More content." (len 13) becomes another chunk.

        self.assertIn(
            "Content directly in", full_extracted_text
        )  # This part should be present
        self.assertNotIn(
            "body.", full_extracted_text
        )  # This part should be filtered out
        self.assertIn(
            "More content.", full_extracted_text
        )  # This part should be present

    @patch(
        "llamasearch.core.chunker.BeautifulSoup",
        side_effect=Exception("BS Parsing Fail"),
    )
    @patch("llamasearch.core.chunker.logger")
    def test_html_chunking_beautifulsoup_error_fallback(
        self, mock_logger, mock_bs_constructor
    ):
        html_text = "<html><body><p>Fallback test, this part should pass the length filter.</p></body></html>"
        chunks = chunk_markdown_text(
            html_text,
            source="test_fallback.html",
            chunk_size=100,
            chunk_overlap=10,
            min_chunk_char_length=10,
        )  # Lowered

        mock_logger.error.assert_any_call(
            "BeautifulSoup parsing/extraction failed for HTML source test_fallback.html: BS Parsing Fail. Falling back to raw text.",
            exc_info=True,
        )
        self.assertTrue(len(chunks) > 0, "Chunks should exist in fallback mode")
        self.assertTrue(any("<html>" in c["chunk"] for c in chunks))
        self.assertEqual(chunks[0]["metadata"]["processing_mode"], "html_fallback_raw")

    def test_txt_file_processing(self):
        txt_text = "Simple text file content.\nWith multiple lines for testing."
        chunks = chunk_markdown_text(
            txt_text,
            source="test.txt",
            chunk_size=20,
            chunk_overlap=5,
            min_chunk_char_length=10,
        )  # Lowered
        self.assertTrue(len(chunks) > 0)
        self.assertEqual(chunks[0]["metadata"]["processing_mode"], "markdown/text")

    def test_unknown_extension_fallback(self):
        unknown_text = "Content of an unknown file type, long enough to pass."
        chunks = chunk_markdown_text(
            unknown_text,
            source="test.xyz",
            chunk_size=20,
            chunk_overlap=5,
            min_chunk_char_length=10,
        )  # Lowered
        self.assertTrue(len(chunks) > 0)
        self.assertEqual(
            chunks[0]["metadata"]["processing_mode"], "unknown_fallback_raw"
        )

    def test_chunk_overlap_greater_than_size(self):
        text = "This is a test text for overlap adjustment and it needs to be long enough to produce a chunk."
        chunks = chunk_markdown_text(
            text,
            source="overlap_test.md",
            chunk_size=20,
            chunk_overlap=30,
            min_chunk_char_length=10,
        )  # Lowered
        self.assertTrue(len(chunks) > 0)

    def test_min_chunk_char_length_filter(self):
        text = "Short.\nVery short indeed.\nA bit longer here for a valid chunk that passes the filter."
        chunks = chunk_markdown_text(
            text,
            source="minlen.txt",
            chunk_size=100,
            chunk_overlap=0,
            min_chunk_char_length=30,
        )  # Adjusted min_chunk_char_length
        self.assertEqual(len(chunks), 1)
        # The splitter might include leading/trailing newlines or parts of separators.
        # Focus on the content that should be there.
        self.assertIn(
            "A bit longer here for a valid chunk that passes the filter",
            chunks[0]["chunk"],
        )

    def test_link_stripping(self):
        text_with_link = "Check this [awesome link](http://example.com/path) for more info, and ensure this text is long enough."
        chunks = chunk_markdown_text(
            text_with_link,
            source="link_test.md",
            chunk_size=100,
            chunk_overlap=0,
            min_chunk_char_length=10,
        )  # Lowered
        self.assertEqual(len(chunks), 1)
        expected_text = "Check this awesome link for more info, and ensure this text is long enough."
        self.assertEqual(chunks[0]["chunk"], expected_text)
        self.assertEqual(chunks[0]["metadata"]["length"], len(expected_text))
        self.assertEqual(chunks[0]["metadata"]["effective_length"], len(expected_text))

    def test_no_valid_chunks_after_processing(self):
        html_text = (
            "<html><body><nav>[Home](home.html)</nav><p>[A](b.html)</p></body></html>"
        )
        # If min_chunk_char_length is high enough, the stripped content ("Home A") might be too short.
        chunks = chunk_markdown_text(
            html_text, source="empty_after.html", min_chunk_char_length=10
        )  # Default is 30
        self.assertEqual(chunks, [])

    def test_repetitive_line_filtering(self):
        text_repetitive = "---\n---\n---\nActual content here that should be kept and is long enough.\n---\n---"
        chunks = chunk_markdown_text(
            text_repetitive,
            source="repetitive.md",
            chunk_size=100,
            chunk_overlap=0,
            min_chunk_char_length=10,
        )
        self.assertEqual(len(chunks), 1)
        # The splitter might include leading/trailing newlines or parts of separators.
        # The core is that the "Actual content" is present.
        self.assertIn(
            "Actual content here that should be kept and is long enough",
            chunks[0]["chunk"],
        )

        text_all_repetitive = "=====\n=====\n=====\n====="
        chunks_all_rep = chunk_markdown_text(
            text_all_repetitive,
            source="all_rep.md",
            chunk_size=100,
            chunk_overlap=0,
            min_chunk_char_length=3,
        )
        self.assertEqual(len(chunks_all_rep), 0)

    def test_chunk_document_generator(self):
        md_text = "Paragraph one. This needs to be long enough.\n\nParagraph two, a bit longer to ensure it passes filters."
        expected_chunks_list = chunk_markdown_text(
            md_text,
            source="gen_test.md",
            chunk_size=30,
            chunk_overlap=2,
            min_chunk_char_length=10,
        )  # Lowered

        generated_chunks = list(
            chunk_document(
                md_text,
                source="gen_test.md",
                chunk_size=30,
                chunk_overlap=2,
                min_chunk_char_length=10,
            )
        )  # Lowered

        self.assertEqual(len(generated_chunks), len(expected_chunks_list))
        for gen_chunk, list_chunk in zip(generated_chunks, expected_chunks_list):
            self.assertEqual(gen_chunk, list_chunk)


if __name__ == "__main__":
    unittest.main()
