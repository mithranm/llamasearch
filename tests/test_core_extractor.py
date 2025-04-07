import pytest
import os
import tempfile
from unittest.mock import patch, mock_open
from llamasearch.core.extractor import (
    extract_text_with_jina,
    save_to_project_tempdir,
    read_links_from_temp,
)

JINA_API_URL = "https://r.jina.ai/"
DUMMY_URL = "https://example.com"

MOCK_JINA_TEXT = "This is extracted text from the webpage."


@pytest.fixture
def mock_requests_get():
    """Mocks requests.get to return extracted text."""
    with patch("requests.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.text = MOCK_JINA_TEXT
        yield mock_get


def test_extract_text_with_jina(mock_requests_get):
    """Test text extraction using Jina API."""
    text = extract_text_with_jina(DUMMY_URL)

    assert mock_requests_get.called
    assert text == MOCK_JINA_TEXT


def test_save_to_project_tempdir():
    """Test saving extracted text to a temp directory."""
    mock_text = "Extracted content"
    mock_url = "https://example.com/test-page"
    mock_root = tempfile.mkdtemp()

    with patch("llamasearch.core.extractor.get_llamasearch_dir", return_value=mock_root):
        with patch("builtins.open", mock_open()) as mock_file:
            file_path = save_to_project_tempdir(mock_text, mock_url)

            assert mock_file.called
            expected_filename = "example-com-test-page.md"
            assert file_path == os.path.join(mock_root, "temp", expected_filename)


def test_read_links_from_temp():
    """Test reading stored links from temp file."""
    mock_links = "https://example.com/page1\nhttps://example.com/page2"
    mock_root = tempfile.mkdtemp()

    with patch("llamasearch.core.extractor.get_llamasearch_dir", return_value=mock_root):
        with patch("os.path.exists", return_value=True):
            with patch("builtins.open", mock_open(read_data=mock_links)):
                links = read_links_from_temp()

                assert len(links) == 2
                assert "https://example.com/page1" in links
                assert "https://example.com/page2" in links
