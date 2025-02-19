#PYTHONPATH=$(pwd)/llamasearch python3 -m pytest llamasearch/tests/
import pytest
import os
import requests
from unittest.mock import patch, mock_open
from core.crawler import fetch_links_with_jina, filter_links_by_structure, save_to_project_tempdir, find_project_root

JINA_API_URL = "https://r.jina.ai/"
DUMMY_URL = "https://example.com"

# Mock response for Jina AI link extraction
MOCK_JINA_RESPONSE = """
https://example.com/page1
https://example.com/page2
https://otherdomain.com/notallowed
https://example.com/image.jpg
https://example.com/video.mp4
"""

@pytest.fixture
def mock_requests_get():
    """Mocks requests.get to return the mock response."""
    with patch("requests.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.text = MOCK_JINA_RESPONSE
        yield mock_get

def test_fetch_links_with_jina(mock_requests_get):
    """Test fetching links using Jina API."""
    links = fetch_links_with_jina(DUMMY_URL, max_links=5)

    assert mock_requests_get.called
    assert len(links) == 5
    assert "https://example.com/page1" in links
    assert "https://otherdomain.com/notallowed" in links
def test_filter_links_by_structure():
    """Test filtering extracted links by the same domain and ignoring media files."""
    links = [
        "https://example.com/page1",
        "https://example.com/page2",
        "https://otherdomain.com/notallowed",
        "https://example.com/image.jpg",
        "https://example.com/video.mp4"
    ]

    filtered = filter_links_by_structure(DUMMY_URL, links)

    assert "https://example.com/page1" in filtered
    assert "https://example.com/page2" in filtered
    assert "https://otherdomain.com/notallowed" not in filtered
    assert "https://example.com/image.jpg" not in filtered
    assert "https://example.com/video.mp4" not in filtered

def test_save_to_project_tempdir():
    """Test saving filtered links to a temp directory."""
    mock_text = "https://example.com/page1\nhttps://example.com/page2"
    import tempfile
    mock_root = tempfile.mkdtemp()


    with patch("core.crawler.find_project_root", return_value=mock_root):
        with patch("builtins.open", mock_open()) as mock_file:
            file_path = save_to_project_tempdir(mock_text, "test_links.txt")

            assert mock_file.called
            assert file_path == os.path.join(mock_root, "temp", "test_links.txt")
