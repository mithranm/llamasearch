# PYTHONPATH=$(pwd)/llamasearch python3 -m pytest llamasearch/tests/
import pytest
import os
from unittest.mock import patch, mock_open
from llamasearch.core.crawler import (
    fetch_links,
    filter_links_by_structure,
    save_crawled_links,
)

JINA_API_URL = "https://r.jina.ai/"
DUMMY_URL = "https://example.com"

MOCK_JINA_RESPONSE = """
<html>
<body>
    <a href="https://example.com/page1">Page 1</a>
    <a href="https://example.com/page2">Page 2</a>
    <a href="https://otherdomain.com/notallowed">External Link</a>
    <a href="https://example.com/image.jpg">Image</a>
    <a href="https://example.com/video.mp4">Video</a>
    <a href="/relative/path">Relative Link</a>
</body>
</html>
"""


@pytest.fixture
def mock_requests_get():
    """Mocks requests.get to return the mock response."""
    with patch("requests.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.text = MOCK_JINA_RESPONSE
        mock_get.return_value.content = MOCK_JINA_RESPONSE.encode("utf-8")
        yield mock_get


def test_fetch_links(mock_requests_get):
    """Test fetching links using Jina API."""
    links = fetch_links(DUMMY_URL, max_links=5)
    assert mock_requests_get.called
    assert len(links) == 5
    assert "https://example.com/page1" in links
    # External links are allowed at depth 1
    assert "https://otherdomain.com/notallowed" in links


def test_filter_links_by_structure():
    """Test filtering extracted links by the same domain and ignoring media files."""
    links = [
        "https://example.com/page1",
        "https://example.com/page2",
        "https://otherdomain.com/notallowed",
        "https://example.com/image.jpg",
        "https://example.com/video.mp4",
    ]

    filtered = filter_links_by_structure(DUMMY_URL, links)

    assert "https://example.com/page1" in filtered
    assert "https://example.com/page2" in filtered
    assert "https://otherdomain.com/notallowed" not in filtered
    assert "https://example.com/image.jpg" not in filtered
    assert "https://example.com/video.mp4" not in filtered


def test_filter_links_subdomains():
    """Test that links from subdomains of the same base domain are kept."""
    # Test with base domain
    original_url = "https://example.com"
    links = [
        "https://example.com/page1",
        "https://www.example.com/page2",
        "https://sub.example.com/page3",
        "https://other-domain.com/page",
    ]

    filtered = filter_links_by_structure(original_url, links)

    assert len(filtered) == 3
    assert "https://example.com/page1" in filtered
    assert "https://www.example.com/page2" in filtered
    assert "https://sub.example.com/page3" in filtered
    assert "https://other-domain.com/page" not in filtered

    # Test with www subdomain
    original_url = "https://www.example.com"

    filtered = filter_links_by_structure(original_url, links)

    assert len(filtered) == 3
    assert "https://example.com/page1" in filtered
    assert "https://www.example.com/page2" in filtered
    assert "https://sub.example.com/page3" in filtered
    assert "https://other-domain.com/page" not in filtered


def test_real_world_gnu_org():
    """Test with real-world example of gnu.org and www.gnu.org."""
    # Test with base domain gnu.org
    original_url = "https://gnu.org"
    links = [
        "https://gnu.org/",
        "https://gnu.org/#More-GNU",
        "https://www.gnu.org/distros/free-distros.html",
        "https://www.gnu.org/gnu/gnu-linux-faq.html",
        "https://www.gnu.org/home.html",
    ]

    filtered = filter_links_by_structure(original_url, links)

    assert len(filtered) == 5  # All links should be kept

    # Test with www subdomain
    original_url = "https://www.gnu.org"

    filtered = filter_links_by_structure(original_url, links)

    assert len(filtered) == 5  # All links should be kept


def test_complex_domains():
    """Test with more complex domain structures."""
    original_url = "https://sub.example.co.uk"
    links = [
        "https://example.co.uk/page1",
        "https://www.example.co.uk/page2",
        "https://sub.example.co.uk/page3",
        "https://other.co.uk/page",
    ]

    filtered = filter_links_by_structure(original_url, links)

    assert len(filtered) == 3
    assert "https://example.co.uk/page1" in filtered
    assert "https://www.example.co.uk/page2" in filtered
    assert "https://sub.example.co.uk/page3" in filtered
    assert "https://other.co.uk/page" not in filtered


def test_save_crawled_links():
    """Test saving crawled links to a temp directory."""
    mock_text = "https://example.com/page1\nhttps://example.com/page2"
    import tempfile

    mock_root = tempfile.mkdtemp()

    with patch("llamasearch.core.crawler.find_project_root", return_value=mock_root):
        with patch("builtins.open", mock_open()) as mock_file:
            file_path = save_crawled_links(mock_text, "test_link.txt")

            assert mock_file.called
            assert file_path == os.path.join(mock_root, "data", "test_link.txt")
