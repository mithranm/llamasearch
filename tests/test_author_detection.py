#!/usr/bin/env python3

import os
import sys
import logging

# Add the parent directory to the path to ensure imports work correctly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now import modules after path is set up
import pytest
from author_test import discover_site_author

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Test data
AUTHOR_TEST_CASES = [
    ("austinkleon.com", "Austin Kleon"),
    ("www.romineustadt.com", "Romi Neustadt"),
    ("mikitaylor.com", "Miki Taylor"),
    ("markmanson.net", "Mark Manson"),
    ("danielgibbsauthor.com", "Daniel Gibbs"),
    ("www.georgeweigel.com", "George Weigel"),
    ("gomason.com/sports/mens-volleyball", "GoMasonMVB"),
]

@pytest.fixture(scope="module", autouse=True)
def setup_module():
    """Setup for the test module."""
    # Install BeautifulSoup if not already installed
    try:
        import bs4
    except ImportError:
        print("Installing BeautifulSoup4...")
        os.system("pip install beautifulsoup4")
    
    # Install pytest-cov if not already installed
    try:
        import pytest_cov
    except ImportError:
        print("Installing pytest-cov...")
        os.system("pip install pytest-cov")
    
    yield


@pytest.mark.parametrize("domain,expected_author", AUTHOR_TEST_CASES)
def test_author_detection(domain, expected_author):
    """Test that the author detection works correctly for various domains."""
    # Get the author information from the domain
    author_info = discover_site_author(domain)
    
    # Extract the detected author
    detected_author = author_info.get("author", None)
    source_url = author_info.get("sourceUrl", None)
    
    # Log the results
    logger.info(f"Domain: {domain}")
    logger.info(f"Expected author: {expected_author}")
    logger.info(f"Detected author: {detected_author}")
    logger.info(f"Source URL: {source_url}")
    
    # Assert that the author was detected
    assert detected_author is not None, f"No author detected for {domain}"
    
    # Assert that the detected author contains the expected author (case-insensitive)
    assert expected_author.lower() in detected_author.lower(), \
        f"Expected '{expected_author}' but got '{detected_author}' for {domain}"


if __name__ == "__main__":
    # This allows the file to be run directly (without pytest)
    # Use a more complete set of arguments to handle both explicit and config-based args
    args = ["-v", __file__]
    # Add any additional args from command line, skipping the script name
    if len(sys.argv) > 1:
        args.extend(sys.argv[1:])
    
    # Run pytest with the assembled arguments
    pytest.main(args) 