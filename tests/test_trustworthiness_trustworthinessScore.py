import pytest
from unittest.mock import patch, MagicMock
from src.llamasearch.trustworthiness.trustworthinessScore import main

@pytest.fixture
def mock_crawler():
    """
    Mock the crawler.smart_crawl function.
    """
    with patch("src.llamasearch.core.crawler.smart_crawl") as mock_crawl:
        yield mock_crawl

@pytest.fixture
def mock_convert_main():
    """
    Mock the convert.py main function.
    """
    with patch("src.llamasearch.trustworthiness.convert.main") as mock_convert:
        yield mock_convert

@pytest.fixture
def mock_link_checker_main():
    """
    Mock the linkChecker.py main function.
    """
    with patch("src.llamasearch.trustworthiness.linkChecker.main") as mock_link_checker:
        yield mock_link_checker

@pytest.fixture
def mock_author_trustworthiness_score():
    """
    Mock the find_author_trustworthiness_score function.
    """
    with patch("src.llamasearch.trustworthiness.authorSearch.find_author_trustworthiness_score") as mock_author_score:
        yield mock_author_score

@pytest.fixture
def mock_calculate_trustworthiness_score():
    """
    Mock the calculate_trustworthiness_score function.
    """
    with patch("src.llamasearch.trustworthiness.score.calculate_trustworthiness_score") as mock_calculate_score:
        yield mock_calculate_score

def test_main_success(
    mock_crawler,
    mock_convert_main,
    mock_link_checker_main,
    mock_author_trustworthiness_score,
    mock_calculate_trustworthiness_score,
    capsys
):
    """
    Test the main function when all components succeed.
    """
    # Mock the crawler to return a list of links
    mock_crawler.return_value = ["https://example.com", "https://trustedsource.org"]

    # Mock the linkChecker main function to return domain and TLD ratios
    mock_link_checker_main.return_value = (75.0, 50.0)

    # Mock the author trustworthiness score
    mock_author_trustworthiness_score.return_value = 80.0

    # Mock the calculate_trustworthiness_score function
    mock_calculate_trustworthiness_score.side_effect = [65.0, 70.0]

    # Call the main function
    stars = main()

    print(f"Returned stars: {stars}")
    # Capture the printed output
    captured = capsys.readouterr()

    # Assert the final star rating
    assert stars == 4

    # Assert the printed output contains the expected results
    assert "Final trustworthiness score: 70.00%" in captured.out
    assert "Star rating: 4 stars" in captured.out

def test_main_crawl_failure(mock_crawler, capsys):
    """
    Test the main function when the crawler fails to collect links.
    """
    # Mock the crawler to return an empty list
    mock_crawler.return_value = []

    # Call the main function
    stars = main()

    # Capture the printed output
    captured = capsys.readouterr()

    # Assert the function exits early with no stars
    assert stars is None
    assert "Crawl failed: No links were collected." in captured.out

def test_main_link_checker_failure(
    mock_crawler,
    mock_convert_main,
    mock_link_checker_main,
    capsys
):
    """
    Test the main function when linkChecker.py fails to return valid ratios.
    """
    # Mock the crawler to return a list of links
    mock_crawler.return_value = ["https://example.com", "https://trustedsource.org"]

    # Mock the linkChecker main function to return None
    mock_link_checker_main.return_value = None

    # Call the main function
    stars = main()

    # Capture the printed output
    captured = capsys.readouterr()

    # Assert the function exits early with no stars
    assert stars is None
    assert "Error: linkChecker.py did not return valid domain and TLD ratios." in captured.out