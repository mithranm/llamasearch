import pytest
import os
import json
from src.llamasearch.trustworthiness.authorSearch import find_author_trustworthiness_score, create_ratio

@pytest.fixture
def mock_json_file(tmp_path):
    """
    Create a temporary JSON file for testing.
    """
    json_file = tmp_path / "domain_authors.json"
    return json_file

def test_valid_json(mock_json_file, monkeypatch):
    """
    Test with a valid JSON file containing authors.
    """
    # Write valid JSON data to the mock file
    mock_json_file.write_text(json.dumps({
        "example.com": {"author": "Author1", "sourceUrl": "https://example.com"},
        "trustedsource.org": {"author": "Author2", "sourceUrl": "https://trustedsource.org"},
        "untrusted.com": {"author": None, "sourceUrl": "https://untrusted.com"}
    }))

    # Mock the path to the JSON file
    monkeypatch.setattr(
        "src.llamasearch.trustworthiness.authorSearch.os.path.join",
        lambda *args: str(mock_json_file)
    )

    # Call the function and assert the result
    score = find_author_trustworthiness_score()
    assert score == pytest.approx(66.67, rel=1e-2)  # 2 valid authors out of 3 total

def test_missing_json_file(monkeypatch):
    """
    Test when the JSON file is missing.
    """
    # Mock the path to a non-existent JSON file
    monkeypatch.setattr(
        "src.llamasearch.trustworthiness.authorSearch.os.path.join",
        lambda *args: "non_existent_file.json"
    )

    # Call the function and assert the result
    score = find_author_trustworthiness_score()
    assert score == 0.0  # Default score when the file is missing

def test_empty_json(mock_json_file, monkeypatch):
    """
    Test with an empty JSON file.
    """
    # Write empty JSON data to the mock file
    mock_json_file.write_text(json.dumps({}))

    # Mock the path to the JSON file
    monkeypatch.setattr(
        "src.llamasearch.trustworthiness.authorSearch.os.path.join",
        lambda *args: str(mock_json_file)
    )

    # Call the function and assert the result
    score = find_author_trustworthiness_score()
    assert score == 0.0  # No keys in the JSON file

def test_invalid_json(mock_json_file, monkeypatch):
    """
    Test with an invalid JSON file.
    """
    # Write invalid JSON data to the mock file
    mock_json_file.write_text("INVALID_JSON")

    # Mock the path to the JSON file
    monkeypatch.setattr(
        "src.llamasearch.trustworthiness.authorSearch.os.path.join",
        lambda *args: str(mock_json_file)
    )

    # Call the function and assert that it raises a JSONDecodeError
    with pytest.raises(json.JSONDecodeError):
        find_author_trustworthiness_score()

def test_create_ratio():
    """
    Test the create_ratio function.
    """
    assert create_ratio(50, 100) == 50.0  # 50%
    assert create_ratio(0, 100) == 0.0    # 0%
    assert create_ratio(100, 100) == 100.0  # 100%
    assert create_ratio(0, 0) == 0.0      # Avoid division by zero