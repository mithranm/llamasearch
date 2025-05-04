import sys
from pathlib import Path
import pytest
import importlib
from llamasearch.trustworthiness.linkChecker import (
    extract_domain,
    resolve_md_file_path,
    validate_file_exists,
    check_links_domain,
    check_links_end,
    create_ratio
)
from llamasearch.trustworthiness import linkChecker

# Add the parent directory of llamasearch to sys.path
#  sys.path.append(str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def sample_url1():
    return "https://www.nasa.gov"

@pytest.fixture
def sample_domain1():
    return "nasa"

@pytest.fixture
def sample_url2():
    return "https://audio-video.gnu.org/video/2015-03-16--rms--free-software-and-your-freedom.webm"

@pytest.fixture
def sample_domain2():
    return "gnu"

@pytest.fixture
def sample_url3():
    return "https://www.nasa.gov"

@pytest.fixture
def sample_domain3():
    return "nasa"

@pytest.fixture
def trusted_module(tmp_path):
    # Create a dummy trustedSources.py file
    trusted_file = tmp_path / "trustedSources.py"
    trusted_file.write_text("trustedSources = ['nasa', 'gnu']")
    sys.path.insert(0, str(tmp_path))  # Add tmp path to module search path
    yield trusted_file
    sys.path.pop(0)  # Clean up path after test


@pytest.fixture
def md_file(tmp_path):
    # Create a markdown file with test links
    md = tmp_path / "test_links.md"
    md.write_text("https://www.nasa.gov\nhttps://audio-video.gnu.org/video/abc\nhttps://untrusted.com\n")
    return md

def test_extractDomainBasic(sample_url1, sample_domain1):
    assert extract_domain(sample_url1) == sample_domain1

def test_extractDomainAdvanced(sample_url2, sample_domain2):
    assert extract_domain(sample_url2) == sample_domain2

def test_extractDomainInvalid():
    with pytest.raises(ValueError):
        extract_domain("not_a_url")

def test_extractDomainInvalidDomain():
    with pytest.raises(ValueError):
        extract_domain("https://.gov")

def test_extractDomainInvalidSuffix():
    with pytest.raises(ValueError):
        extract_domain("https://www.nasa")

def test_extractDomainEmpty():
    with pytest.raises(ValueError):
        extract_domain("")

def test_extractDomainNumber():
    with pytest.raises(ValueError):
        extract_domain(123)

def test_resolve_md_file_path_absolute():
    abs_path = Path(__file__).resolve()
    result = resolve_md_file_path(abs_path)
    assert result == abs_path

def test_resolve_md_file_path_relative():
    # Provide a relative path and check if it returns the expected default path
    result = resolve_md_file_path("links.md")
    
    # Expected default path constructed from project structure
    expected_path = Path(__file__).resolve().parents[1] / "temp" / "links.md"
    assert result == expected_path

def test_validate_file_exists_valid(tmp_path):
    temp_file = tmp_path / "test.md"
    temp_file.write_text("Test content")
    assert validate_file_exists(temp_file) is True

def test_validate_file_exists_nonexistent(tmp_path):
    non_existent_file = tmp_path / "does_not_exist.md"
    assert validate_file_exists(non_existent_file) is False

class DummyDatabase:
    trustedSources = ['nasa', 'gnu']

def test_check_links_domain(tmp_path):
    test_file = tmp_path / "test_links.md"
    test_file.write_text("https://www.nasa.gov\nhttps://audio-video.gnu.org/video/abc\nhttps://untrusted.com")
    
    count, total = check_links_domain(test_file, DummyDatabase())
    assert count == 2
    assert total == 3

def test_check_links_end(tmp_path):
    test_file = tmp_path / "test_links.md"
    test_file.write_text("https://example.edu\nhttps://agency.gov\nhttps://university.int\nhttps://notrust.com\n")
    
    count, total = check_links_end(test_file)
    assert count == 3
    assert total == 4

def test_create_ratio_normal():
    assert create_ratio(2, 4) == 50.0

def test_create_ratio_zero_total():
    assert create_ratio(0, 0) == 0

def test_main_success(monkeypatch, capsys, md_file, trusted_module):
    # Simulate command-line arguments
    monkeypatch.setattr(sys, "argv", ["linkChecker.py", str(md_file), trusted_module.name])

    # Import fresh module so dynamic import works
    importlib.invalidate_caches()
    if trusted_module.stem in sys.modules:
        del sys.modules[trusted_module.stem]

    linkChecker.main()
    captured = capsys.readouterr()

    assert "Percentage of links from trusted domains:" in captured.out
    assert "Percentage of links with trusted TLDs" in captured.out


def test_main_missing_args(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["linkChecker.py"])
    linkChecker.main()
    captured = capsys.readouterr()
    assert "Usage: python linkChecker.py" in captured.out


def test_main_invalid_module(monkeypatch, capsys, md_file):
    monkeypatch.setattr(sys, "argv", ["linkChecker.py", str(md_file), "nonexistentModule"])
    linkChecker.main()
    captured = capsys.readouterr()
    assert "Error: Could not find module 'nonexistentModule'" in captured.out


def test_main_file_does_not_exist(tmp_path, monkeypatch, capsys):
    # Simulate command-line arguments with a non-existent .md file
    fake_md_path = tmp_path / "nonexistent.md"
    test_args = ["linkChecker.py", str(fake_md_path), "trustedSources.py"]
    monkeypatch.setattr(sys, "argv", test_args)

    # Run the main function
    linkChecker.main()

    # Capture printed output and verify correct error message
    captured = capsys.readouterr()
    assert f"Error: Markdown file '{fake_md_path}' does not exist or is not a file." in captured.out

"""def main():
    if test_resolve_md_file_path_relative() is True:
        return "Result True"
    else:
        return "Wrong"

if __name__ == "__main__":
    main()
"""
