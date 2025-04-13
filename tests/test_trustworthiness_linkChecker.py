import sys
from pathlib import Path
import pytest
from llamasearch.trustworthiness.linkChecker import (
    extract_domain,
    resolve_md_file_path
)

# Add the parent directory of llamasearch to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))

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
    expected_path = Path(__file__).resolve().parents[2] / "temp" / "links.md"
    assert result == expected_path

"""def main():
    if test_resolve_md_file_path_relative() is True:
        return "Result True"
    else:
        return "Wrong"

if __name__ == "__main__":
    main()
"""
