import sys
from pathlib import Path
import pytest
from llamasearch.trustworthiness.linkChecker import (
    extract_domain
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

def test_extractDomainEmpty():
    with pytest.raises(ValueError):
        extract_domain("")
