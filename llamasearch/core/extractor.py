import requests
import os
import re
import json
from datetime import datetime

from ..setup_utils import find_project_root

# Changing this to my personal Jina API URL
JINA_API_URL = "https://postgres.mithran.org/oodo/"


def slugify_url(url):
    """
    Converts a URL into a valid filename by removing or replacing invalid characters.
    """
    # Remove protocol (http/https) and "www"
    url = re.sub(r"https?://(www\.)?", "", url)
    # Replace any character that is not a letter, number, or underscore with a dash
    url = re.sub(r"[^a-zA-Z0-9_]+", "-", url)
    # Remove leading and trailing dashes
    url = url.strip("-")
    # Truncate to a reasonable length
    url = url[:200]  # Limit filename length
    return url


def save_to_project_tempdir(text, url):
    """
    Saves extracted content to a `temp` directory inside the project root,
    including metadata in an HTML comment that will be skipped by the chunker.
    Uses the URL to create a unique filename.
    """
    project_root = find_project_root()
    temp_dir = os.path.join(project_root, "temp")
    os.makedirs(temp_dir, exist_ok=True)  # Ensure the directory exists

    filename = slugify_url(url) + ".md"  # Create filename from URL
    file_path = os.path.join(temp_dir, filename)

    # Create metadata as JSON in an HTML comment
    metadata = {
        "source_url": url,
        "extracted_at": datetime.now().isoformat(),
        "extractor_version": "1.0"
    }
    
    metadata_str = f"""<!-- 
METADATA: {json.dumps(metadata, indent=2)}
-->

"""

    # Write content to the file with metadata
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(metadata_str + f"## {url}\n\n{text}")  # Include metadata + URL heading + content

    return file_path


def read_links_from_data():
    """Reads URLs from the saved links file in the data directory."""
    project_root = find_project_root()
    data_file = os.path.join(project_root, "data", "links.txt")

    if not os.path.exists(data_file):
        print("No links found. Run crawler.py first to crawl links to the data directory.")
        return []

    with open(data_file, "r", encoding="utf-8") as f:
        links = [line.strip() for line in f.readlines()]

    return links


def extract_text_with_jina(url):
    """Fetches cleaned text using Jina AI."""
    try:
        # Increased timeout to 120 seconds to accommodate longer processing times
        response = requests.get(JINA_API_URL + url, timeout=120)
        response.raise_for_status()
        return response.text.strip()
    except requests.RequestException as e:
        print(f"Error fetching from Jina AI: {e}")
        return None


if __name__ == "__main__":
    links = read_links_from_data()

    if not links:
        print("No links to process.")
        exit()

    # Synchronous - avoids rate limiting
    for link in links:
        print(f"Extracting content from: {link}")
        text = extract_text_with_jina(link)

        if text:
            file_path = save_to_project_tempdir(
                text, link
            )  # Pass the URL to save_to_project_tempdir
            print(f"Extracted content saved at: {file_path}")