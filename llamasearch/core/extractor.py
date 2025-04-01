import requests
import os
import re

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
    """Saves extracted content to a `temp` directory inside the project root, using the URL to create a unique filename."""
    project_root = find_project_root()
    temp_dir = os.path.join(project_root, "temp")
    os.makedirs(temp_dir, exist_ok=True)  # Ensure the directory exists

    filename = slugify_url(url) + ".md"  # Create filename from URL
    file_path = os.path.join(temp_dir, filename)

    # Write content to the file
    with open(
        file_path, "w", encoding="utf-8"
    ) as file:  # Use "w" to create/overwrite each file
        file.write(f"## {url}\n\n{text}")  # Include URL as a heading in the file

    return file_path


def read_links_from_temp():
    """Reads URLs from the saved links file."""
    project_root = find_project_root()
    temp_file = os.path.join(project_root, "temp", "links.txt")

    if not os.path.exists(temp_file):
        print("No links found. Run crawler.py first.")
        return []

    with open(temp_file, "r", encoding="utf-8") as f:
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
    links = read_links_from_temp()

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
