import requests
import re
import os
from urllib.parse import urlparse

JINA_API_URL = "https://r.jina.ai/"


def find_project_root():
    """Finds the root of the project by looking for `pyproject.toml`."""
    current_dir = os.path.abspath(os.path.dirname(__file__))

    while current_dir != os.path.dirname(current_dir):
        if os.path.exists(os.path.join(current_dir, "pyproject.toml")):
            return current_dir

        current_dir = os.path.dirname(current_dir)

    raise RuntimeError(
        "Could not find project root. Please check your project structure."
    )


def save_to_project_tempdir(text, filename="links.md"):
    """Saves text to a `temp` directory inside the project root."""
    project_root = find_project_root()
    temp_dir = os.path.join(project_root, "temp")

    os.makedirs(temp_dir, exist_ok=True)  # Create temp dir if it doesn't exist
    file_path = os.path.join(temp_dir, filename)  # Define full file path

    with open(file_path, "w", encoding="utf-8") as file:
        file.write(text)

    return file_path  # Return the file path for reference


def fetch_links_with_jina(url, max_links=50):
    """Fetches structured content from Jina AI and extracts only links, with an optional limit."""
    try:
        response = requests.get(JINA_API_URL + url, timeout=10)
        response.raise_for_status()
        content = response.text

        all_links = list(set(re.findall(r"https?://[^\s)>\"]+", content)))

        return all_links[:max_links]

    except requests.RequestException as e:
        print(f"Error fetching from Jina AI: {e}")
        return None


def filter_links_by_structure(original_url, links):
    """Filters links to only include those from the same domain, while ignoring media files."""
    parsed_url = urlparse(original_url)

    filtered_links = []

    for link in links:
        parsed_link = urlparse(link)

        # Allow only links from the same domain
        if parsed_link.netloc != parsed_url.netloc:
            continue

        # Ignore media files and social media links
        if re.search(r"\.(jpg|jpeg|png|gif|mp4|webp|svg)$", link):
            continue

        # Add to filtered list without strict path checking
        filtered_links.append(link)

    return filtered_links


if __name__ == "__main__":
    url = input("Enter the webpage URL: ").strip()

    links = fetch_links_with_jina(url)

    if links:
        filtered_links = filter_links_by_structure(url, links)

        if filtered_links:
            file_path = save_to_project_tempdir("\n".join(filtered_links), "links.md")
            print(f"\nFiltered links saved at: {file_path}")
        else:
            print("\nNo relevant links found.")
