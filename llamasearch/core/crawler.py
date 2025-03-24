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
    """
    Fetches structured content from Jina AI and extracts only links, with an optional limit.

    Args:
        url (str): The URL to fetch links from
        max_links (int): Maximum number of links to return (default: 50)
                        This enforces the 50-child limit per page requirement

    Returns:
        list: List of links found on the page, limited to max_links
    """
    try:
        response = requests.get(JINA_API_URL + url, timeout=10)
        response.raise_for_status()
        content = response.text

        all_links = list(set(re.findall(r"https?://[^\s)>\"]+", content)))

        # Enforce the 50-child limit per page by slicing the list
        return all_links[:max_links]

    except requests.RequestException as e:
        print(f"Error fetching from Jina AI: {e}")
        return None


# TODO: Make this match any subdomain, not just www and allow a search into one external link as well. Terminate the search if
# we hit 50 children on a single node or two external links. Also terminate if we go down to a depth of 3.
def filter_links_by_structure(original_url, links):
    """Filters links to only include those from the same domain or subdomains, while ignoring media files."""
    parsed_url = urlparse(original_url)
    netloc_parts = parsed_url.netloc.split(".")

    # Handle special TLDs like .co.uk, .com.au, etc.
    if len(netloc_parts) >= 3 and netloc_parts[-2] in [
        "co",
        "com",
        "org",
        "net",
        "ac",
        "gov",
        "edu",
    ]:
        # For domains like example.co.uk, take the last three parts
        base_domain = ".".join(netloc_parts[-3:])
    else:
        # For regular domains like example.com, take the last two parts
        base_domain = ".".join(netloc_parts[-2:])

    filtered_links = []

    for link in links:
        parsed_link = urlparse(link)
        link_parts = parsed_link.netloc.split(".")

        # Apply the same logic to determine the base domain of the link
        if len(link_parts) >= 3 and link_parts[-2] in [
            "co",
            "com",
            "org",
            "net",
            "ac",
            "gov",
            "edu",
        ]:
            link_base_domain = ".".join(link_parts[-3:])
        else:
            link_base_domain = ".".join(link_parts[-2:])

        # Allow links from the same base domain (including subdomains)
        if link_base_domain != base_domain:
            continue

        # Ignore media files and social media links
        if re.search(r"\.(jpg|jpeg|png|gif|mp4|webp|svg)$", link):
            continue

        # Add to filtered list without strict path checking
        filtered_links.append(link)

    return filtered_links


def crawl(
    url, depth=1, max_depth=3, visited=None, all_links=None, external_taken=False
):
    if visited is None:
        visited = set()

    if all_links is None:
        all_links = []

    # Stop if maximum depth is reached
    if depth > max_depth:
        return all_links

    # Avoid reprocessing URLs
    if url in visited:
        return all_links

    print(f"Crawling (depth {depth}/{max_depth}): {url}")
    visited.add(url)

    # Fetch links from the page
    links = fetch_links_with_jina(url)
    if not links:
        return all_links

    # Determine the base domain of the original URL
    parsed_url = urlparse(url)
    netloc_parts = parsed_url.netloc.split(".")
    if len(netloc_parts) >= 3 and netloc_parts[-2] in [
        "co",
        "com",
        "org",
        "net",
        "ac",
        "gov",
        "edu",
    ]:
        base_domain = ".".join(netloc_parts[-3:])
    else:
        base_domain = ".".join(netloc_parts[-2:])

    # Process internal links (same base domain including subdomains)
    internal_links = filter_links_by_structure(url, links)
    for link in internal_links:
        if link not in all_links:
            all_links.append(link)
        if link not in visited and depth < max_depth:
            crawl(link, depth + 1, max_depth, visited, all_links, external_taken)

    # Process external links: Allow one external link if not already processed
    if not external_taken and depth < max_depth:
        for link in links:
            # Check if the link is external by comparing base domains
            parsed_link = urlparse(link)
            link_parts = parsed_link.netloc.split(".")
            if len(link_parts) >= 3 and link_parts[-2] in [
                "co",
                "com",
                "org",
                "net",
                "ac",
                "gov",
                "edu",
            ]:
                link_base_domain = ".".join(link_parts[-3:])
            else:
                link_base_domain = ".".join(link_parts[-2:])

            if link_base_domain != base_domain:
                # Skip media files or other unwanted formats
                if re.search(r"\.(jpg|jpeg|png|gif|mp4|webp|svg)$", link):
                    continue

                # Process this external link and mark external_taken as True
                if link not in visited:
                    print(f"Processing external link: {link}")
                    if link not in all_links:
                        all_links.append(link)
                    crawl(
                        link,
                        depth + 1,
                        max_depth,
                        visited,
                        all_links,
                        external_taken=True,
                    )
                    break  # Only allow one external link

    return all_links


if __name__ == "__main__":
    url = input("Enter the webpage URL: ").strip()

    # Start crawling from depth 1 with a maximum depth of 3
    all_collected_links = crawl(url, depth=1, max_depth=3, external_taken=False)

    if all_collected_links:
        file_path = save_to_project_tempdir("\n".join(all_collected_links), "links.md")
        print(f"\nCrawled links saved at: {file_path}")
        print(f"Total links collected: {len(all_collected_links)}")
    else:
        print("\nNo relevant links found.")
