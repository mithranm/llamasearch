import requests
import re
import os
from ..setup_utils import find_project_root
from urllib.parse import urlparse
from bs4 import BeautifulSoup, Tag

JINA_API_URL = "https://r.jina.ai/"


def save_to_project_tempdir(text, filename="links.md"):
    """Saves text to a `temp` directory inside the project root."""
    project_root = find_project_root()
    temp_dir = os.path.join(project_root, "temp")

    os.makedirs(temp_dir, exist_ok=True)  # Create temp dir if it doesn't exist
    file_path = os.path.join(temp_dir, filename)  # Define full file path

    with open(file_path, "w", encoding="utf-8") as file:
        file.write(text)

    return file_path  # Return the file path for reference


def fetch_links(
    url,
    max_links=3,
    user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36",
):
    """
    Fetches HTML content and extracts links using BeautifulSoup.

    Args:
        url (str): The URL to fetch links from
        max_links (int): Maximum number of links to return
                        This enforces the child limit per page requirement
        user_agent (str): User agent string to use for the request

    Returns:
        list: List of links found on the page, limited to max_links
    """
    try:
        headers = {"User-Agent": user_agent}
        response = requests.get(url, timeout=10, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")

        # Find all a tags and extract href attributes
        links = []
        for a_tag in soup.find_all("a", href=True):
            # Check if the element is a Tag
            if isinstance(a_tag, Tag):
                # Get href as string (fixes the AttributeValueList issue)
                if hasattr(a_tag, "get"):  # Check if the element has get method
                    href = str(a_tag.get("href", ""))  # type: ignore
                else:
                    continue

                # Convert relative URLs to absolute
                if href.startswith("http"):
                    links.append(href)
                elif href.startswith("/"):
                    # Handle relative URLs
                    parsed_url = urlparse(url)
                    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
                    links.append(f"{base_url}{href}")

        # Remove duplicates and limit to max_links
        return list(set(links))[:max_links]

    except requests.RequestException as e:
        print(f"Error fetching HTML: {e}")
        return None
    except ImportError:
        print("BeautifulSoup is required. Install it using: pip install beautifulsoup4")
        return None


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

    if depth == 1 and url not in all_links:
        all_links.append(url)

    # Stop if maximum depth is reached
    if depth > max_depth:
        return all_links

    # Avoid reprocessing URLs
    if url in visited:
        return all_links

    print(f"Crawling (depth {depth}/{max_depth}): {url}")
    visited.add(url)

    # Fetch links from the page
    links = fetch_links(url)
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
        file_path = save_to_project_tempdir("\n".join(all_collected_links), "links.txt")
        print(f"\nCrawled links saved at: {file_path}")
        print(f"Total links collected: {len(all_collected_links)}")
    else:
        print("\nNo relevant links found.")
