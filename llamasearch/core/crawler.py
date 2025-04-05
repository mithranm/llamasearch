import requests
import re
import os
from urllib.parse import urlparse

# Updated to use the development Scraping API
SCRAPING_API_URL = "https://postgres.mithran.org/html/"


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


def save_to_project_dir(text, filename="links.txt", directory="data"):
    """Saves text to a specified directory inside the project root."""
    project_root = find_project_root()
    target_dir = os.path.join(project_root, directory)

    os.makedirs(target_dir, exist_ok=True)  # Create directory if it doesn't exist
    file_path = os.path.join(target_dir, filename)  # Define full file path

    with open(file_path, "w", encoding="utf-8") as file:
        file.write(text)

    return file_path  # Return the file path for reference


def fetch_links_with_api(url, max_links=4):
    """
    Fetches structured content from scraping API and extracts only links, with an optional limit.

    Args:
        url (str): The URL to fetch links from
        max_links (int): Maximum number of links to return

    Returns:
        list: List of links found on the page, limited to max_links
    """
    try:
        print(f"\nFetching from Scraping API: {SCRAPING_API_URL}{url}")
        response = requests.get(f"{SCRAPING_API_URL}{url}", timeout=45)
        response.raise_for_status()
        content = response.text

        # Save raw API response for debugging
        debug_filename = f"scraping_response_{urlparse(url).netloc.replace('.', '_')}.html"
        debug_path = save_to_project_dir(content, filename=debug_filename, directory="debug")
        print(f"Saved raw API response to: {debug_path}")

        # Extract both absolute and relative links
        absolute_links = set(re.findall(r'href=[\'"]([^\"\']*)[\"\'](\s|>)', content))
        all_links = []
        base_url = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
        
        for link in absolute_links:
            link = link[0].strip()
            if not link:
                continue
            # Handle relative URLs
            if link.startswith('/'):
                link = f"{base_url}{link}"
            elif not link.startswith(('http://', 'https://')):
                link = f"{base_url}/{link}"
            all_links.append(link)
        print(f"Found {len(all_links)} total links in response")

        # Log the first few links for debugging
        for i, link in enumerate(all_links[:3]):
            print(f"Sample link {i + 1}: {link}")

        # Enforce the link limit by slicing the list
        return all_links[:max_links]

    except requests.RequestException as e:
        print(f"\nError fetching from Scraping API: {e}")
        print(f"Full error details: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status code: {e.response.status_code}")
            print(f"Response text: {e.response.text[:500]}...")
        return None


def normalize_url(url):
    """Normalize URL by adding https:// if missing and handling www."""
    if not url:
        return url
        
    # Add scheme if missing
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
        
    # Parse URL
    parsed = urlparse(url)
    
    # Keep the original netloc (with or without www) to avoid loops
    # Just ensure trailing slash consistency
    normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    if not parsed.path:
        normalized += '/'
    if parsed.query:
        normalized += f"?{parsed.query}"
    if parsed.fragment:
        normalized += f"#{parsed.fragment}"
    
    return normalized

def get_base_domain(url):
    """Extract the base domain from a URL."""
    parsed_url = urlparse(url)
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
        return ".".join(netloc_parts[-3:])
    else:
        # For regular domains like example.com, take the last two parts
        return ".".join(netloc_parts[-2:])

def is_valid_content_url(url):
    """
    Check if a URL likely points to actual content rather than utility endpoints.
    """
    parsed = urlparse(url)
    path = parsed.path.lower()
    
    # Skip utility paths
    skip_patterns = [
        r'/cdn-cgi/',           # Cloudflare utilities
        r'/wp-json/',           # WordPress API
        r'/wp-admin/',          # WordPress admin
        r'/wp-content/',        # WordPress resources
        r'/api/',               # Generic API endpoints
        r'#',                   # Anchor links
        r'/feed/',              # RSS/Atom feeds
        r'/xmlrpc\.php',       # XML-RPC endpoints
        r'/wp-includes/',       # WordPress includes
        r'/cdn/',               # Generic CDN paths
        r'/assets/',            # Generic asset paths
        r'/static/',            # Static files
        r'/email-protection',   # Email protection scripts
        r'/ajax/',              # AJAX endpoints
        r'/rss/',               # RSS feeds
        r'/login',              # Login pages
        r'/signup',             # Signup pages
        r'/register',           # Registration pages
        r'/search'              # Search endpoints
    ]
    
    # Check if URL matches any skip pattern
    if any(re.search(pattern, path) for pattern in skip_patterns):
        return False
        
    # Skip URLs with suspicious number of query parameters (likely utility endpoints)
    if len(parsed.query) > 50:  # Arbitrary threshold
        return False
        
    return True

def filter_links_by_structure(original_url, links):
    """
    Filters and categorizes links into internal and external links.
    Also filters out media, resource files, and utility endpoints.
    
    Returns:
        tuple: (internal_links, external_links)
    """
    base_domain = get_base_domain(original_url)
    internal_links = []
    external_links = []

    for link in links:
        # Skip empty or invalid links
        if not link:
            continue

        # Ignore media files and resources
        if re.search(r"\.(jpg|jpeg|png|gif|mp4|webp|svg|css|js|ico|woff|woff2|ttf|eot|pdf|zip|tar|gz|rar)$", link):
            continue
            
        # Skip non-content URLs
        if not is_valid_content_url(link):
            continue

        link_base_domain = get_base_domain(link)
        
        # Categorize based on domain
        if link_base_domain == base_domain:
            internal_links.append(link)
        else:
            external_links.append(link)

    return internal_links, external_links


def crawl(
    url, depth=1, max_depth=3, max_links=4, visited=None, all_links=None, external_taken=False
):
    """
    Crawls a website starting from the given URL, up to a specified depth.
    
    Args:
        url (str): The starting URL to crawl
        depth (int): Current crawl depth
        max_depth (int): Maximum depth to crawl
        max_links (int): Maximum number of links to fetch per page
        visited (set): Set of already visited URLs
        all_links (list): List to store all collected links
        external_taken (bool): Whether an external link has already been followed
        
    Returns:
        list: All collected links
    """
    if visited is None:
        visited = set()
    
    if all_links is None:
        all_links = []
    
    # Normalize the input URL
    url = normalize_url(url)
    
    # Add normalized URL to visited set
    if url in visited:
        return all_links

    # Check depth limit
    if depth > max_depth:
        return all_links

    print(f"Crawling (depth {depth}/{max_depth}): {url}")
    visited.add(url)

    # Fetch links from the page using Scraping API
    links = fetch_links_with_api(url, max_links=max_links)
    if not links:
        return all_links

    # Separate internal and external links
    internal_links, external_links = filter_links_by_structure(url, links)

    # First process internal links to prioritize them
    for link in internal_links:
        if link not in all_links:
            all_links.append(link)
        if link not in visited and depth < max_depth:
            crawl(
                url=link,
                depth=depth + 1,
                max_depth=max_depth,
                max_links=max_links,
                visited=visited,
                all_links=all_links,
                external_taken=external_taken
            )

    # Then process one external link if allowed
    if not external_taken and depth < max_depth and external_links:
        # Take the first external link
        link = external_links[0]
        if link not in all_links:
            all_links.append(link)
            print(f"Processing external link: {link}")
            crawl(
                url=link,
                depth=depth + 1,
                max_depth=max_depth,
                max_links=max_links,
                visited=visited,
                all_links=all_links,
                external_taken=True
            )

    return all_links


if __name__ == "__main__":
    url = input("Enter the webpage URL: ").strip()

    # Start crawling from depth 1 with a maximum depth of 3
    all_collected_links = crawl(url, depth=1, max_depth=3, external_taken=False)

    if all_collected_links:
        file_path = save_to_project_dir("\n".join(all_collected_links), "links.txt", "data")
        print(f"\nCrawled links saved at: {file_path}")
        print(f"Total links collected: {len(all_collected_links)}")
    else:
        print("\nNo relevant links found.")