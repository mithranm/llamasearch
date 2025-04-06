# llamasearch/core/pcrawler.py

import requests
import re
import os
import json
import hashlib
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
from datetime import datetime
from typing import Optional, List

from . import apiauth

# API URL for scraping
SCRAPING_API_URL = "https://api.mithran.org/html/"

def find_project_root():
    """Finds the root of the project by looking for pyproject.toml."""
    current_dir = os.path.abspath(os.path.dirname(__file__))
    while current_dir != os.path.dirname(current_dir):
        if os.path.exists(os.path.join(current_dir, "pyproject.toml")):
            return current_dir
        current_dir = os.path.dirname(current_dir)
    raise RuntimeError("Could not find project root. Please check your project structure.")

def save_to_crawl_dir(text: str, filename: str, subdir: Optional[str] = None) -> str:
    """Saves text to a file in the crawl_data directory structure.
    
    Args:
        text: Content to save
        filename: Name of the file
        subdir: Optional subdirectory within crawl_data (e.g., 'raw')
        
    Returns:
        str: Full path to the saved file
    """
    project_root = find_project_root()
    base_dir = os.path.join(project_root, "crawl_data")
    if subdir:
        target_dir = os.path.join(base_dir, subdir)
    else:
        target_dir = base_dir
    
    os.makedirs(target_dir, exist_ok=True)
    file_path = os.path.join(target_dir, filename)
    
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(text)
    
    return file_path

def update_reverse_lookup_table(hash_value: str, url: str) -> None:
    """Updates the reverse lookup table mapping file hashes to original URLs."""
    project_root = find_project_root()
    lookup_path = os.path.join(project_root, "crawl_data", "reverse_lookup.json")
    
    try:
        with open(lookup_path, "r", encoding="utf-8") as file:
            reverse_lookup = json.load(file)
    except FileNotFoundError:
        reverse_lookup = {}
    
    reverse_lookup[hash_value] = url
    
    with open(lookup_path, "w", encoding="utf-8") as file:
        json.dump(reverse_lookup, file, indent=2)

def save_extracted_content(url: str, content: str) -> str:
    """Save extracted content and update the reverse lookup table."""
    url_hash = hashlib.sha256(url.encode()).hexdigest()
    filename = f"{url_hash}.html"
    
    metadata = {
        "source": url,
        "extracted_at": datetime.now().isoformat()
    }
    metadata_str = f"""<!--
METADATA: {json.dumps(metadata, indent=2)}
-->
"""
    full_content = metadata_str + "\n" + content
    file_path = save_to_crawl_dir(full_content, filename, subdir="raw")
    update_reverse_lookup_table(url_hash, url)
    return file_path

def normalize_url(url):
    """Normalize URL by adding https:// if missing and handling paths."""
    if not url:
        return url
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    parsed = urlparse(url)
    path = re.sub(r'(?<!:)//+', '/', parsed.path)
    normalized = f"{parsed.scheme}://{parsed.netloc}{path}"
    if not path:
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
    if len(netloc_parts) >= 3 and netloc_parts[-2] in ["co", "com", "org", "net", "ac", "gov", "edu"]:
        return ".".join(netloc_parts[-3:])
    else:
        return ".".join(netloc_parts[-2:])

def is_valid_content_url(url):
    """Check if a URL likely points to actual content rather than utility endpoints."""
    parsed = urlparse(url)
    path = parsed.path.lower()
    skip_patterns = [
        r'/cdn-cgi/', r'/wp-json/', r'/wp-admin/', r'/wp-content/', r'/api/', r'/#',
        r'/feed/', r'/xmlrpc\.php', r'/wp-includes/', r'/cdn/', r'/assets/', 
        r'/static/', r'/email-protection', r'/ajax/', r'/rss/', r'/login', 
        r'/signup', r'/register', r'/search'
    ]
    if any(re.search(pattern, path) for pattern in skip_patterns):
        return False
    if len(parsed.query) > 50:
        return False
    return True

def get_auth_headers(key_id=None):
    """Get authentication headers using JWT token.
    
    Args:
        key_id (str, optional): Unique identifier for the key.
    """
    try:
        assert key_id is not None, "key_id must be provided"
        token = apiauth.generate_jwt(
            private_key_path=os.path.expanduser("~/.ssh/id_rsa"),
            key_id=key_id
        )
        return {"Authorization": f"Bearer {token}"}
    except Exception as e:
        raise RuntimeError(f"Failed to generate authentication token: {e}")

def fetch_and_parse(url, max_links=10, key_id=None):
    """
    Fetches content from a URL using the Scraping API and extracts both main content and links.
    
    Returns:
        tuple: (content_text, scored_links, raw_html)
    """
    try:
        print(f"\nFetching from Scraping API: {SCRAPING_API_URL}{url}")
        headers = get_auth_headers(key_id=key_id)
        response = requests.get(f"{SCRAPING_API_URL}{url}", headers=headers, timeout=45)
        response.raise_for_status()
        html_content = response.text
        debug_filename = f"scraping_response_{urlparse(url).netloc.replace('.', '_')}.html"
        debug_path = save_to_crawl_dir(html_content, filename=debug_filename, subdir="debug")
        print(f"Saved raw API response to: {debug_path}")
        soup = BeautifulSoup(html_content, 'html.parser')
        content_text = extract_main_content(soup)
        scored_links = extract_and_score_links(soup, url)
        scored_links.sort(key=lambda x: x[1], reverse=True)
        top_links = [link for link, score in scored_links[:max_links]]
        print(f"Found {len(scored_links)} total links, selected top {len(top_links)}")
        for i, link in enumerate(top_links[:3]):
            print(f"Sample link {i + 1}: {link}")
        return content_text, top_links, html_content
    except requests.RequestException as e:
        print(f"\nError fetching from Scraping API: {e}")
        print(f"Full error details: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status code: {e.response.status_code}")
            print(f"Response text: {e.response.text[:500]}...")
        return None, [], None

def extract_main_content(soup):
    """Extract the main content from a parsed HTML page."""
    main_content = ""
    content_elements = soup.select('main, article, #content, .content, #main, .main')
    if content_elements:
        main_content = content_elements[0].get_text(strip=True)
    else:
        for script in soup(['script', 'style', 'header', 'footer', 'nav']):
            script.extract()
        main_content = soup.body.get_text(strip=True) if soup.body else ""
    return main_content

def extract_and_score_links(soup, base_url):
    """
    Extract links from HTML and score them based on their context.
    
    Returns:
        list: (link, score) tuples.
    """
    base_domain = get_base_domain(base_url)
    scored_links = []
    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href'].strip()
        if not href:
            continue
        if href.startswith('/'):
            link = urljoin(base_url, href)
        elif not href.startswith(('http://', 'https://')):
            link = urljoin(base_url, '/' + href)
        else:
            link = href
        link = normalize_url(link)
        if re.search(r"\.(jpg|jpeg|png|gif|mp4|webp|svg|css|js|ico|woff|woff2|ttf|eot|pdf|zip|tar|gz|rar)$", link):
            continue
        if not is_valid_content_url(link):
            continue
        score = 1.0
        parent_tags = [parent.name for parent in a_tag.parents if parent.name]
        if any(tag in ['article', 'main', 'section', 'p'] for tag in parent_tags):
            score += 2.0
        if any(tag in ['nav', 'footer', 'header', 'aside'] for tag in parent_tags):
            score -= 0.5
        link_text = a_tag.get_text(strip=True)
        if len(link_text) > 5 and not re.match(r'^(click|here|link|more)$', link_text.lower()):
            score += 1.0
        link_domain = get_base_domain(link)
        if link_domain == base_domain:
            score += 1.0
        if re.search(r'/[a-z]{2}(-[a-z]{2})?\.html$', link) or re.search(r'\.([a-z]{2}|[a-z]{2}-[a-z]{2})$', link):
            score -= 1.5
        scored_links.append((link, score))
    return scored_links

def smart_crawl(start_url: str, target_links: int = 50, max_depth: int = 3, max_links_per_page: int = 10, key_id: Optional[str] = None) -> List[str]:
    """Crawls the web starting from start_url."""
    collected_links = []
    queue = [(start_url, 1)]
    
    while queue and len(collected_links) < target_links:
        current_url, depth = queue.pop(0)
        normalized_url = normalize_url(current_url)
        if normalized_url in collected_links or depth > max_depth:
            continue
        
        print(f"Crawling (depth {depth}/{max_depth}): {normalized_url}")
        content_text, links, html = fetch_and_parse(normalized_url, max_links=max_links_per_page, key_id=key_id)
        if html:
            content_path = save_extracted_content(normalized_url, html)
            print(f"Saved content to: {content_path}")
        
        if normalized_url not in collected_links:
            collected_links.append(normalized_url)
        
        if len(collected_links) >= target_links:
            break
        
        if not links:
            continue
        
        base_domain = get_base_domain(normalized_url)
        internal_links = []
        external_links = []
        for link in links:
            link_domain = get_base_domain(link)
            if link_domain == base_domain:
                internal_links.append(link)
            else:
                external_links.append(link)
        
        for link in internal_links:
            if link not in collected_links:
                queue.append((link, depth + 1))
        
        if external_links:
            external_link = external_links[0]
            if external_link not in collected_links:
                queue.append((external_link, depth + 1))
    
    # Save collected links
    if collected_links:
        links_text = "\n".join(collected_links)
        file_path = save_to_crawl_dir(links_text, "links.txt")
        print(f"\nCrawled links saved at: {file_path}")
        print(f"Total links collected: {len(collected_links)}")
        
        # Save extracted content
        for url in collected_links:
            try:
                content_text, _, html_content = fetch_and_parse(url, key_id=key_id)
                if content_text and html_content:
                    file_path = save_extracted_content(url, html_content)
                    print(f"Saved content from {url} to {file_path}")
            except Exception as e:
                print(f"Error processing {url}: {e}")
    
    return collected_links

if __name__ == "__main__":
    url = input("Enter the webpage URL: ").strip()
    if not url:
        print("\nError: Please provide a valid URL")
        exit(1)
    url = normalize_url(url)
    if not url.startswith(("http://", "https://")):
        print("\nError: Invalid URL format. URL must start with http:// or https://")
        exit(1)
    
    key_id = os.environ.get("SCRAPING_API_KEY_ID", "mithran-windows")
    print(f"Using API key ID: {key_id}")
    
    target_links_count = 15
    links = smart_crawl(url, target_links=target_links_count, max_depth=3, max_links_per_page=10, key_id=key_id)
    
    if links:
        project_root = find_project_root()
        print(f"Extracted content saved to: {os.path.join(project_root, 'crawl_data')}")
    else:
        print("\nNo links were collected during crawling.")
