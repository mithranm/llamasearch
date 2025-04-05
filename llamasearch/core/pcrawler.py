# llamasearch/core/pcrawler.py

import requests
import re
import os
import json
import hashlib
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
from datetime import datetime
import shutil

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

def save_to_project_dir(text, filename="links.txt", directory="data"):
    """Saves text to a specified directory inside the project root."""
    project_root = find_project_root()
    target_dir = os.path.join(project_root, directory)
    os.makedirs(target_dir, exist_ok=True)
    file_path = os.path.join(target_dir, filename)
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(text)
    return file_path

def clear_temp_directory():
    """Clears the temp directory to ensure a clean crawl."""
    project_root = find_project_root()
    temp_dir = os.path.join(project_root, "temp")
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

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
        debug_path = save_to_project_dir(html_content, filename=debug_filename, directory="debug")
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

def smart_crawl(start_url, target_links=50, max_depth=3, max_links_per_page=10, key_id=None):
    """
    Crawls the web starting from start_url until target_links unique links are collected,
    or the crawl reaches max_depth. Clears the temp directory before crawling.
    
    Args:
        start_url (str): The starting URL.
        target_links (int): Number of unique links to collect.
        max_depth (int): Maximum crawl depth.
        max_links_per_page (int): Maximum links to extract per page.
        key_id (str, optional): For authentication.
        
    Returns:
        list: Unique collected links.
    """
    clear_temp_directory()
    visited = set()
    collected_links = []
    queue = [(start_url, 1)]
    
    while queue and len(collected_links) < target_links:
        current_url, depth = queue.pop(0)
        normalized_url = normalize_url(current_url)
        if normalized_url in visited or depth > max_depth:
            continue
        visited.add(normalized_url)
        
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
            if link not in visited:
                queue.append((link, depth + 1))
        
        if external_links:
            external_link = external_links[0]
            if external_link not in visited:
                queue.append((external_link, depth + 1))
    
    return collected_links

def update_reverse_lookup_table(hash_value, url):
    """
    Updates the reverse lookup table with a mapping from file hash to original URL.
    The table is stored as a JSON file in project_root/data/reverse_lookup.json.
    """
    project_root = find_project_root()
    data_dir = os.path.join(project_root, "data")
    os.makedirs(data_dir, exist_ok=True)
    reverse_lookup_path = os.path.join(data_dir, "reverse_lookup.json")
    
    try:
        with open(reverse_lookup_path, "r", encoding="utf-8") as file:
            reverse_lookup = json.load(file)
    except FileNotFoundError:
        reverse_lookup = {}
    
    reverse_lookup[hash_value] = url
    
    with open(reverse_lookup_path, "w", encoding="utf-8") as file:
        json.dump(reverse_lookup, file, indent=2)

def save_extracted_content(url, content, directory="temp"):
    """
    Save extracted content to a file and update the reverse lookup table.
    Embeds metadata (including source traceability) as an HTML comment.
    """
    url_hash = hashlib.sha256(url.encode()).hexdigest()
    filename = f"{url_hash}.html"
    metadata = {
        "source": url,  # Original source URL or file path
        "extracted_at": datetime.now().isoformat()
    }
    metadata_str = f"""<!--
METADATA: {json.dumps(metadata, indent=2)}
-->
"""
    full_content = metadata_str + "\n" + content
    file_path = save_to_project_dir(full_content, filename=filename, directory=directory)
    update_reverse_lookup_table(url_hash, url)
    return file_path

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
        links_text = "\n".join(links)
        file_path = save_to_project_dir(links_text)
        print(f"\nCrawled links saved at: {file_path}")
        print(f"Total links collected: {len(links)}")
        project_root = find_project_root()
        print(f"Extracted content saved to: {os.path.join(project_root, 'temp')}")
    else:
        print("\nNo links were collected during crawling.")
