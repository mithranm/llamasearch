# llamasearch/core/pcrawler.py

import requests
import re
import os
import json
import hashlib
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup, Tag
from datetime import datetime
from typing import Optional, List, Tuple, cast

from . import apiauth

# API URL for scraping
SCRAPING_API_URL = "https://api.mithran.org/markdown/"

def find_project_root():
    """Finds the root of the project by looking for pyproject.toml."""
    current_dir = os.path.abspath(os.path.dirname(__file__))
    while current_dir != os.path.dirname(current_dir):
        if os.path.exists(os.path.join(current_dir, "pyproject.toml")):
            return current_dir
        current_dir = os.path.dirname(current_dir)
    raise RuntimeError("Could not find project root. Please check your project structure.")

def clear_crawl_data_directory() -> None:
    """Clears all content from the crawl_data directory structure."""
    import shutil
    project_root = find_project_root()
    crawl_data_dir = os.path.join(project_root, "crawl_data")
    
    if os.path.exists(crawl_data_dir):
        print(f"Clearing previous crawl data from: {crawl_data_dir}")
        for item in os.listdir(crawl_data_dir):
            item_path = os.path.join(crawl_data_dir, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            elif os.path.isfile(item_path):
                os.remove(item_path)
    else:
        os.makedirs(crawl_data_dir, exist_ok=True)
        print(f"Created crawl data directory: {crawl_data_dir}")

def save_to_crawl_dir(text: str, filename: str, subdir: Optional[str] = None) -> str:
    """Saves text to a file in the crawl_data directory structure."""
    project_root = find_project_root()
    base_dir = os.path.join(project_root, "crawl_data")
    target_dir = os.path.join(base_dir, subdir) if subdir else base_dir
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
    filename = f"{url_hash}.md"
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
    path_clean = re.sub(r'(?<!:)//+', '/', parsed.path)
    normalized = f"{parsed.scheme}://{parsed.netloc}{path_clean}"
    if not path_clean:
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
    path_lower = parsed.path.lower()
    skip_patterns = [
        r'/cdn-cgi/', r'/wp-json/', r'/wp-admin/', r'/wp-content/', r'/api/', r'/#',
        r'/feed/', r'/xmlrpc\.php', r'/wp-includes/', r'/cdn/', r'/assets/', 
        r'/static/', r'/email-protection', r'/ajax/', r'/rss/', r'/login', 
        r'/signup', r'/register', r'/search'
    ]
    if any(re.search(pattern, path_lower) for pattern in skip_patterns):
        return False
    if len(parsed.query) > 50:
        return False
    return True

def get_auth_headers(key_id=None):
    """Get authentication headers using JWT token."""
    try:
        assert key_id is not None, "key_id must be provided"
        token = apiauth.generate_jwt(
            private_key_path=os.path.expanduser("~/.ssh/id_rsa"),
            key_id=key_id
        )
        return {"Authorization": f"Bearer {token}"}
    except Exception as e:
        raise RuntimeError(f"Failed to generate authentication token: {e}")

def extract_markdown_links(markdown_text: str, base_url: str) -> List[Tuple[str, float]]:
    """Extract links from markdown text and score them."""
    # Regular expression to match markdown links [text](url)
    markdown_link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
    scored_links = []
    base_domain = get_base_domain(base_url)
    
    for match in re.finditer(markdown_link_pattern, markdown_text):
        link_text = match.group(1)
        link_url = match.group(2)
        
        # Normalize the URL
        if link_url.startswith('/'):
            link_url = urljoin(base_url, link_url)
        elif not link_url.startswith(('http://', 'https://')):
            link_url = urljoin(base_url, '/' + link_url)
            
        link_url = normalize_url(link_url)
        
        # Skip media files and other non-content URLs
        if re.search(r"\.(jpg|jpeg|png|gif|mp4|webp|svg|css|js|ico|woff|woff2|ttf|eot|pdf|zip|tar|gz|rar)$", link_url):
            continue
            
        # Check if it's a valid content URL
        if not is_valid_content_url(link_url):
            continue
            
        # Basic scoring
        score = 1.0
        
        # Give higher score for links with descriptive text
        if len(link_text) > 5 and not re.match(r'^(click|here|link|more)$', link_text.lower()):
            score += 1.0
        
        # Give higher score for links on the same domain
        link_domain = get_base_domain(link_url)
        if link_domain == base_domain:
            score += 1.0
            
        scored_links.append((link_url, score))
    
    return scored_links

def fetch_and_parse(url: str, max_links: int = 10, key_id: Optional[str] = None) -> Tuple[str, List[str], str]:
    """
    Fetches content from a URL using the Scraping API and extracts both main content and links.
    
    Returns:
        tuple: (content_text, scored_links, raw_content)
    """
    try:
        print(f"\nFetching from Scraping API: {SCRAPING_API_URL}{url}")
        headers = get_auth_headers(key_id=key_id)
        response = requests.get(f"{SCRAPING_API_URL}{url}", headers=headers, timeout=45)
        response.raise_for_status()
        content = response.text
        
        # Determine if content is markdown by checking either the URL or content structure
        is_markdown = url.endswith(('.md', '.markdown')) or '<!--' in content[:100] or content.startswith('#')
        
        debug_filename = f"scraping_response_{urlparse(url).netloc.replace('.', '_')}.{'md' if is_markdown else 'html'}"
        debug_path = save_to_crawl_dir(content, filename=debug_filename, subdir="debug")
        print(f"Saved raw API response to: {debug_path}")
        
        scored_links = []
        
        if is_markdown:
            # Process as markdown
            content_text = content  # For markdown, we use the whole content
            print("Detected markdown content, extracting markdown links")
            scored_links = extract_markdown_links(content, url)
        else:
            # Process as HTML
            soup = BeautifulSoup(content, 'html.parser')
            content_text = extract_main_content(soup)
            scored_links = extract_and_score_links(soup, url)
        
        scored_links.sort(key=lambda x: x[1], reverse=True)
        top_links = [link for link, score in scored_links[:max_links]]
        print(f"Found {len(scored_links)} total links, selected top {len(top_links)}")
        for i, link in enumerate(top_links[:3]):
            print(f"Sample link {i + 1}: {link}")
        return content_text, top_links, content
    except requests.RequestException as e:
        print(f"\nError fetching from Scraping API: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status code: {e.response.status_code}")
            print(f"Response text: {e.response.text[:500]}...")
        return "", [], ""

def extract_main_content(soup: BeautifulSoup) -> str:
    """Extract the main content from a parsed HTML page."""
    main_elements = soup.select('main, article, #content, .content, #main, .main')
    if main_elements:
        return main_elements[0].get_text(strip=True)
    else:
        for script in soup(['script', 'style', 'header', 'footer', 'nav']):
            script.extract()
        return soup.body.get_text(strip=True) if soup.body else ""

def extract_and_score_links(soup: BeautifulSoup, base_url: str) -> List[Tuple[str, float]]:
    """Extract links from HTML and score them based on context."""
    base_domain = get_base_domain(base_url)
    scored_links = []
    for a_tag in soup.find_all('a', href=True):
        if not isinstance(a_tag, Tag):
            continue
        href = str(a_tag.get('href', '')).strip()
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
    results_cache = {}  # Cache results to avoid duplicate fetching
    
    while queue and len(collected_links) < target_links:
        current_url, depth = queue.pop(0)
        normalized_url = normalize_url(current_url)
        if normalized_url in collected_links or depth > max_depth:
            continue
        
        print(f"Crawling (depth {depth}/{max_depth}): {normalized_url}")
        if normalized_url in results_cache:
            content_text, links, content = results_cache[normalized_url]
        else:
            content_text, links, content = fetch_and_parse(normalized_url, max_links=max_links_per_page, key_id=key_id)
            results_cache[normalized_url] = (content_text, links, content)
        
        if content:
            content_path = save_extracted_content(normalized_url, content)
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
    
    # Save collected links and their content from cache
    if collected_links:
        links_text = "\n".join(collected_links)
        file_path = save_to_crawl_dir(links_text, "links.txt")
        print(f"\nCrawled links saved at: {file_path}")
        print(f"Total links collected: {len(collected_links)}")
        
        for url in collected_links:
            if url in results_cache:
                _, _, content = results_cache[url]
                if content:
                    file_path = save_extracted_content(url, content)
                    print(f"Saved content from {url} to {file_path}")
    
    return collected_links

if __name__ == "__main__":
    clear_crawl_data_directory()
    
    url = input("Enter the webpage URL: ").strip()
    link_count = input("Enter the number of links to collect (default 15): ").strip()
    if not url:
        print("\nError: Please provide a valid URL")
        exit(1)
    url = normalize_url(url)
    
    key_id = os.environ.get("SCRAPING_API_KEY_ID", "mithran-macos")
    print(f"Using API key ID: {key_id}")
    
    links = smart_crawl(url, target_links=int(link_count) if link_count else 15, max_depth=3, key_id=key_id)
    
    if links:
        project_root = find_project_root()
        print(f"Extracted content saved to: {os.path.join(project_root, 'crawl_data')}")
    else:
        print("\nNo links were collected during crawling.")