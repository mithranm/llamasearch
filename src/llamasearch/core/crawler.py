# src/llamasearch/core/crawler.py

import os
import re
import json
import hashlib
import shutil
import requests
import logging
from urllib.parse import urlparse, urljoin
from datetime import datetime
from typing import List, Tuple, Optional

from llamasearch.core import apiauth

from llamasearch.setup_utils import get_data_paths
from llamasearch.utils import setup_logging

# API URLs
JINA_API_URL = "https://r.jina.ai/"
MITHRAN_API_URL = "https://api.mithran.org/markdown/"

# Setup logging
logger = setup_logging(__name__)

def normalize_url(url: str) -> str:
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

def get_base_domain(url: str) -> str:
    """Extract the base domain from a URL."""
    parsed_url = urlparse(url)
    netloc_parts = parsed_url.netloc.split(".")
    
    if len(netloc_parts) >= 3 and netloc_parts[-2] in ["co", "com", "org", "net", "ac", "gov", "edu"]:
        return ".".join(netloc_parts[-3:])
    else:
        return ".".join(netloc_parts[-2:])

def is_valid_content_url(url: str) -> bool:
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
        
    # Skip media files and other non-content URLs
    if re.search(r"\.(jpg|jpeg|png|gif|mp4|webp|svg|css|js|ico|woff|woff2|ttf|eot|pdf|zip|tar|gz|rar)$", url):
        return False
        
    if len(parsed.query) > 50:
        return False
        
    return True

def generate_jwt(private_key_path: str, key_id: str) -> Optional[str]:
    """Generate JWT token for Mithran API authentication."""
    try:
        token = apiauth.generate_jwt(
            private_key_path=os.path.expanduser(private_key_path),
            key_id=key_id
        )
        return token
    except Exception as e:
        logger.error(f"Failed to generate JWT token: {e}")
        return None

def clear_crawl_data_directory() -> None:
    """Clears all content from the crawl_data directory structure."""
    paths = get_data_paths()
    crawl_data_dir = paths["crawl_data"]
    
    shutil.rmtree(crawl_data_dir, ignore_errors=True)
    logger.info(f"Cleared crawl data directory: {crawl_data_dir}")
    
    # Recreate directory structure
    os.makedirs(crawl_data_dir, exist_ok=True)
    os.makedirs(crawl_data_dir / "raw", exist_ok=True)
    os.makedirs(crawl_data_dir / "debug", exist_ok=True)
    logger.info(f"Created crawl data directory structure at: {crawl_data_dir}")

def update_reverse_lookup_table(hash_value: str, url: str) -> None:
    """Updates the reverse lookup table mapping file hashes to original URLs."""
    paths = get_data_paths()
    lookup_path = paths["crawl_data"] / "reverse_lookup.json"
    
    try:
        with open(lookup_path, "r", encoding="utf-8") as file:
            reverse_lookup = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        reverse_lookup = {}
        
    reverse_lookup[hash_value] = url
    
    with open(lookup_path, "w", encoding="utf-8") as file:
        json.dump(reverse_lookup, file, indent=2)

def save_extracted_content(url: str, content: str) -> str:
    """Save extracted content and update the reverse lookup table."""
    paths = get_data_paths()
    raw_dir = paths["crawl_data"] / "raw"
    
    # Create hash of URL for filename
    url_hash = hashlib.sha256(url.encode()).hexdigest()
    filename = f"{url_hash}.md"
    
    # Add metadata
    metadata = {
        "source": url,
        "extracted_at": datetime.now().isoformat()
    }
    metadata_str = f"""<!--
METADATA: {json.dumps(metadata, indent=2)}
-->
"""
    full_content = metadata_str + "\n" + content
    
    # Save to file
    file_path = raw_dir / filename
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(full_content)
        
    # Update lookup table
    update_reverse_lookup_table(url_hash, url)
    
    return str(file_path)

def extract_and_score_links(content: str, base_url: str, use_jina: bool = True) -> List[Tuple[str, float]]:
    """Extract links from content and score them based on context."""
    if use_jina:
        # For Jina API, content is already markdown, extract links with regex
        links = set(re.findall(r'https?://[^\s)>"]+', content))
        scored_links = []
        
        for link in links:
            if is_valid_content_url(link):
                # Basic scoring - prefer same domain
                score = 1.0
                if get_base_domain(link) == get_base_domain(base_url):
                    score += 1.0
                scored_links.append((link, score))
                
        return scored_links
    else:
        # For Mithran API / HTML content
        try:
            from bs4 import BeautifulSoup, Tag
            soup = BeautifulSoup(content, 'html.parser')
            
            base_domain = get_base_domain(base_url)
            scored_links = []
            
            for a_tag in soup.find_all('a', href=True):
                # Type assertion to satisfy Pyright
                if not isinstance(a_tag, Tag):
                    continue
                    
                href = str(a_tag.get('href', '')).strip()
                
                if not href:
                    continue
                    
                # Handle relative URLs
                if href.startswith('/'):
                    link = urljoin(base_url, href)
                elif not href.startswith(('http://', 'https://')):
                    link = urljoin(base_url, '/' + href)
                else:
                    link = href
                    
                link = normalize_url(link)
                
                if not is_valid_content_url(link):
                    continue
                    
                # Score the link
                score = 1.0
                
                # Prefer links with descriptive text
                link_text = a_tag.get_text(strip=True)
                if len(link_text) > 5 and not re.match(r'^(click|here|link|more)$', link_text.lower()):
                    score += 1.0
                
                # Prefer links on the same domain
                link_domain = get_base_domain(link)
                if link_domain == base_domain:
                    score += 1.0
                    
                scored_links.append((link, score))
                
            return scored_links
        except ImportError:
            # If BeautifulSoup is not available, fall back to regex
            logger.warning("BeautifulSoup not available, falling back to regex-based link extraction")
            links = set(re.findall(r'https?://[^\s)>"]+', content))
            return [(link, 1.0) for link in links if is_valid_content_url(link)]

def fetch_content(url: str, api_type: str = "jina", private_key_path: Optional[str] = None, key_id: Optional[str] = None) -> Tuple[str, List[Tuple[str, float]], bool]:
    """
    Fetch content from a URL using either Jina or Mithran API.
    
    Args:
        url: URL to fetch content from
        api_type: "jina" or "mithran"
        private_key_path: Path to RSA private key (for Mithran API)
        key_id: API key ID (for Mithran API)
        
    Returns:
        Tuple of (content, scored_links, is_success)
    """
    use_jina = (api_type.lower() == "jina")
    
    try:
        if use_jina:
            # Fetch content using Jina API
            logger.info(f"Fetching content using Jina API: {url}")
            response = requests.get(f"{JINA_API_URL}{url}", timeout=30)
            response.raise_for_status()
            content = response.text
            
            # Extract links from the content
            scored_links = extract_and_score_links(content, url, use_jina=True)
            
            return content, scored_links, True
        else:
            # Fetch content using Mithran API
            logger.info(f"Fetching content using Mithran API: {url}")
            
            if not key_id:
                key_id = os.environ.get("MITHRAN_API_KEY_ID")
                
            if not private_key_path:
                private_key_path = os.environ.get("MITHRAN_PRIVATE_KEY_PATH", "~/.ssh/id_rsa")
                
            if not key_id:
                logger.error("No API key ID provided for Mithran API")
                return "", [], False
                
            # Generate JWT token
            token = generate_jwt(private_key_path, key_id)
            if not token:
                logger.error("Failed to generate JWT token")
                return "", [], False
                
            headers = {"Authorization": f"Bearer {token}"}
            response = requests.get(f"{MITHRAN_API_URL}{url}", headers=headers, timeout=45)
            response.raise_for_status()
            content = response.text
            
            # Determine if content is HTML or markdown
            is_markdown = url.endswith(('.md', '.markdown')) or '<!--' in content[:100] or content.startswith('#')
            
            # Save debug content
            paths = get_data_paths()
            debug_filename = f"scraping_response_{urlparse(url).netloc.replace('.', '_')}.{'md' if is_markdown else 'html'}"
            debug_path = paths["crawl_data"] / "debug" / debug_filename
            with open(debug_path, "w", encoding="utf-8") as f:
                f.write(content)
                
            # Extract links using appropriate method
            scored_links = extract_and_score_links(content, url, use_jina=is_markdown)
            
            return content, scored_links, True
    except Exception as e:
        logger.error(f"Error fetching content from {url}: {e}")
        return "", [], False

def crawl(start_url: str, 
          target_links: int = 15, 
          max_depth: int = 2, 
          api_type: str = "jina",
          private_key_path: Optional[str] = None, 
          key_id: Optional[str] = None) -> List[str]:
    """
    Crawl a website starting from the given URL.
    
    Args:
        start_url: Starting URL for crawling
        target_links: Maximum number of links to collect
        max_depth: Maximum crawl depth
        api_type: "jina" or "mithran"
        private_key_path: Path to RSA private key (for Mithran API)
        key_id: API key ID (for Mithran API)
        
    Returns:
        List of URLs that were successfully crawled
    """
    # Clear and initialize crawl data directory
    clear_crawl_data_directory()
    
    # Normalize the starting URL
    start_url = normalize_url(start_url)
    logger.info(f"Starting crawl from {start_url}")
    logger.info(f"Using API: {api_type}")
    logger.info(f"Max links: {target_links}, Max depth: {max_depth}")
    
    # Initialize crawl state
    collected_links = []
    queue = [(start_url, 1)]  # (url, depth)
    processed_urls = set()
    
    while queue and len(collected_links) < target_links:
        # Get next URL to process
        current_url, depth = queue.pop(0)
        
        # Skip if already processed or too deep
        if current_url in processed_urls or depth > max_depth:
            continue
        
        # Add to processed set
        processed_urls.add(current_url)
        
        logger.info(f"Crawling (depth {depth}/{max_depth}): {current_url}")
        
        # Fetch content
        content, scored_links, success = fetch_content(
            current_url,
            api_type,
            private_key_path,
            key_id
        )
        
        if success and content:
            # Save the content
            file_path = save_extracted_content(current_url, content)
            logger.info(f"Saved content to: {file_path}")
            
            # Add to collected links
            collected_links.append(current_url)
            
            # Stop if we've reached the target
            if len(collected_links) >= target_links:
                break
            
            # Sort links by score and add to queue
            if scored_links:
                # Sort by score (highest first)
                scored_links.sort(key=lambda x: x[1], reverse=True)
                
                # Separate internal and external links
                base_domain = get_base_domain(current_url)
                internal_links = []
                external_links = []
                
                for link, _ in scored_links:
                    if link not in processed_urls:
                        if get_base_domain(link) == base_domain:
                            internal_links.append(link)
                        else:
                            external_links.append(link)
                
                # Add internal links to queue first
                for link in internal_links:
                    queue.append((link, depth + 1))
                
                # Add one external link to maintain diversity
                if external_links and depth < max_depth:
                    queue.append((external_links[0], depth + 1))
    
    # Save a list of all collected links
    if collected_links:
        paths = get_data_paths()
        links_file = paths["crawl_data"] / "links.txt"
        with open(links_file, "w", encoding="utf-8") as f:
            f.write("\n".join(collected_links))
        
        logger.info(f"Crawl complete. Collected {len(collected_links)} links.")
    else:
        logger.warning("Crawl complete, but no links were collected.")
    
    return collected_links

def smart_crawl(start_url: str, 
                target_links: int = 20, 
                max_depth: int = 2, 
                api_type: str = "jina",
                private_key_path: Optional[str] = None, 
                key_id: Optional[str] = None) -> List[str]:
    """
    Smart crawl wrapper that selects the appropriate API and handles errors.
    
    Args:
        start_url: Starting URL to crawl
        target_links: Maximum number of links to collect
        max_depth: Maximum crawl depth
        api_type: "jina" (default) or "mithran"
        private_key_path: Path to RSA private key (for Mithran API)
        key_id: API key ID (for Mithran API)
        
    Returns:
        List of URLs that were successfully crawled
    """
    # Detect if Mithran API access is possible
    if api_type.lower() == "mithran":
        if not key_id:
            logger.warning("Mithran API requested but no key_id provided. Falling back to Jina API.")
            api_type = "jina"
    
    # Normalize the URL
    try:
        start_url = normalize_url(start_url)
    except Exception as e:
        logger.error(f"Error normalizing URL: {e}")
        return []
    
    # Start crawling
    try:
        return crawl(
            start_url=start_url,
            target_links=target_links,
            max_depth=max_depth,
            api_type=api_type,
            private_key_path=private_key_path,
            key_id=key_id
        )
    except Exception as e:
        logger.error(f"Error during crawl: {e}")
        return []

# Command-line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="LlamaSearch Web Crawler")
    parser.add_argument("url", help="Starting URL to crawl")
    parser.add_argument("--api", choices=["jina", "mithran"], default="jina", help="API to use (default: jina)")
    parser.add_argument("--links", type=int, default=15, help="Maximum links to collect (default: 15)")
    parser.add_argument("--depth", type=int, default=2, help="Maximum crawl depth (default: 2)")
    parser.add_argument("--key-id", help="Mithran API key ID")
    parser.add_argument("--private-key", help="Path to RSA private key for Mithran API")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    # Start crawling
    links = smart_crawl(
        start_url=args.url,
        target_links=args.links,
        max_depth=args.depth,
        api_type=args.api,
        private_key_path=args.private_key,
        key_id=args.key_id
    )
    
    if links:
        paths = get_data_paths()
        print(f"\nCrawl complete. Collected {len(links)} links.")
        print(f"Crawl data saved to: {paths['crawl_data']}")
    else:
        print("\nCrawl failed: No links were collected.")