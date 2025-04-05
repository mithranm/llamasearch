import requests
import re
import os
import json
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
from datetime import datetime

from . import apiauth

# API URL for scraping
SCRAPING_API_URL = "https://postgres.mithran.org/html/"

def find_project_root():
    """Finds the root of the project by looking for pyproject.toml."""
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

def normalize_url(url):
    """Normalize URL by adding https:// if missing and handling paths."""
    if not url:
        return url
        
    # Add scheme if missing
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
        
    # Parse URL
    parsed = urlparse(url)
    
    # Fix double slashes in path (except after protocol)
    path = re.sub(r'(?<!:)//+', '/', parsed.path)
    
    # Reconstruct URL with fixed path
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

    # Handle special TLDs like .co.uk, .com.au, etc.
    if len(netloc_parts) >= 3 and netloc_parts[-2] in [
        "co", "com", "org", "net", "ac", "gov", "edu",
    ]:
        # For domains like example.co.uk, take the last three parts
        return ".".join(netloc_parts[-3:])
    else:
        # For regular domains like example.com, take the last two parts
        return ".".join(netloc_parts[-2:])

def is_valid_content_url(url):
    """Check if a URL likely points to actual content rather than utility endpoints."""
    parsed = urlparse(url)
    path = parsed.path.lower()
    
    # Skip utility paths
    skip_patterns = [
        r'/cdn-cgi/',           # Cloudflare utilities
        r'/wp-json/',           # WordPress API
        r'/wp-admin/',          # WordPress admin
        r'/wp-content/',        # WordPress resources
        r'/api/',               # Generic API endpoints
        r'/#',                  # Anchor links
        r'/feed/',              # RSS/Atom feeds
        r'/xmlrpc\.php',        # XML-RPC endpoints
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

def get_auth_headers():
    """Get authentication headers using JWT token."""
    try:
        # Use default Ed25519 key from ~/.ssh/id_ed25519
        token = apiauth.generate_jwt(os.path.expanduser("~/.ssh/id_ed25519"), algorithm="Ed25519")
        return {"Authorization": f"Bearer {token}"}
    except Exception as e:
        raise RuntimeError(f"Failed to generate authentication token: {e}")

def fetch_and_parse(url, max_links=10):
    """
    Fetches content from a URL and extracts both content and links with scores.
    
    Returns:
        tuple: (content_text, scored_links, raw_html)
    """
    try:
        print(f"\nFetching from Scraping API: {SCRAPING_API_URL}{url}")
        headers = get_auth_headers()
        response = requests.get(f"{SCRAPING_API_URL}{url}", headers=headers, timeout=45)
        response.raise_for_status()
        html_content = response.text
        
        # Save raw API response for debugging
        debug_filename = f"scraping_response_{urlparse(url).netloc.replace('.', '_')}.html"
        debug_path = save_to_project_dir(html_content, filename=debug_filename, directory="debug")
        print(f"Saved raw API response to: {debug_path}")
        
        # Parse HTML with BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract main content
        content_text = extract_main_content(soup)
        
        # Extract and score links
        scored_links = extract_and_score_links(soup, url)
        
        # Sort links by score (descending)
        scored_links.sort(key=lambda x: x[1], reverse=True)
        
        # Take top N links
        top_links = [link for link, score in scored_links[:max_links]]
        print(f"Found {len(scored_links)} total links, selected top {len(top_links)}")
        
        # Log the first few links for debugging
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
    # Try to find main content area
    main_content = ""
    
    # Look for common content containers
    content_elements = soup.select('main, article, #content, .content, #main, .main')
    
    if content_elements:
        # Use the first content element found
        main_content = content_elements[0].get_text(strip=True)
    else:
        # Fallback: get text from body, excluding scripts, styles, etc.
        for script in soup(['script', 'style', 'header', 'footer', 'nav']):
            script.extract()
        main_content = soup.body.get_text(strip=True) if soup.body else ""
    
    return main_content

def extract_and_score_links(soup, base_url):
    """
    Extract links from HTML and score them based on their context and position.
    Returns a list of (link, score) tuples.
    """
    base_domain = get_base_domain(base_url)
    scored_links = []
    
    # Process all links
    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href'].strip()
        
        # Skip empty links
        if not href:
            continue
            
        # Convert relative URLs to absolute
        if href.startswith('/'):
            link = urljoin(base_url, href)
        elif not href.startswith(('http://', 'https://')):
            link = urljoin(base_url, '/' + href)
        else:
            link = href
            
        # Normalize the URL
        link = normalize_url(link)
        
        # Skip media files and resources
        if re.search(r"\.(jpg|jpeg|png|gif|mp4|webp|svg|css|js|ico|woff|woff2|ttf|eot|pdf|zip|tar|gz|rar)$", link):
            continue
            
        # Skip non-content URLs
        if not is_valid_content_url(link):
            continue
            
        # Calculate base score
        score = 1.0
        
        # Boost score based on context
        parent_tags = [parent.name for parent in a_tag.parents if parent.name]
        
        # Boost links in content areas
        if any(tag in ['article', 'main', 'section', 'p'] for tag in parent_tags):
            score += 2.0
            
        # Penalize links in navigation, footer, etc.
        if any(tag in ['nav', 'footer', 'header', 'aside'] for tag in parent_tags):
            score -= 0.5
            
        # Boost links with descriptive text
        link_text = a_tag.get_text(strip=True)
        if len(link_text) > 5 and not re.match(r'^(click|here|link|more)$', link_text.lower()):
            score += 1.0
            
        # Boost internal links
        link_domain = get_base_domain(link)
        if link_domain == base_domain:
            score += 1.0
            
        # Penalize likely language variants
        if re.search(r'/[a-z]{2}(-[a-z]{2})?\.html$', link) or re.search(r'\.([a-z]{2}|[a-z]{2}-[a-z]{2})$', link):
            score -= 1.5
            
        # Add to results
        scored_links.append((link, score))
    
    return scored_links

def smart_crawl(url, depth=1, max_depth=3, max_links=10, visited=None, all_links=None, external_taken=False):
    """
    Smart crawler that uses content analysis to make better crawling decisions.
    
    Args:
        url (str): The starting URL to crawl
        depth (int): Current crawl depth
        max_depth (int): Maximum depth to crawl
        max_links (int): Maximum number of links to follow per page
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
    
    # Skip if already visited
    if url in visited:
        return all_links
    
    # Check depth limit
    if depth > max_depth:
        return all_links
    
    print(f"Crawling (depth {depth}/{max_depth}): {url}")
    visited.add(url)
    
    # Fetch and parse the page
    content_text, links, html = fetch_and_parse(url, max_links=max_links)
    
    # If fetch failed, return current links
    if not links:
        return all_links
        
    # Save extracted content to file
    if html:
        content_path = save_extracted_content(url, html)
        print(f"Saved content to: {content_path}")
    
    # Add current URL to all_links if not already present
    if url not in all_links:
        all_links.append(url)
    
    # Separate internal and external links
    base_domain = get_base_domain(url)
    internal_links = []
    external_links = []
    
    for link in links:
        link_domain = get_base_domain(link)
        if link_domain == base_domain:
            internal_links.append(link)
        else:
            external_links.append(link)
    
    # Process internal links first
    for link in internal_links:
        if link not in visited and depth < max_depth:
            smart_crawl(
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
        print(f"Processing external link: {link}")
        smart_crawl(
            url=link,
            depth=depth + 1,
            max_depth=max_depth,
            max_links=max_links,
            visited=visited,
            all_links=all_links,
            external_taken=True
        )
    
    return all_links

def save_extracted_content(url, content, directory="temp"):
    """Save extracted content to a file."""
    # Create a filename from the URL
    filename = re.sub(r"https?://(www\.)?", "", url)
    filename = re.sub(r"[^a-zA-Z0-9_]+", "-", filename)
    filename = filename.strip("-")[:200] + ".html"
    
    # Create metadata
    metadata = {
        "source_url": url,
        "extracted_at": datetime.now().isoformat(),
        "extractor_version": "1.0"
    }
    
    metadata_str = f"""<!--
METADATA: {json.dumps(metadata, indent=2)}
-->

"""
    
    # Save to project directory
    return save_to_project_dir(metadata_str + content, filename=filename, directory=directory)

if __name__ == "__main__":
    url = input("Enter the webpage URL: ").strip()
    
    # Normalize the URL
    url = normalize_url(url)
    
    # Create temp directory for extracted content
    project_root = find_project_root()
    temp_dir = os.path.join(project_root, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Crawl the website
    links = smart_crawl(url, max_depth=3, max_links=10)
    
    # Save the links to a file
    if links:
        links_text = "\n".join(links)
        file_path = save_to_project_dir(links_text)
        print(f"\nCrawled links saved at: {file_path}")
        print(f"Total links collected: {len(links)}")
        print(f"Extracted content saved to: {os.path.join(project_root, 'temp')}")
    else:
        print("\nNo links were collected during crawling.")