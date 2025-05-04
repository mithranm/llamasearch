#!/usr/bin/env python3
"""
crawler.py – Asynchronous BFS crawler with concurrency and export functionality.

Each crawl run creates a unique crawl folder (named with a sanitized root URL,
the target phrase hash, and a timestamp). After crawling, the entire crawl folder
(can include raw markdown files and a reverse lookup JSON) can be exported as a tar.gz archive.
"""

import asyncio
import time
import re
import logging
import requests
import threading
from typing import List, Optional, Callable, Dict, Set
from urllib.parse import urlparse, urljoin
from pathlib import Path
import json
import os
import hashlib
import tarfile

logger = logging.getLogger(__name__)

# Global rate limit for Jina
JINA_REQUEST_TIMES: List[float] = []
JINA_LOCK = threading.Lock()

def check_jina_rate_limit(rate_limit: int = 20, interval: int = 60):
    with JINA_LOCK:
        now = time.time()
        while JINA_REQUEST_TIMES and (now - JINA_REQUEST_TIMES[0] > interval):
            JINA_REQUEST_TIMES.pop(0)
        if len(JINA_REQUEST_TIMES) >= rate_limit:
            oldest = JINA_REQUEST_TIMES[0]
            to_sleep = interval - (now - oldest)
            if to_sleep > 0:
                logger.info(f"[RateLimit] Sleeping {to_sleep:.1f}s...")
                time.sleep(to_sleep)
        JINA_REQUEST_TIMES.append(time.time())

def synchronous_fetch(url: str, api_type: str="jina", key_id: Optional[str]=None, private_key: Optional[str]=None) -> str:
    try:
        if api_type == "jina":
            check_jina_rate_limit()
            full_url = f"https://r.jina.ai/{url}"
            r = requests.get(full_url, timeout=30)
            r.raise_for_status()
            return r.text
        else:
            full_url = f"https://api.mithran.org/markdown/{url}"
            r = requests.get(full_url, timeout=45)
            r.raise_for_status()
            return r.text
    except Exception as e:
        logger.error(f"[Fetch] Error fetching {url}: {e}")
        raise

def sanitize_string(s: str, max_length: int = 30) -> str:
    """Convert to a filesystem-friendly string."""
    s = re.sub(r'https?://', '', s)
    s = re.sub(r'[^A-Za-z0-9\-_]', '_', s)
    s = re.sub(r'_+', '_', s)
    return s[:max_length]

def generate_archive_filename(root_url: str, phrase: str) -> str:
    sanitized_url = sanitize_string(root_url)
    sanitized_phrase = sanitize_string(phrase)
    combined = f"{root_url}_{phrase}"
    hash_digest = hashlib.md5(combined.encode("utf-8")).hexdigest()[:8]
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{sanitized_url}_{sanitized_phrase}_{hash_digest}_{timestamp}.tar.gz"
    return filename

class ConcurrentAsyncCrawler:
    """
    An asynchronous BFS crawler.
    Each crawl run uses a unique crawl folder (named based on a sanitized root URL, phrase, and timestamp).
    An on_fetch_page callback is invoked for each fetched page.
    """
    def __init__(
        self,
        root_urls: List[str],
        base_crawl_dir: Path,
        on_fetch_page: Callable[[str, str, Path], None],
        target_links: int=10,
        max_depth: int=2,
        api_type: str="jina",
        phrase: Optional[str]=None,
        key_id: Optional[str]=None,
        private_key: Optional[str]=None,
        concurrency: int=5,
    ):
        self.root_urls = [self.normalize_url(u) for u in root_urls]
        self.target_links = target_links
        self.max_depth = max_depth
        self.api_type = api_type
        self.phrase = phrase if phrase else ""
        self.key_id = key_id
        self.private_key = private_key
        self.concurrency = concurrency
        self.on_fetch_page = on_fetch_page
        self.visited: Set[str] = set()
        self.collected: List[str] = []
        self.lock = asyncio.Lock()
        self.queue = asyncio.Queue()
        self.base_crawl_dir = base_crawl_dir
        # Create a unique crawl folder name based on the first root URL and phrase
        first_url = self.root_urls[0] if self.root_urls else "crawl"
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        folder_name = f"{sanitize_string(first_url)}_{sanitize_string(self.phrase)}_{timestamp}"
        self.crawl_folder = self.base_crawl_dir / folder_name
        self.crawl_folder.mkdir(parents=True, exist_ok=True)
        self.reverse_lookup_path = self.crawl_folder / "reverse_lookup.json"
        self._reverse_lookup: Dict[str,str] = {}

    def normalize_url(self, url: str) -> str:
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        parsed = urlparse(url)
        if not parsed.path:
            url += "/"
        return url

    async def run_crawl(self) -> List[str]:
        for u in self.root_urls:
            await self.queue.put((u, 1))
        workers = [asyncio.create_task(self.worker()) for _ in range(self.concurrency)]
        await self.queue.join()
        for w in workers:
            w.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
        self._save_reverse_lookup()
        return self.collected

    async def worker(self):
        while True:
            try:
                url, depth = await self.queue.get()
                await self.process_url(url, depth)
                self.queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[Worker] error: {e}")
                self.queue.task_done()

    async def process_url(self, url: str, depth: int):
        async with self.lock:
            if url in self.visited or len(self.collected) >= self.target_links:
                return
            self.visited.add(url)
        loop = asyncio.get_running_loop()
        try:
            content = await loop.run_in_executor(None, lambda: synchronous_fetch(url, self.api_type, self.key_id, self.private_key))
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            return
        async with self.lock:
            self.collected.append(url)
            import hashlib
            h = hashlib.sha256(url.encode()).hexdigest()
            self._update_reverse_lookup(h, url)
        if self.on_fetch_page:
            await loop.run_in_executor(None, lambda: self.on_fetch_page(url, content, self.crawl_folder))
        async with self.lock:
            if len(self.collected) >= self.target_links:
                return
        child_links = self.extract_links(content, url)
        for ln in child_links:
            async with self.lock:
                if ln not in self.visited and depth < self.max_depth and len(self.collected) < self.target_links:
                    await self.queue.put((ln, depth+1))

    def extract_links(self, content: str, base: str) -> List[str]:
        is_html = ("<html" in content[:500].lower()) or ("<body" in content[:500].lower())
        found = set()
        if is_html:
            matches = re.findall(r'href="([^"]+)"', content)
            for m in matches:
                link = urljoin(base, m)
                if self.is_valid_content_url(link):
                    found.add(link)
        else:
            matches = re.findall(r'https?://[^\s)]+', content)
            for m in matches:
                if self.is_valid_content_url(m):
                    found.add(m)
        return list(found)

    def is_valid_content_url(self, url: str) -> bool:
        path_low = urlparse(url).path.lower()
        for pattern in [r'/cdn-cgi/', r'/wp-json/', r'/wp-admin/', r'/wp-content/', r'/api/', r'/#',
                        r'/feed/', r'/xmlrpc\.php', r'/wp-includes/', r'/cdn/', r'/assets/', 
                        r'/static/', r'/email-protection', r'/ajax/', r'/rss/', r'/login', 
                        r'/signup', r'/register', r'/search']:
            if re.search(pattern, path_low):
                return False
        if re.search(r'\.(jpg|jpeg|png|gif|mp4|webp|svg|css|js|ico|woff|woff2|ttf|eot|pdf|zip|tar|gz|rar)$', path_low):
            return False
        return True

    def _update_reverse_lookup(self, key: str, url: str):
        self._reverse_lookup[key] = url

    def _save_reverse_lookup(self):
        try:
            with open(str(self.reverse_lookup_path), "w", encoding="utf-8") as f:
                json.dump(self._reverse_lookup, f, indent=2)
            logger.info(f"Reverse lookup saved to {self.reverse_lookup_path}")
        except Exception as e:
            logger.error(f"Error saving reverse lookup: {e}")

    def export_archive(self) -> str:
        """
        Create a tar.gz archive of the entire crawl folder.
        The filename is generated based on the first root URL and the query phrase.
        """
        if not self.crawl_folder.exists():
            raise ValueError("Crawl folder does not exist.")
        # Use the first root URL and the phrase to generate the filename.
        root = self.root_urls[0] if self.root_urls else "crawl"
        filename = generate_archive_filename(root, self.phrase)
        archive_path = self.crawl_folder.parent / filename
        with tarfile.open(str(archive_path), "w:gz") as tar:
            tar.add(str(self.crawl_folder), arcname=self.crawl_folder.name)
        logger.info(f"Exported crawl archive to {archive_path}")
        return str(archive_path)
