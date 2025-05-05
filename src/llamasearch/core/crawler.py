#!/usr/bin/env python3
"""
crawler.py – Asynchronous crawler using a custom priority queue logic.

Crawls web content based on keyword relevance in scraped markdown and link URLs,
saving files to a central 'raw' directory and maintaining a global 'reverse_lookup.json'.
Includes timeouts and robust queue handling.
"""

import asyncio
import time
import re
import logging
import os
from urllib.parse import urlparse, urljoin, unquote
from pathlib import Path
import json
import hashlib
import inspect
from typing import List, Dict, Tuple, Optional, Set

# --- Crawl4AI Imports ---
from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CrawlerRunConfig,
    CacheMode,
    CrawlResult,
)
from crawl4ai.models import CrawlResultContainer
from crawl4ai.async_crawler_strategy import AsyncPlaywrightCrawlerStrategy

# Use standard logger, setup is handled by llamasearch.utils
logger = logging.getLogger(__name__)

# --- Default Keywords for Scoring ---
DEFAULT_RELEVANCE_KEYWORDS = [
    "documentation",
    "guide",
    "tutorial",
    "api",
    "reference",
    "manual",
    "developer",
    "usage",
    "examples",
    "concepts",
    "getting started",
]

# --- Constants ---
DEFAULT_PAGE_TIMEOUT_MS = 30000  # 30 seconds page load timeout
DEFAULT_FETCH_TIMEOUT_S = (
    45  # 45 seconds overall fetch timeout including potential redirects
)

# --- Helper Functions ---


def sanitize_string(s: str, max_length: int = 40) -> str:
    """Convert to a filesystem-friendly string."""
    s = unquote(s)
    s = re.sub(r"^[a-zA-Z]+://", "", s)
    s = re.sub(r"https?://(www\.)?", "", s)
    s = re.sub(r'[/:\\?*"<>| ]+', "_", s)
    s = re.sub(r"_+", "_", s)
    s = s.strip("_")
    if len(s) > max_length:
        s = s[:max_length]
    return s if s else "default"


async def fetch_single(
    crawler: AsyncWebCrawler,
    url: str,
    cfg: CrawlerRunConfig,
    timeout_seconds: float = DEFAULT_FETCH_TIMEOUT_S,
) -> Optional[CrawlResult]:
    """
    Helper to fetch a single URL with an overall timeout.
    Handles documented return types: CrawlResultContainer, AsyncGenerator.
    """
    try:
        # Apply overall timeout to the arun call
        raw = await asyncio.wait_for(
            crawler.arun(url=url, config=cfg), timeout=timeout_seconds
        )
        logger.debug(f"Type returned by arun for {url}: {type(raw)}")

        # 1. Container (stream=False - Expected in v0.6.x)
        if isinstance(raw, CrawlResultContainer):
            if raw._results and isinstance(raw[0], CrawlResult):
                logger.debug(
                    f"Handling CrawlResultContainer for {url}, extracting first result."
                )
                return raw[0]
            else:
                logger.warning(
                    f"CrawlResultContainer for {url} was empty or contained unexpected type."
                )
                return None

        # 2. Direct result (very rare in v0.6.x for stream=False)
        elif isinstance(raw, CrawlResult):
            logger.warning(
                f"arun directly returned CrawlResult for {url} (unexpected for stream=False)."
            )
            return raw

        # 3. Async generator (Expected for stream=True, which we aren't using)
        elif inspect.isasyncgen(raw):
            logger.warning(
                f"arun returned an async generator for {url} unexpectedly (stream=False). Consuming first item."
            )
            try:
                async for first_result in raw:
                    # Should only yield one in this scenario if it happens
                    if isinstance(first_result, CrawlResult):
                        return first_result
                    else:
                        logger.warning(
                            f"Generator for {url} yielded non-CrawlResult: {type(first_result)}"
                        )
                        return None
                logger.warning(f"Generator for {url} was empty.")
                return None
            except Exception as agen_ex:
                logger.error(
                    f"Error consuming unexpected generator for {url}: {agen_ex}",
                    exc_info=True,
                )
                return None
            finally:
                # Ensure generator is closed if it exists and has aclose
                if hasattr(raw, "aclose"):
                    await raw.aclose()

        # 4. Fallback / Unexpected
        else:
            logger.error(
                f"Unexpected return type from crawler.arun for {url}: {type(raw)}. Check Crawl4AI version/behavior."
            )
            return None

    except asyncio.TimeoutError:
        logger.error(
            f"Timeout ({timeout_seconds}s) occurred during fetch_single for {url}"
        )
        return None  # Treat timeout as failure
    except Exception as e:
        logger.error(f"Exception during fetch_single for {url}: {e}", exc_info=True)
        return None


class Crawl4AICrawler:
    """Asynchronous crawler prioritizing links based on keyword relevance."""

    def __init__(
        self,
        root_urls: List[str],
        base_crawl_dir: Path,
        target_links: int = 15,
        max_depth: int = 2,
        relevance_keywords: Optional[List[str]] = None,
        headless: bool = True,  # Option for debugging
        verbose_logging: bool = False,  # Option for debugging
    ):
        if not root_urls:
            raise ValueError("At least one root URL must be provided.")
        self.root_urls = [self.normalize_url(u) for u in root_urls]
        self.target_links = target_links
        self.max_crawl_level = max_depth + 1
        self.base_crawl_dir = base_crawl_dir
        self.relevance_keywords = (
            [kw.lower() for kw in relevance_keywords]
            if relevance_keywords
            else DEFAULT_RELEVANCE_KEYWORDS
        )
        logger.info(f"Using relevance keywords: {self.relevance_keywords}")

        self.raw_markdown_dir = self.base_crawl_dir / "raw"
        self.reverse_lookup_path = self.base_crawl_dir / "reverse_lookup.json"
        self.raw_markdown_dir.mkdir(parents=True, exist_ok=True)
        self._reverse_lookup: Dict[str, str] = {}
        self._load_reverse_lookup()

        logger.info(
            f"Initialized Crawl4AICrawler for root(s): {', '.join(self.root_urls)}"
        )
        logger.info(f"Output directory (markdown): {self.raw_markdown_dir}")
        logger.info(f"Reverse lookup file: {self.reverse_lookup_path}")
        logger.info(
            f"Target links: {self.target_links}, Max Depth (User): {max_depth} => Max Crawl Level: {self.max_crawl_level}"
        )

        # --- Browser/Strategy Config - incorporating debugging options ---
        self._browser_cfg = BrowserConfig(
            browser_type="chromium",
            headless=headless,  # Control headful/headless mode
            verbose=verbose_logging,  # Control verbose logs
            # Enable capture features for debugging hangs (if needed)
            # capture_console=True,
            # capture_network=True
        )
        self._strategy = AsyncPlaywrightCrawlerStrategy(config=self._browser_cfg)
        # --- Run Config - adding page timeout ---
        self._run_cfg = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            remove_overlay_elements=True,
            stream=False,
            page_timeout=DEFAULT_PAGE_TIMEOUT_MS,  # Add page load timeout
        )

    def normalize_url(self, url: str) -> str:
        """Ensure URL has schema, strip fragment, ensure trailing slash for root."""
        if not isinstance(url, str):
            logger.warning(
                f"Normalize received non-string URL: {type(url)}. Returning empty string."
            )
            return ""
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        url, *_ = url.split("#", 1)
        parsed = urlparse(url)
        path = parsed.path if (parsed.path and parsed.path != "/") else "/"
        normalized = f"{parsed.scheme}://{parsed.netloc}{path}"
        if parsed.query:
            normalized += f"?{parsed.query}"
        if (not parsed.path or parsed.path == "/") and not normalized.endswith("/"):
            normalized += "/"
        return normalized

    def _generate_key(self, url: str) -> str:
        """Generates a unique key (hash) for a URL."""
        return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]

    def _load_reverse_lookup(self):
        """Loads the global reverse lookup file if it exists."""
        if self.reverse_lookup_path.exists():
            try:
                with open(self.reverse_lookup_path, "r", encoding="utf-8") as f:
                    self._reverse_lookup = json.load(f)
                logger.info(
                    f"Loaded existing global reverse lookup ({len(self._reverse_lookup)} entries)."
                )
            except json.JSONDecodeError:
                logger.error(
                    f"Error decoding JSON from {self.reverse_lookup_path}. Starting fresh."
                )
                self._reverse_lookup = {}
            except Exception as e:
                logger.error(f"Error loading reverse lookup: {e}. Starting fresh.")
                self._reverse_lookup = {}
        else:
            logger.info("No existing global reverse lookup found.")
            self._reverse_lookup = {}

    def _save_reverse_lookup(self):
        """Saves the updated global hash -> URL mapping to JSON."""
        temp_path = None
        try:
            temp_path = self.reverse_lookup_path.with_suffix(".json.tmp")
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(self._reverse_lookup, f, indent=2, ensure_ascii=False)
            os.replace(temp_path, self.reverse_lookup_path)
            logger.info(f"Global reverse lookup saved to {self.reverse_lookup_path}")
        except Exception as e:
            logger.error(f"Error saving reverse lookup: {e}")
            if temp_path and temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass

    def _score_link_priority(self, link_url: str) -> float:
        """
        Calculates the priority score for adding a link URL to the queue.
        Scores based on keywords found in the URL path/query. Lower is better.
        """
        link_score_component = 0
        link_lower = link_url.lower()

        try:
            parsed_link = urlparse(link_lower)
            path_query = parsed_link.path + "?" + parsed_link.query
            for keyword in self.relevance_keywords:
                try:
                    link_score_component += len(
                        re.findall(r"\b" + re.escape(keyword) + r"\b", path_query)
                    )
                except re.error:
                    link_score_component += path_query.count(keyword)

        except Exception as e:
            logger.warning(f"Could not parse or score link URL {link_url}: {e}")
            return float("inf")

        link_priority_score = 1.0 / (link_score_component + 0.01)
        return max(link_priority_score, 1e-6)

    async def run_crawl(self) -> List[str]:
        """Performs the asynchronous crawl using a priority queue based on link URL relevance. Robust against hangs."""
        logger.info(
            f"Starting keyword-priority crawl. Output Dir: {self.raw_markdown_dir}"
        )
        collected_urls: List[str] = []
        failed_urls: List[str] = []
        queue: asyncio.PriorityQueue[Tuple[float, int, str]] = asyncio.PriorityQueue()
        visited: Set[str] = set()
        last_progress_time = time.time()
        STALL_TIMEOUT = 60  # seconds without progress before warning
        GLOBAL_TIMEOUT = 600  # max crawl duration in seconds (10 min)
        start_time = time.time()

        for url in self.root_urls:
            norm_url = self.normalize_url(url)
            if norm_url and norm_url not in visited:
                root_priority = self._score_link_priority(norm_url)
                await queue.put((root_priority, 1, norm_url))
                visited.add(norm_url)

        crawler = AsyncWebCrawler(crawler_strategy=self._strategy)
        collected_count = 0

        async def crawl_loop():
            nonlocal last_progress_time, collected_count
            while not queue.empty() and collected_count < self.target_links:
                logger.info(f"--- DIAG: Queue size: {queue.qsize()}, Collected: {collected_count}, Visited: {len(visited)} ---")
                priority, current_depth_level, current_url = await queue.get()
                logger.info(f"--- DIAG: Dequeued URL: {current_url} (Priority: {priority}, Depth: {current_depth_level}) ---")
                try:
                    logger.debug(
                        f"Processing Q (Pri:{priority:.3f}, D:{current_depth_level}): {current_url}"
                    )
                    if current_depth_level > self.max_crawl_level:
                        logger.debug(
                            f"Max crawl level ({self.max_crawl_level}) reached for {current_url} (level {current_depth_level}). Skipping fetch."
                        )
                        continue
                    logger.info(
                        f"Crawling [L:{current_depth_level}/{self.max_crawl_level} | C:{collected_count}/{self.target_links} | Q:{queue.qsize()}]: {current_url}"
                    )
                    result = await fetch_single(
                        crawler, current_url, self._run_cfg
                    )
                    if result is None or not getattr(result, 'success', True):
                        logger.warning(
                            f"Fetch failed or returned None for {current_url}. Skipping."
                        )
                        failed_urls.append(current_url)
                        continue
                    if getattr(result, 'markdown', None) is not None:
                        key = self._generate_key(current_url)
                        md_path = self.raw_markdown_dir / f"{key}.md"
                        try:
                            md_content = result.markdown
                            if not isinstance(md_content, str):
                                logger.error(f"Markdown content for {current_url} is not a string. Skipping.")
                                failed_urls.append(current_url)
                                continue
                            md_path.write_text(md_content, encoding="utf-8")
                            self._reverse_lookup[key] = current_url
                            if current_url not in collected_urls:
                                collected_urls.append(current_url)
                                collected_count += 1
                                last_progress_time = time.time()
                            logger.debug(
                                f"Saved MD: {md_path.name} ({len(md_content)} chars)"
                            )
                        except OSError as e:
                            logger.error(f"Failed write MD {md_path}: {e}")
                            failed_urls.append(current_url)
                            continue
                        next_depth_level = current_depth_level + 1
                        if next_depth_level <= self.max_crawl_level and getattr(result, 'links', None):
                            links_added_from_page = 0
                            internal_links_raw = (
                                result.links.get("internal", [])
                                if isinstance(result.links, dict)
                                else []
                            )
                            for link_item in internal_links_raw:
                                link_url_rel = None
                                if isinstance(link_item, dict):
                                    link_url_rel = link_item.get("href")
                                elif isinstance(link_item, str):
                                    link_url_rel = link_item
                                else:
                                    logger.warning(
                                        f"Skipping unexpected link item type: {type(link_item)} from {current_url}"
                                    )
                                    continue
                                if not link_url_rel or not isinstance(link_url_rel, str):
                                    logger.debug(
                                        f"Skipping link item with missing or invalid href: {link_item}"
                                    )
                                    continue
                                try:
                                    abs_link = urljoin(current_url, link_url_rel)
                                    norm_link = self.normalize_url(abs_link)
                                    if not norm_link:
                                        continue
                                    if (
                                        self.is_valid_content_url(norm_link)
                                        and norm_link not in visited
                                    ):
                                        visited.add(norm_link)
                                        link_priority = self._score_link_priority(norm_link)
                                        logger.info(f"--- DIAG: Enqueuing URL: {norm_link} (Priority: {link_priority}, Depth: {next_depth_level}) ---")
                                        await queue.put((link_priority, next_depth_level, norm_link))
                                        links_added_from_page += 1
                                except Exception as link_err:
                                    logger.debug(
                                        f"Err processing link '{link_url_rel}' from {current_url}: {link_err}"
                                    )
                            if links_added_from_page > 0:
                                logger.debug(
                                    f"Added {links_added_from_page} links from {current_url} to queue (depth {next_depth_level}) with calculated priorities."
                                )
                    else:
                        logger.warning(
                            f"No markdown content retrieved from {current_url}"
                        )
                        failed_urls.append(current_url)
                except asyncio.CancelledError:
                    logger.info("Crawl task cancelled.")
                    break
                except Exception as e:
                    logger.error(
                        f"Error processing item {current_url}: {e}", exc_info=True
                    )
                    failed_urls.append(current_url)
                finally:
                    queue.task_done()
                # Stall detection
                if time.time() - last_progress_time > STALL_TIMEOUT:
                    logger.warning(f"Crawl appears stalled: no progress for {STALL_TIMEOUT}s. Queue size: {queue.qsize()}.")
                    # Dump queue contents for debugging
                    try:
                        queue_contents = []
                        qsize = queue.qsize()
                        for _ in range(qsize):
                            item = await queue.get()
                            queue_contents.append(item)
                            await queue.put(item)  # put it back
                        logger.warning(f"--- DIAG: Current queue contents (up to {len(queue_contents)}): {[item[2] for item in queue_contents]}")
                    except Exception as diag_exc:
                        logger.error(f"Error dumping queue contents: {diag_exc}")
                    last_progress_time = time.time()  # reset to avoid repeated warnings

            # After loop exits, log why
            if queue.empty():
                logger.info("--- DIAG: Crawl loop exited because queue is empty.")
            if collected_count >= self.target_links:
                logger.info("--- DIAG: Crawl loop exited because collected_count reached target.")
            if not queue.empty() and collected_count < self.target_links:
                logger.warning(f"--- DIAG: Crawl loop exited with non-empty queue and collected_count < target. Queue size: {queue.qsize()}.")
                # Dump remaining queue contents
                try:
                    queue_contents = []
                    qsize = queue.qsize()
                    for _ in range(qsize):
                        item = await queue.get()
                        queue_contents.append(item)
                        await queue.put(item)
                    logger.warning(f"--- DIAG: Remaining queue contents (up to {len(queue_contents)}): {[item[2] for item in queue_contents]}")
                except Exception as diag_exc:
                    logger.error(f"Error dumping remaining queue contents: {diag_exc}")

        try:
            # Run crawl loop with global timeout
            logger.info("--- DIAG: Entering asyncio.wait_for(crawl_loop)... ---")
            await asyncio.wait_for(crawl_loop(), timeout=GLOBAL_TIMEOUT)
            logger.info("--- DIAG: Exited asyncio.wait_for(crawl_loop) successfully. ---")
        except asyncio.TimeoutError:
            logger.error(
                f"Global crawl timeout ({GLOBAL_TIMEOUT}s) reached. Terminating crawl early."
            )
        except Exception as e:
            logger.error(f"Unexpected error during crawl: {e}")
            failed_urls.append(f"GENERAL_ERROR: {e}")
        finally:
            logger.info("--- DIAG: Entering finally block of run_crawl... ---")
            end_time = time.time()
            duration = end_time - start_time
            # Log summary
            if failed_urls:
                logger.warning(f"Crawl skipped/failed for {len(failed_urls)} URLs:")
                for failure in failed_urls[:10]: # Log first 10
                    logger.warning(f"  - {failure}")

            logger.info("--- DIAG: Calling _save_reverse_lookup()... ---")
            self._save_reverse_lookup()
            logger.info("--- DIAG: Finished _save_reverse_lookup(). ---")

            logger.info(
                f"Crawl finished in {duration:.2f}s. "
                f"Collected: {len(collected_urls)}, "
                f"Visited: {len(visited)}, Queue Left: {queue.qsize() if 'queue' in locals() else 'N/A'}"
            )

            logger.info("--- DIAG: Preparing to return from run_crawl. ---")

        logger.info("--- DIAG: End of run_crawl function. Returning results. ---")
        return list(self._reverse_lookup.keys())

    def is_valid_content_url(self, url: str) -> bool:
        """Basic filtering for URLs likely not containing primary content."""
        if not isinstance(url, str) or not url:
            return False
        try:
            parsed = urlparse(url)
            if parsed.scheme not in ["http", "https"]:
                return False
            path_low = parsed.path.lower()
            ignore_patterns = [
                r"/cdn-cgi/",
                r"/__",
                r"/wp-json/",
                r"/wp-admin/",
                r"/wp-content/",
                r"/wp-includes/",
                r"/api/",
                r"/v[1-9]/api/",
                r"/rest/",
                r"/feed/",
                r"/rss/",
                r"/atom/",
                r"/xmlrpc\.php",
                r"/assets/",
                r"/static/",
                r"/media/",
                r"/images/",
                r"/css/",
                r"/js/",
                r"/fonts/",
                r"/email-protection",
                r"/cdn-cgi/l/email-protection",
                r"/ajax/",
                r"/json/",
                r"/login",
                r"/signup",
                r"/register",
                r"/auth/",
                r"/account",
                r"/user",
                r"/profile",
                r"/admin",
                r"/search",
                r"/find",
                r"/cart",
                r"/checkout",
                r"/order",
                r"/tag/",
                r"/category/",
                r"/author/",
                r"tel:",
                r"mailto:",
                r"javascript:",
            ]
            if any(re.search(p, path_low) for p in ignore_patterns):
                return False
            path_part_for_ext = parsed.path.split("?")[0].lower()
            ignore_extensions = r"\.(jpg|jpeg|png|gif|bmp|tif|tiff|mp3|mp4|avi|mov|wmv|webp|svg|css|js|json|xml|ico|woff|woff2|ttf|eot|otf|pdf|zip|tar|gz|rar|7z|exe|dmg|iso|ppt|pptx|doc|docx|xls|xlsx|csv|txt|rtf)$"
            if re.search(ignore_extensions, path_part_for_ext):
                return False

            root_domains = {
                urlparse(root).netloc
                for root in self.root_urls
                if urlparse(root).netloc
            }
            current_domain = parsed.netloc
            if not current_domain:
                return False
            if current_domain not in root_domains:
                is_subdomain_or_match = False
                for root_domain in root_domains:
                    if root_domain and (
                        current_domain == root_domain
                        or current_domain.endswith("." + root_domain)
                    ):
                        is_subdomain_or_match = True
                        break
                if not is_subdomain_or_match:
                    return False
            return True
        except Exception as e:
            logger.debug(f"Error validating URL '{url}': {e}")
            return False
