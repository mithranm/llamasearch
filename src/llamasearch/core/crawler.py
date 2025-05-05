# src/llamasearch/core/crawler.py

"""
crawler.py - Asynchronous crawler using a custom priority queue logic.
Includes pre-check for URL accessibility and rate limiting.
"""

import asyncio
import hashlib
import inspect
import json
import os
import re
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import unquote, urljoin, urlparse

import aiohttp  # Import aiohttp
from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CacheMode,
    CrawlerRunConfig,
    CrawlResult,
)
from crawl4ai.async_crawler_strategy import AsyncPlaywrightCrawlerStrategy
from crawl4ai.models import CrawlResultContainer

from llamasearch.utils import setup_logging

logger = setup_logging(__name__, use_qt_handler=True)

DEFAULT_RELEVANCE_KEYWORDS = [
    "documentation", "guide", "tutorial", "api", "reference", "manual",
    "developer", "usage", "examples", "concepts", "getting started",
]
DEFAULT_PAGE_TIMEOUT_MS = 30000
DEFAULT_FETCH_TIMEOUT_S = 45
DEFAULT_CRAWL_DELAY_S = 0.5 # Default delay between requests
DEFAULT_PRECHECK_TIMEOUT_S = 10 # Timeout for HEAD request

# Allowed content types for scraping (adjust as needed)
ALLOWED_CONTENT_TYPES = [
    "text/html", "application/xhtml+xml", "text/plain", "application/xml"
]

def sanitize_string(s: str, max_length: int = 40) -> str:
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
    shutdown_event: Optional[threading.Event] = None,
    timeout_seconds: float = DEFAULT_FETCH_TIMEOUT_S,
) -> Optional[CrawlResult]:
    # (Keep fetch_single implementation as is, it handles the playwright part)
    if shutdown_event and shutdown_event.is_set():
        logger.info(f"Shutdown before fetching {url}")
        return None
    try:
        raw = await asyncio.wait_for(
            crawler.arun(url=url, config=cfg), timeout=timeout_seconds
        )
        if shutdown_event and shutdown_event.is_set():
            logger.info(f"Shutdown after fetching {url}")
            return None
        logger.debug(f"Arun type for {url}: {type(raw)}")
        if isinstance(raw, CrawlResultContainer):
            if raw._results and isinstance(raw[0], CrawlResult):
                return raw[0]
            else:
                logger.warning(f"CrawlResultContainer empty/invalid for {url}.")
                return None
        elif isinstance(raw, CrawlResult):
            logger.warning(f"arun direct CrawlResult for {url}.")
            return raw
        elif inspect.isasyncgen(raw):
            logger.warning(f"arun async gen for {url}. Consuming first.")
            try:
                async for first in raw:
                    return first if isinstance(first, CrawlResult) else None
                    break # Ensure only the first item is consumed
            finally:
                if hasattr(raw, "aclose"):
                    await raw.aclose()
            return None
        else:
            logger.error(f"Unexpected arun type for {url}: {type(raw)}.")
            return None
    except asyncio.TimeoutError:
        logger.error(f"Timeout ({timeout_seconds}s) fetching {url} with Playwright")
        return None
    except Exception as e:
        if not (shutdown_event and shutdown_event.is_set()):
             # More specific logging for Playwright errors
            if "Target page, context or browser has been closed" in str(e):
                 logger.warning(f"Playwright page closed unexpectedly for {url}: {e}")
            elif "Timeout" in str(e) and "ms exceeded" in str(e):
                 logger.error(f"Playwright internal timeout fetching {url}: {e}")
            else:
                 logger.error(f"Playwright exception fetching {url}: {e}", exc_info=True)
        return None


class Crawl4AICrawler:
    """
    Asynchronous crawler prioritizing links based on keyword relevance.
    Includes URL accessibility pre-check and rate limiting.
    """

    def __init__(
        self,
        root_urls: List[str],
        base_crawl_dir: Path,
        target_links: int = 5,
        max_depth: int = 2,
        relevance_keywords: Optional[List[str]] = None,
        headless: bool = True,
        verbose_logging: bool = False,
        shutdown_event: Optional[threading.Event] = None,
        crawl_delay: float = DEFAULT_CRAWL_DELAY_S, # Add crawl delay param
        user_agent: Optional[str] = None, # Add user_agent parameter
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
        logger.info(f"Relevance keywords: {self.relevance_keywords}")
        self.raw_markdown_dir = self.base_crawl_dir / "raw"
        self.reverse_lookup_path = self.base_crawl_dir / "reverse_lookup.json"
        self.raw_markdown_dir.mkdir(parents=True, exist_ok=True)
        self._reverse_lookup: Dict[str, str] = {}
        self._load_reverse_lookup()
        self._user_abort = False
        self._shutdown_event = shutdown_event
        self.crawl_delay = max(0.1, crawl_delay) # Ensure minimum delay
        self._http_session: Optional[aiohttp.ClientSession] = None # Session for HEAD requests
        self._crawler: Optional[AsyncWebCrawler] = None # Initialize _crawler

        logger.info(f"Init Crawl4AICrawler for: {', '.join(self.root_urls)}")
        logger.info(f"Output dir (MD): {self.raw_markdown_dir}")
        logger.info(f"Reverse lookup: {self.reverse_lookup_path}")
        logger.info(
            f"Target: {self.target_links}, Max Depth: {max_depth} => Max Level: {self.max_crawl_level}"
        )
        logger.info(f"Crawl delay between requests: {self.crawl_delay}s")

        # --- Playwright/Crawl4AI setup remains the same ---
        browser_config_args = {
            "browser_type": "chromium",
            "headless": headless,
            "verbose": verbose_logging,
        }
        if user_agent:
            browser_config_args["user_agent"] = user_agent

        self._browser_cfg = BrowserConfig(**browser_config_args)

        # Increase Playwright page load timeout slightly
        self._run_cfg = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            remove_overlay_elements=True,
            stream=False,
            page_timeout=DEFAULT_PAGE_TIMEOUT_MS + 10000, # Increased timeout
        )
        # Strategy needs to be created later, within the async context if possible,
        # or ensure Playwright resources are managed correctly. For simplicity,
        # create it here but be mindful of cleanup.
        try:
            self._strategy = AsyncPlaywrightCrawlerStrategy(config=self._browser_cfg)
        except Exception as strategy_err:
             logger.error(f"Failed to initialize Playwright strategy: {strategy_err}", exc_info=True)
             raise RuntimeError("Could not initialize Playwright crawler strategy.") from strategy_err


    async def _ensure_session(self):
        """Creates an aiohttp ClientSession if one doesn't exist."""
        if self._http_session is None or self._http_session.closed:
            # Set a reasonable total timeout for the session
            timeout = aiohttp.ClientTimeout(total=DEFAULT_PRECHECK_TIMEOUT_S + 5)
            # Add basic headers to mimic a browser
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
            }
            try:
                self._http_session = aiohttp.ClientSession(timeout=timeout, headers=headers)
                logger.debug("aiohttp session created.")
            except Exception as e:
                 logger.error(f"Failed to create aiohttp session: {e}", exc_info=True)
                 self._http_session = None # Ensure it's None on failure

    async def _close_session(self):
        """Closes the aiohttp ClientSession if it exists."""
        if self._http_session and not self._http_session.closed:
            await self._http_session.close()
            logger.debug("aiohttp session closed.")
            self._http_session = None
        # Allow time for connections to close gracefully
        await asyncio.sleep(0.1)

    async def close(self):
        """Closes the crawler's resources like aiohttp session and Playwright browser."""
        logger.debug("Closing Crawl4AICrawler resources...")
        await self._close_session() # Close aiohttp session

        # Close the underlying AsyncWebCrawler's browser if it exists
        if hasattr(self, '_crawler') and self._crawler:
            try:
                await self._crawler.close()
                logger.debug("Closed underlying AsyncWebCrawler resources (Playwright).")
            except Exception as e:
                logger.error(f"Error closing underlying AsyncWebCrawler: {e}")
        else:
            logger.debug("No underlying AsyncWebCrawler instance found to close.")


    async def _check_url_accessibility(self, url: str) -> bool:
        """Performs a HEAD request to check status code and content type."""
        await self._ensure_session()
        if not self._http_session:
             logger.error(f"Cannot check accessibility for {url}: aiohttp session not available.")
             return False # Cannot proceed without session

        try:
            async with self._http_session.head(url, allow_redirects=True, timeout=aiohttp.ClientTimeout(total=DEFAULT_PRECHECK_TIMEOUT_S)) as response:
                status = response.status
                content_type = response.headers.get("Content-Type", "").lower().split(";")[0].strip()

                if status == 200:
                    if not content_type or any(allowed in content_type for allowed in ALLOWED_CONTENT_TYPES):
                        logger.debug(f"URL accessible and content type OK ({status}, '{content_type}'): {url}")
                        return True
                    else:
                        logger.info(f"Skipping URL due to disallowed content type '{content_type}': {url}")
                        return False
                else:
                    logger.info(f"Skipping URL due to non-200 status code ({status}): {url}")
                    return False
        except asyncio.TimeoutError:
            logger.warning(f"Accessibility check timed out ({DEFAULT_PRECHECK_TIMEOUT_S}s) for: {url}")
            return False
        except aiohttp.ClientError as e:
             # Log common client errors without full stack trace unless debugging
             if isinstance(e, (aiohttp.ClientConnectorError, aiohttp.ClientSSLError)):
                  logger.warning(f"Accessibility check connection/SSL error for {url}: {e}")
             else:
                  logger.warning(f"Accessibility check failed for {url}: {e}")
             return False
        except Exception as e:
            logger.error(f"Unexpected error during accessibility check for {url}: {e}")
            return False

    def normalize_url(self, url: str) -> str:
        # (Keep normalize_url implementation as is)
        if not isinstance(url, str):
            return ""
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        url, *_ = url.split("#", 1)
        parsed = urlparse(url)
        path = parsed.path if (parsed.path and parsed.path != "/") else "/"
        # Handle cases where netloc might be missing if urljoin was imperfect
        if not parsed.scheme or not parsed.netloc:
            logger.debug(f"Could not properly parse URL for normalization: {url}")
            return ""
        normalized = f"{parsed.scheme}://{parsed.netloc}{path}"
        if parsed.query:
            normalized += f"?{parsed.query}"
        # Ensure trailing slash for root paths
        if (not parsed.path or parsed.path == "/") and not normalized.endswith("/"):
            normalized += "/"
        return normalized


    def _generate_key(self, url: str) -> str:
        # (Keep _generate_key implementation as is)
        return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]

    def _load_reverse_lookup(self):
        # (Keep _load_reverse_lookup implementation as is)
        if self.reverse_lookup_path.exists():
            try:
                with open(self.reverse_lookup_path, "r", encoding="utf-8") as f:
                    self._reverse_lookup = json.load(f)
                logger.info(
                    f"Loaded global reverse lookup ({len(self._reverse_lookup)} entries)."
                )
            except Exception as e:
                logger.error(f"Error loading reverse lookup: {e}. Starting fresh.")
                self._reverse_lookup = {}
        else:
            logger.info("No existing global reverse lookup found.")
            self._reverse_lookup = {}

    def _save_reverse_lookup(self):
        # (Keep _save_reverse_lookup implementation as is)
        temp_path = None
        try:
            temp_path = self.reverse_lookup_path.with_suffix(".json.tmp")
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(self._reverse_lookup, f, indent=2, ensure_ascii=False)
            os.replace(temp_path, self.reverse_lookup_path)
            logger.info(
                f"Global reverse lookup saved ({len(self._reverse_lookup)} entries)."
            )
        except Exception as e:
            logger.error(f"Error saving reverse lookup: {e}")
            if temp_path and temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass


    def _score_link_priority(self, link_url: str) -> float:
        # (Keep _score_link_priority implementation as is)
        score = 0.0
        link_lower = link_url.lower()
        try:
            parsed = urlparse(link_lower)
            # Include path and query for keyword scoring
            path_query = (parsed.path or "") + ("?" + parsed.query if parsed.query else "")
            for keyword in self.relevance_keywords:
                try:
                    # Use word boundaries for more specific matches
                    score += len(
                        re.findall(r"\b" + re.escape(keyword) + r"\b", path_query)
                    )
                except re.error:
                    # Fallback for keywords that might cause regex errors
                    score += path_query.count(keyword)
        except Exception as e:
            logger.warning(f"Could not parse/score link {link_url}: {e}")
            # Return high value (low priority) if scoring fails
            return float("inf")
        # Invert score for priority queue (lower value = higher priority)
        # Add small epsilon to avoid division by zero, ensure priority > 0
        return max(1.0 / (score + 0.01), 1e-6)


    def abort(self):
        # (Keep abort implementation as is)
        self._user_abort = True
        logger.info("User abort requested for crawler.")


    async def run_crawl(self) -> List[str]:
        logger.info(f"Starting crawl. Output Dir: {self.raw_markdown_dir}")
        collected_urls: List[str] = []
        failed_urls: List[str] = []
        queue: asyncio.PriorityQueue[Tuple[float, int, str]] = asyncio.PriorityQueue()
        visited: Set[str] = set()
        last_progress_time = time.time()
        STALL_TIMEOUT = 90
        GLOBAL_TIMEOUT = 1800 # 30 minutes
        start_time = time.time()

        # Initialize session and Playwright crawler within the async context
        await self._ensure_session()
        if not self._http_session:
             logger.error("Failed to initialize HTTP session. Cannot start crawl.")
             return []

        # --- Safely initialize crawler ---
        crawler = None
        try:
            # Assuming _strategy is initialized in __init__ now
            if not hasattr(self, '_strategy') or self._strategy is None:
                 raise RuntimeError("Crawler strategy not initialized.")
            crawler = AsyncWebCrawler(crawler_strategy=self._strategy)
            logger.debug("AsyncWebCrawler initialized.")
        except Exception as crawler_init_err:
            logger.error(f"Failed to initialize AsyncWebCrawler: {crawler_init_err}", exc_info=True)
            await self._close_session()
            return []
        # --- End safe initialization ---


        for url in self.root_urls:
            norm_url = self.normalize_url(url)
            if norm_url and norm_url not in visited:
                # Add initial URLs with level 1
                await queue.put((self._score_link_priority(norm_url), 1, norm_url))
                visited.add(norm_url)

        collected_count = 0

        async def crawl_loop():
            nonlocal last_progress_time, collected_count
            while not queue.empty() and collected_count < self.target_links:
                if self._user_abort or (
                    self._shutdown_event and self._shutdown_event.is_set()
                ):
                    logger.warning("Crawl loop aborted.")
                    break

                # --- Add Delay ---
                await asyncio.sleep(self.crawl_delay)

                logger.debug(
                    f"--- DIAG: Q:{queue.qsize()}, C:{collected_count}, V:{len(visited)} ---"
                )
                priority, level, url = await queue.get()
                logger.debug(
                    f"--- DIAG: Dequeued {url} (Pri:{priority:.3f}, Lvl:{level}) ---"
                )
                try:
                    if level > self.max_crawl_level:
                        logger.debug(
                            f"Max level {self.max_crawl_level} reached for {url}. Skip."
                        )
                        continue

                    # --- Pre-check URL ---
                    # if not await self._check_url_accessibility(url):
                    #     logger.debug(f"Skipping inaccessible/disallowed URL: {url}")
                    #     failed_urls.append(f"{url} (inaccessible)")
                    #     continue
                    # --- End Pre-check ---

                    logger.info(
                        f"Crawling [L:{level}/{self.max_crawl_level} | C:{collected_count}/{self.target_links} | Q:{queue.qsize()}]: {url}"
                    )
                    # Call the playwright fetcher
                    result = await fetch_single(
                        crawler, url, self._run_cfg, self._shutdown_event
                    )

                    if result is None:
                        if not (self._shutdown_event and self._shutdown_event.is_set()):
                            logger.warning(f"Playwright fetch failed/aborted for {url}.")
                        failed_urls.append(f"{url} (fetch_error)")
                        continue

                    # Process successful result
                    if (
                        getattr(result, "success", True) # Assume success if attr missing? Default yes.
                        and getattr(result, "markdown", None) is not None
                        and isinstance(result.markdown, str) # Ensure markdown is string
                        and len(result.markdown.strip()) > 10 # Basic check for non-trivial content
                    ):
                        key = self._generate_key(url)
                        md_path = self.raw_markdown_dir / f"{key}.md"
                        try:
                            md_content = result.markdown
                            md_path.write_text(md_content, encoding="utf-8")
                            self._reverse_lookup[key] = url
                            if url not in collected_urls: # Count unique URLs saved
                                collected_urls.append(url)
                                collected_count += 1
                                last_progress_time = time.time() # Update progress time
                            logger.debug(
                                f"Saved MD: {md_path.name} ({len(md_content)} chars)"
                            )
                        except OSError as e:
                            logger.error(f"Failed write MD {md_path}: {e}")
                            failed_urls.append(f"{url} (write_error)")
                            continue

                        # Process links for next level
                        next_level = level + 1
                        if next_level <= self.max_crawl_level and getattr(
                            result, "links", None
                        ):
                            links_added = 0
                            internal_links = (
                                result.links.get("internal", [])
                                if isinstance(result.links, dict)
                                else []
                            )
                            for link_item in internal_links:
                                href = (
                                    link_item.get("href")
                                    if isinstance(link_item, dict)
                                    else (
                                        link_item
                                        if isinstance(link_item, str)
                                        else None
                                    )
                                )
                                if not href:
                                    continue
                                try:
                                    abs_link = urljoin(url, href)
                                    norm_link = self.normalize_url(abs_link)
                                    if not norm_link:
                                        continue
                                    # Check validity and if not visited before adding
                                    if (
                                        self.is_valid_content_url(norm_link)
                                        and norm_link not in visited
                                    ):
                                        visited.add(norm_link)
                                        priority_score = self._score_link_priority(norm_link)
                                        await queue.put(
                                            (priority_score, next_level, norm_link)
                                        )
                                        links_added += 1
                                except Exception as link_err:
                                    logger.debug(
                                        f"Err processing link '{href}' from {url}: {link_err}"
                                    )
                            if links_added > 0:
                                logger.debug(
                                    f"Added {links_added} links from {url} (Lvl:{next_level})."
                                )
                    else:
                        # Log reason for skipping (no MD, success=False, or too short)
                        reason = "success=False" if not getattr(result, "success", True) else \
                                 "no markdown" if getattr(result, "markdown", None) is None else \
                                 "markdown not string" if not isinstance(getattr(result, "markdown", ""), str) else \
                                 "markdown too short" if len(getattr(result, "markdown", "").strip()) <= 10 else \
                                 "unknown reason"
                        if not (self._shutdown_event and self._shutdown_event.is_set()):
                             logger.warning(f"Skipping result for {url} (Reason: {reason})")
                        failed_urls.append(f"{url} ({reason})")

                except asyncio.CancelledError:
                    logger.info("Crawl task cancelled.")
                    break
                except Exception as e:
                    if not (self._shutdown_event and self._shutdown_event.is_set()):
                        logger.error(f"Error processing {url}: {e}", exc_info=True)
                    failed_urls.append(f"{url} (processing_error)")
                finally:
                    queue.task_done() # Mark task as done regardless of outcome

                # Stall check
                if time.time() - last_progress_time > STALL_TIMEOUT:
                    logger.warning(
                        f"Crawl stalled? No progress {STALL_TIMEOUT}s. Q:{queue.qsize()}."
                    )
                    # Reset timer to avoid repeated warnings immediately
                    last_progress_time = time.time()

            # Log exit reason more clearly
            if self._user_abort or (
                self._shutdown_event and self._shutdown_event.is_set()
            ):
                logger.info("--- DIAG: Crawl loop exit: abort/shutdown.")
            elif queue.empty():
                logger.info("--- DIAG: Crawl loop exit: queue empty.")
            elif collected_count >= self.target_links:
                logger.info("--- DIAG: Crawl loop exit: target reached.")
            else:
                # This case might happen if all remaining queue items fail pre-check
                logger.warning(
                    f"--- DIAG: Crawl loop exit: unknown or all remaining URLs inaccessible. Q:{queue.qsize()}, C:{collected_count}"
                )

        try:
            # Wrap the crawl loop with a global timeout
            await asyncio.wait_for(crawl_loop(), timeout=GLOBAL_TIMEOUT)
        except asyncio.TimeoutError:
            logger.error(f"Global crawl timeout ({GLOBAL_TIMEOUT}s) reached.")
            self.abort() # Signal abort if timeout occurs
        except asyncio.CancelledError:
            logger.warning("Crawl main task cancelled externally.")
        except Exception as e:
            logger.error(f"Unexpected error in crawl run: {e}", exc_info=True)
            failed_urls.append(f"GENERAL_ERROR:{e}")
        finally:
            # --- Cleanup ---
            duration = time.time() - start_time
            # Attempt Playwright cleanup if crawler was initialized
            if crawler and hasattr(crawler, 'shutdown'):
                 try:
                      logger.debug("Shutting down Playwright crawler...")
                      # crawl4ai might not have an explicit shutdown, rely on GC/process end
                      # If it *does* have a shutdown/close, call it here.
                      # Example: await crawler.shutdown()
                      pass # No explicit shutdown in crawl4ai v0.6.2
                 except Exception as pe:
                      logger.warning(f"Error during Playwright cleanup: {pe}")

            # Close the aiohttp session
            await self._close_session()

            if failed_urls:
                logger.warning(
                    f"Crawl issues for {len(failed_urls)} URLs (max 10 shown): {failed_urls[:10]}"
                )
            self._save_reverse_lookup()
            logger.info(
                f"Crawl finished in {duration:.2f}s. Collected: {len(collected_urls)}, Visited: {len(visited)}, Q Left: {queue.qsize()}"
            )

        # Return URLs corresponding to *actually saved* markdown files
        return list(self._reverse_lookup.values())


    def is_valid_content_url(self, url: str) -> bool:
        # (Keep is_valid_content_url implementation as is)
        if not isinstance(url, str) or not url:
            return False
        try:
            parsed = urlparse(url)
            if parsed.scheme not in ["http", "https"]:
                return False

            # Enhanced ignore patterns
            ignore_patterns = [
                r"/__(.*?)__/",               # Double underscore paths (e.g., /__pycache__/)
                r"/(cdn|static|assets|media|images|img|css|js|fonts|font)/", # Common asset paths
                r"/node_modules/",
                r"/\.git/",
                r"/\.well-known/",
                r"(/|\.)(login|signin|signup|register|auth|account|profile|user|admin|dashboard)($|/|\?|#)", # Auth/account pages
                r"/search(\?|$|/)",            # Search result pages
                r"/(cart|checkout|order|wishlist)($|/|\?|#)", # E-commerce actions
                r"/(tag|tags|category|categories|author|authors)/", # Taxonomy/author archives
                r"\?replytocom=",            # Comment reply links
                r"#",                         # Fragment identifiers usually point within page
                # Protocol handlers
                r"^(mailto|tel|javascript|ftp|irc):",
                # API/Feed endpoints
                r"/(api|rest|rpc|service|feed|rss|atom|xmlrpc)(\.php)?($|/|\?|#)",
                # Common CMS/Framework paths
                r"/(wp-content|wp-admin|wp-includes|wp-json)/",
                r"/(_next|_nuxt|_svelte)/",   # JS framework internals
                r"/(cgi-bin|plesk-stat|webmail)/",
                r"/ajax/",
            ]
            url_lower = url.lower()
            if any(re.search(p, url_lower) for p in ignore_patterns):
                logger.debug(f"Ignoring URL due to pattern match: {url}")
                return False

            # Ignore common file extensions
            ignore_extensions = r"\.(jpg|jpeg|png|gif|bmp|tif|tiff|ico|webp|svg|)" + \
                                r"\.(css|js|json|xml|)" + \
                                r"\.(mp3|mp4|avi|mov|wmv|flv|)" + \
                                r"\.(woff|woff2|ttf|eot|otf|)" + \
                                r"\.(pdf|zip|tar|gz|rar|7z|exe|dmg|iso|bin|)" + \
                                r"\.(doc|docx|xls|xlsx|ppt|pptx|csv|rtf|)" + \
                                r"\.(txt|log|ini|cfg|sql|db|bak)$" # Added more text/config extensions

            path_part = parsed.path.split("?")[0].lower()
            if re.search(ignore_extensions, path_part):
                 logger.debug(f"Ignoring URL due to extension match: {url}")
                 return False

            # Domain check (ensure it's within the scope of root URLs)
            root_domains = {
                urlparse(root).netloc.lower() # Lowercase for comparison
                for root in self.root_urls
                if urlparse(root).netloc
            }
            current_domain = parsed.netloc.lower() # Lowercase for comparison
            if not current_domain:
                return False # Invalid URL if no domain

            # Allow exact match or subdomain match
            if current_domain not in root_domains:
                if not any(
                    rd and (current_domain == rd or current_domain.endswith("." + rd))
                    for rd in root_domains
                ):
                    logger.debug(f"Ignoring URL due to domain mismatch ({current_domain} vs {root_domains}): {url}")
                    return False

            return True # Passed all checks
        except Exception as e:
            logger.debug(f"URL validation error '{url}': {e}")
            return False