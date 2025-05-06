# src/llamasearch/core/crawler.py

"""
crawler.py - Asynchronous crawler using crawl4ai, with custom priority queue logic
based on keyword relevance in URLs and link text. Relies on crawl4ai/Playwright
for fetching and error handling.
"""

import asyncio
import hashlib
import inspect
import json
import logging
import os
import re
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import unquote, urljoin, urlparse

# Removed aiohttp import

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

# Use the global logger setup, potentially with Qt handler if running GUI
logger = setup_logging(__name__, use_qt_handler=True)

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
DEFAULT_PAGE_TIMEOUT_MS = 30000  # Timeout for Playwright page operations
DEFAULT_FETCH_TIMEOUT_S = 45  # Overall timeout for fetching a single URL
DEFAULT_CRAWL_DELAY_S = 0.5  # Default delay between requests

# Allowed content types for scraping (crawl4ai might handle this internally, but kept for reference)
ALLOWED_CONTENT_TYPES = [
    "text/html",
    "application/xhtml+xml",
    "text/plain",
    "application/xml",
]


def sanitize_string(s: str, max_length: int = 40) -> str:
    """Sanitizes a string to be used as part of a filename or directory name."""
    s = unquote(s)  # Decode URL encoding
    s = re.sub(r"^[a-zA-Z]+://", "", s)  # Remove scheme
    s = re.sub(r"https?://(www\.)?", "", s)  # Remove http/https/www
    s = re.sub(r'[/:\\?*"<>| ]+', "_", s)  # Replace unsafe characters with underscore
    s = re.sub(r"_+", "_", s)  # Collapse multiple underscores
    s = s.strip("_")  # Remove leading/trailing underscores
    if len(s) > max_length:
        s = s[:max_length]  # Truncate if too long
    return s if s else "default"  # Return "default" if string becomes empty


async def fetch_single(
    crawler: AsyncWebCrawler,
    url: str,
    cfg: CrawlerRunConfig,
    shutdown_event: Optional[threading.Event] = None,
    timeout_seconds: float = DEFAULT_FETCH_TIMEOUT_S,
) -> Optional[CrawlResult]:
    """
    Fetches a single URL using the crawl4ai AsyncWebCrawler instance.
    Handles timeouts and shutdown signals.
    """
    if shutdown_event and shutdown_event.is_set():
        logger.info(f"Shutdown requested before fetching {url}")
        return None
    try:
        # Run the crawler's asynchronous run method with a timeout
        raw = await asyncio.wait_for(
            crawler.arun(url=url, config=cfg), timeout=timeout_seconds
        )
        # Check for shutdown again after the potentially long operation
        if shutdown_event and shutdown_event.is_set():
            logger.info(f"Shutdown requested after fetching {url}")
            return None

        logger.debug(f"Arun result type for {url}: {type(raw)}")

        # Process the result based on its type (crawl4ai can return different types)
        if isinstance(raw, CrawlResultContainer):
            # Standard result container
            if raw._results and isinstance(raw[0], CrawlResult):
                return raw[0]
            else:
                logger.warning(f"CrawlResultContainer empty or invalid for {url}.")
                return None
        elif isinstance(raw, CrawlResult):
            # Sometimes returns a single result directly
            logger.warning(f"arun returned direct CrawlResult for {url}.")
            return raw
        elif inspect.isasyncgen(raw):
            # Handles cases where it returns an async generator (stream mode?)
            logger.warning(
                f"arun returned async generator for {url}. Consuming first item."
            )
            try:
                async for first_item in raw:
                    # Return the first valid CrawlResult found
                    return first_item if isinstance(first_item, CrawlResult) else None
                    break  # Only consume the first item
            finally:
                # Ensure the generator is closed properly
                if hasattr(raw, "aclose"):
                    await raw.aclose()
            return None  # No valid item found in generator
        else:
            logger.error(f"Unexpected arun result type for {url}: {type(raw)}.")
            return None
    except asyncio.TimeoutError:
        logger.error(f"Timeout ({timeout_seconds}s) fetching {url} with Playwright")
        return None
    except Exception as e:
        # Log Playwright-specific errors or general errors
        if not (shutdown_event and shutdown_event.is_set()):
            if "Target page, context or browser has been closed" in str(e):
                logger.warning(f"Playwright page closed unexpectedly for {url}: {e}")
            elif "Timeout" in str(e) and "ms exceeded" in str(e):
                logger.error(f"Playwright internal timeout fetching {url}: {e}")
            else:
                logger.error(f"Playwright exception fetching {url}: {e}", exc_info=True)
        return None


class Crawl4AICrawler:
    """
    Asynchronous crawler using crawl4ai and Playwright, prioritizing links based
    on keyword relevance in both URL and anchor text.
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
        crawl_delay: float = DEFAULT_CRAWL_DELAY_S,
        user_agent: Optional[str] = None,
    ):
        if not root_urls:
            raise ValueError("At least one root URL must be provided.")
        self.root_urls = [self.normalize_url(u) for u in root_urls]
        self.target_links = target_links
        # Max crawl level is depth + 1 (root is level 1)
        self.max_crawl_level = max_depth + 1
        self.base_crawl_dir = base_crawl_dir
        self.relevance_keywords = (
            [kw.lower() for kw in relevance_keywords]
            if relevance_keywords
            else DEFAULT_RELEVANCE_KEYWORDS
        )
        logger.info(f"Using relevance keywords: {self.relevance_keywords}")

        # Setup directories and reverse lookup
        self.raw_markdown_dir = self.base_crawl_dir / "raw"
        self.reverse_lookup_path = self.base_crawl_dir / "reverse_lookup.json"
        self.raw_markdown_dir.mkdir(parents=True, exist_ok=True)
        self._reverse_lookup: Dict[str, str] = {}
        self._load_reverse_lookup()  # Load existing lookup data

        self._user_abort = False
        self._shutdown_event = shutdown_event
        self.crawl_delay = max(0.1, crawl_delay)  # Ensure minimum delay
        # Removed self._http_session initialization
        self._crawler: Optional[AsyncWebCrawler] = None  # Underlying crawl4ai instance

        logger.info(
            f"Initializing Crawl4AICrawler for root URLs: {', '.join(self.root_urls)}"
        )
        logger.info(f"Markdown output directory: {self.raw_markdown_dir}")
        logger.info(f"Reverse lookup file: {self.reverse_lookup_path}")
        logger.info(
            f"Target Links: {self.target_links}, Max Depth: {max_depth} => Max Level: {self.max_crawl_level}"
        )
        logger.info(f"Delay between requests: {self.crawl_delay}s")

        # Configure Playwright browser settings
        browser_config_args = {
            "browser_type": "chromium",
            "headless": headless,
            "verbose": verbose_logging,
        }
        if user_agent:
            browser_config_args["user_agent"] = user_agent
        self._browser_cfg = BrowserConfig(**browser_config_args)

        # Configure crawler run settings (timeouts, etc.)
        self._run_cfg = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,  # Don't cache pages
            remove_overlay_elements=True,  # Attempt to remove popups/overlays
            stream=False,  # Get result after page is fully processed
            page_timeout=DEFAULT_PAGE_TIMEOUT_MS
            + 10000,  # Increased timeout for page load
        )

        # Initialize the Playwright strategy (can raise errors)
        try:
            self._strategy = AsyncPlaywrightCrawlerStrategy(config=self._browser_cfg)
            logger.debug("AsyncPlaywrightCrawlerStrategy initialized.")
        except Exception as strategy_err:
            logger.error(
                f"Failed to initialize Playwright strategy: {strategy_err}",
                exc_info=True,
            )
            raise RuntimeError(
                "Could not initialize Playwright crawler strategy."
            ) from strategy_err

    # Removed _ensure_session method
    # Removed _close_session method

    async def close(self):
        """Closes the crawler's Playwright browser resources."""
        logger.info("Closing Crawl4AICrawler Playwright resources...")
        # Close the underlying AsyncWebCrawler's Playwright resources
        if self._crawler:
            try:
                await self._crawler.close()
                logger.info("Closed underlying AsyncWebCrawler resources (Playwright).")
            except Exception as e:
                logger.error(f"Error closing underlying AsyncWebCrawler: {e}")
            finally:
                self._crawler = None  # Clear reference
        else:
            logger.debug(
                "No underlying AsyncWebCrawler instance found or already closed."
            )

    # Removed _check_url_accessibility method

    def normalize_url(self, url: str) -> str:
        """
        Normalizes a URL: ensures scheme, removes fragment, standardizes path.
        Returns normalized URL or empty string on failure.
        """
        if not isinstance(url, str):
            return ""
        # Ensure scheme is present
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        # Remove fragment identifier
        url, *_ = url.split("#", 1)
        try:
            parsed = urlparse(url)
            # Standardize path (use '/' for root)
            path = parsed.path if (parsed.path and parsed.path != "/") else "/"
            # Reconstruct URL, handling potential missing parts
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
        except ValueError as e:  # Catch potential parsing errors
            logger.warning(f"URL normalization failed for '{url}': {e}")
            return ""

    def _generate_key(self, url: str) -> str:
        """Generates a short SHA256 hash key for a normalized URL."""
        return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]

    def _load_reverse_lookup(self):
        """Loads the URL reverse lookup map from its JSON file."""
        if self.reverse_lookup_path.exists():
            try:
                with open(self.reverse_lookup_path, "r", encoding="utf-8") as f:
                    self._reverse_lookup = json.load(f)
                logger.info(
                    f"Loaded URL reverse lookup ({len(self._reverse_lookup)} entries) from {self.reverse_lookup_path}."
                )
            except Exception as e:
                logger.error(
                    f"Error loading reverse lookup file: {e}. Starting fresh.",
                    exc_info=True,
                )
                self._reverse_lookup = {}
        else:
            logger.info(
                f"No existing reverse lookup file found at {self.reverse_lookup_path}."
            )
            self._reverse_lookup = {}

    def _save_reverse_lookup(self):
        """Saves the current reverse lookup map to its JSON file atomically."""
        temp_path = None
        try:
            # Ensure parent directory exists
            self.reverse_lookup_path.parent.mkdir(parents=True, exist_ok=True)
            # Write to temporary file first
            temp_path = self.reverse_lookup_path.with_suffix(".json.tmp")
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(self._reverse_lookup, f, indent=2, ensure_ascii=False)
            # Atomically replace the original file
            os.replace(temp_path, self.reverse_lookup_path)
            logger.info(
                f"Global reverse lookup saved ({len(self._reverse_lookup)} entries) to {self.reverse_lookup_path}."
            )
        except Exception as e:
            logger.error(f"Error saving reverse lookup file: {e}", exc_info=True)
            # Clean up temporary file if it exists on error
            if temp_path and temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass

    def _score_link_priority(
        self, link_url: str, link_text: Optional[str] = None
    ) -> float:
        """Scores link priority based on keywords found in URL path/query AND link text."""
        url_score = 0.0
        text_score = 0.0
        link_lower = link_url.lower()  # Lowercase URL for scoring

        # Score URL Path and Query
        try:
            parsed = urlparse(link_lower)
            # Combine path and query for URL keyword matching
            path_query = (parsed.path or "") + (
                "?" + parsed.query if parsed.query else ""
            )
            for keyword in self.relevance_keywords:
                # Use simple count for broader matching in URL segments
                url_score += path_query.count(keyword)
        except Exception as e:
            logger.warning(f"Could not parse/score URL {link_url}: {e}")
            # If URL parsing/scoring fails badly, give it very low priority
            return float("inf")

        # Score Link Text (if provided)
        if link_text:
            text_lower = link_text.lower()  # Lowercase link text for scoring
            for keyword in self.relevance_keywords:
                # Use simple count for broader matching in text
                text_score += text_lower.count(keyword)

        # Combine scores (additive weighting, could be adjusted)
        total_score = url_score + text_score

        # Log the components for debugging
        log_level = logging.DEBUG
        logger.log(
            log_level,
            f"Scoring '{link_url}' (Text: '{(link_text or '')[:30]}...'): "
            f"URL Score={url_score:.2f}, Text Score={text_score:.2f}, Total={total_score:.2f}",
        )

        # Invert score for priority queue (lower value = higher priority)
        total_score = max(0.0, total_score)
        priority = max(1.0 / (total_score + 0.01), 1e-6)  # Add epsilon, ensure > 0
        logger.log(log_level, f"  -> Priority: {priority:.4f}")
        return priority

    def abort(self):
        """Signals the crawler to stop processing."""
        self._user_abort = True
        logger.info("User abort requested for crawler.")

    async def run_crawl(self) -> List[str]:
        """
        Runs the main crawl loop.
        Initializes queue, manages visited set, fetches pages, saves markdown,
        prioritizes and queues new links based on keywords and depth.
        Returns a list of URLs for which markdown files were successfully saved.
        """
        logger.info(f"Starting crawl. Output Dir: {self.raw_markdown_dir}")
        # Initialize collections for tracking state
        collected_urls: List[str] = []
        failed_urls: List[str] = []
        # Priority queue stores (priority_score, level, url) - lower score = higher priority
        queue: asyncio.PriorityQueue[Tuple[float, int, str]] = asyncio.PriorityQueue()
        visited: Set[str] = set()
        last_progress_time = time.time()
        STALL_TIMEOUT = 90  # Seconds without progress before warning
        GLOBAL_TIMEOUT = 1800  # Max crawl duration (30 minutes)
        start_time = time.time()

        # Removed aiohttp session initialization

        # --- Safely initialize crawl4ai AsyncWebCrawler ---
        crawler = None
        try:
            # Ensure the strategy created in __init__ is available
            if not hasattr(self, "_strategy") or self._strategy is None:
                raise RuntimeError("Crawler strategy not initialized.")
            # Create the crawler instance using the strategy
            crawler = AsyncWebCrawler(crawler_strategy=self._strategy)
            self._crawler = crawler  # Store reference for cleanup
            logger.debug("AsyncWebCrawler initialized.")
        except Exception as crawler_init_err:
            logger.error(
                f"Failed to initialize AsyncWebCrawler: {crawler_init_err}",
                exc_info=True,
            )
            # Removed call to _close_session
            return []
        # --- End safe initialization ---

        # Seed the queue with initial root URLs
        for url in self.root_urls:
            norm_url = self.normalize_url(url)
            if norm_url and norm_url not in visited:
                # Add initial URLs with level 1 and default priority (scored based on URL only initially)
                initial_priority = self._score_link_priority(norm_url)
                await queue.put((initial_priority, 1, norm_url))
                visited.add(norm_url)
                logger.debug(
                    f"Added root URL to queue: {norm_url} (Pri: {initial_priority:.3f})"
                )

        collected_count = 0

        # --- Main Crawl Loop ---
        async def crawl_loop():
            nonlocal last_progress_time, collected_count
            while not queue.empty() and collected_count < self.target_links:
                # Check for shutdown or user abort signals
                if self._user_abort or (
                    self._shutdown_event and self._shutdown_event.is_set()
                ):
                    logger.warning("Crawl loop aborted by signal.")
                    break

                # --- Rate Limiting Delay ---
                await asyncio.sleep(self.crawl_delay)

                # Get the next highest priority URL from the queue
                priority, level, url = await queue.get()
                logger.debug(
                    f"--- Dequeued {url} (Pri:{priority:.3f}, Lvl:{level}) ---"
                )

                try:
                    # Check if max depth/level has been exceeded
                    if level > self.max_crawl_level:
                        logger.debug(
                            f"Max level {self.max_crawl_level} reached for {url}. Skipping."
                        )
                        continue

                    # --- Removed pre-check call ---

                    logger.info(
                        f"Crawling [L:{level}/{self.max_crawl_level} | C:{collected_count}/{self.target_links} | Q:{queue.qsize()}]: {url}"
                    )
                    # Fetch the page content using Playwright via fetch_single
                    result = await fetch_single(
                        crawler,
                        url,
                        self._run_cfg,
                        self._shutdown_event,
                        timeout_seconds=DEFAULT_FETCH_TIMEOUT_S,
                    )

                    # Handle fetch failure or shutdown during fetch
                    if result is None:
                        if not (self._shutdown_event and self._shutdown_event.is_set()):
                            logger.warning(
                                f"Playwright fetch failed or aborted for {url}."
                            )
                        failed_urls.append(f"{url} (fetch_error)")
                        continue

                    # Process successful crawl result
                    if (
                        getattr(result, "success", True)
                        and getattr(result, "markdown", None) is not None
                        and isinstance(result.markdown, str)
                        and len(result.markdown.strip()) > 10
                    ):
                        # Generate a unique key for the URL
                        key = self._generate_key(url)
                        md_path = self.raw_markdown_dir / f"{key}.md"
                        try:
                            # Write the extracted markdown content to the file
                            md_content = result.markdown
                            md_path.write_text(md_content, encoding="utf-8")
                            # Update the reverse lookup map and collected URLs list
                            self._reverse_lookup[key] = url
                            if url not in collected_urls:
                                collected_urls.append(url)
                                collected_count += 1
                                last_progress_time = time.time()  # Update progress time
                            logger.debug(
                                f"Saved MD: {md_path.name} ({len(md_content)} chars) for URL: {url}"
                            )
                        except OSError as e:
                            logger.error(f"Failed to write MD file {md_path}: {e}")
                            failed_urls.append(f"{url} (write_error)")
                            continue  # Skip link processing if write fails

                        # Process links found on the page for the next level
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
                                href: Optional[str] = None
                                link_text: Optional[str] = None  # Initialize link_text

                                if isinstance(link_item, dict):
                                    href = link_item.get("href")
                                    link_text = link_item.get("text")
                                elif isinstance(link_item, str):
                                    href = link_item

                                if not href:
                                    continue

                                try:
                                    abs_link = urljoin(url, href)
                                    norm_link = self.normalize_url(abs_link)
                                    if not norm_link:
                                        continue

                                    if (
                                        self.is_valid_content_url(norm_link)
                                        and norm_link not in visited
                                    ):
                                        visited.add(norm_link)
                                        priority_score = self._score_link_priority(
                                            norm_link, link_text
                                        )
                                        await queue.put(
                                            (priority_score, next_level, norm_link)
                                        )
                                        links_added += 1
                                except Exception as link_err:
                                    logger.debug(
                                        f"Error processing link '{href}' from {url}: {link_err}"
                                    )

                            if links_added > 0:
                                logger.debug(
                                    f"Added {links_added} prioritized links from {url} (Next Lvl:{next_level})."
                                )
                    else:
                        # Log reason for skipping the result
                        reason = (
                            "success=False"
                            if not getattr(result, "success", True)
                            else "no markdown"
                            if getattr(result, "markdown", None) is None
                            else "markdown not string"
                            if not isinstance(getattr(result, "markdown", ""), str)
                            else "markdown too short"
                            if len(getattr(result, "markdown", "").strip()) <= 10
                            else "unknown reason"
                        )
                        if not (self._shutdown_event and self._shutdown_event.is_set()):
                            logger.warning(
                                f"Skipping result for {url} (Reason: {reason})"
                            )
                        failed_urls.append(f"{url} ({reason})")

                except asyncio.CancelledError:
                    logger.info(f"Crawl task for {url} cancelled.")
                    break  # Exit loop if task is cancelled
                except Exception as e:
                    # Log unexpected errors during processing of a single URL
                    if not (self._shutdown_event and self._shutdown_event.is_set()):
                        logger.error(f"Error processing {url}: {e}", exc_info=True)
                    failed_urls.append(f"{url} (processing_error)")
                finally:
                    # Ensure task_done is called even if errors occur
                    queue.task_done()

                # Stall check: Log warning if no new pages collected for a while
                if time.time() - last_progress_time > STALL_TIMEOUT:
                    logger.warning(
                        f"Crawl may be stalled? No new pages collected for {STALL_TIMEOUT}s. Queue size: {queue.qsize()}."
                    )
                    last_progress_time = time.time()  # Reset timer

            # Log the reason for exiting the crawl loop
            if self._user_abort or (
                self._shutdown_event and self._shutdown_event.is_set()
            ):
                logger.info("--- Crawl loop exited due to abort/shutdown signal. ---")
            elif queue.empty():
                logger.info("--- Crawl loop exited because the queue is empty. ---")
            elif collected_count >= self.target_links:
                logger.info(
                    f"--- Crawl loop exited because target link count ({self.target_links}) was reached. ---"
                )
            else:
                logger.warning(
                    f"--- Crawl loop exited unexpectedly. Q:{queue.qsize()}, Collected:{collected_count}/{self.target_links} ---"
                )

        # --- End Crawl Loop Definition ---

        # --- Execute the crawl loop with timeout and cleanup ---
        try:
            await asyncio.wait_for(crawl_loop(), timeout=GLOBAL_TIMEOUT)
        except asyncio.TimeoutError:
            logger.error(f"Global crawl timeout ({GLOBAL_TIMEOUT}s) reached.")
            self.abort()  # Signal abort if timeout occurs
        except asyncio.CancelledError:
            logger.warning("Crawl main task cancelled externally.")
        except Exception as e:
            logger.error(f"Unexpected error during crawl run: {e}", exc_info=True)
            failed_urls.append(f"GENERAL_ERROR:{e}")
        finally:
            # Final cleanup sequence
            duration = time.time() - start_time
            # Close Playwright browser and resources
            await self.close()
            # Removed call to _close_session

            # Log summary of failed URLs (limited count)
            if failed_urls:
                logger.warning(
                    f"Crawl encountered issues for {len(failed_urls)} URLs (max 10 shown): {failed_urls[:10]}"
                )
            # Save the final state of the reverse lookup map
            self._save_reverse_lookup()
            logger.info(
                f"Crawl finished in {duration:.2f}s. Collected: {len(collected_urls)}, Visited: {len(visited)}, Queue Left: {queue.qsize()}"
            )
        # --- End Timeout and Cleanup ---

        # Return the list of URLs for which markdown files were successfully saved in this run
        return collected_urls if collected_urls is not None else []

    def is_valid_content_url(self, url: str) -> bool:
        """
        Checks if a URL is likely to contain crawlable content, filtering out
        common asset types, authentication pages, API endpoints, etc.
        Also checks if the domain matches or is a subdomain of the root URLs.
        """
        if not isinstance(url, str) or not url:
            return False
        try:
            parsed = urlparse(url)
            # Check for valid schemes
            if parsed.scheme not in ["http", "https"]:
                return False

            # Ignore common non-content paths and patterns
            ignore_patterns = [
                r"/__(.*?)__/",  # Double underscore paths
                r"/(cdn|static|assets|media|images|img|css|js|fonts|font)/",  # Assets
                r"/node_modules/",
                r"/\.git/",
                r"/\.well-known/",
                r"(/|\.)(login|signin|signup|register|auth|account|profile|user|admin|dashboard)($|/|\?|#)",  # Auth
                r"/search(\?|$|/)",  # Search results
                r"/(cart|checkout|order|wishlist)($|/|\?|#)",  # E-commerce
                r"/(tag|tags|category|categories|author|authors)/",  # Taxonomies
                r"\?replytocom=",  # Comments
                r"#",  # Fragments
                r"^(mailto|tel|javascript|ftp|irc):",  # Other protocols
                r"/(api|rest|rpc|service|feed|rss|atom|xmlrpc)(\.php)?($|/|\?|#)",  # APIs/Feeds
                r"/(wp-content|wp-admin|wp-includes|wp-json)/",  # WordPress
                r"/(_next|_nuxt|_svelte)/",  # JS frameworks
                r"/(cgi-bin|plesk-stat|webmail)/",  # Server utils
                r"/ajax/",
            ]
            url_lower = url.lower()
            if any(re.search(p, url_lower) for p in ignore_patterns):
                logger.log(logging.DEBUG, f"Ignoring URL due to pattern match: {url}")
                return False

            # Ignore common file extensions for non-crawlable assets
            ignore_extensions = (
                r"\.(jpg|jpeg|png|gif|bmp|tif|tiff|ico|webp|svg|)"
                + r"\.(css|js|json|xml|)"
                + r"\.(mp3|mp4|avi|mov|wmv|flv|mkv|webm|)"
                + r"\.(woff|woff2|ttf|eot|otf|)"
                + r"\.(pdf|zip|tar|gz|rar|7z|exe|dmg|iso|bin|msi|)"
                + r"\.(doc|docx|xls|xlsx|ppt|pptx|csv|rtf|odt|ods|odp|)"
                + r"\.(txt|log|ini|cfg|sql|db|bak|conf|yaml|yml|toml)$"
            )

            path_part = parsed.path.split("?")[0].lower()
            if re.search(ignore_extensions, path_part):
                logger.log(logging.DEBUG, f"Ignoring URL due to extension match: {url}")
                return False

            # Domain check: Ensure the URL is within the scope of the root URLs
            root_domains = {
                urlparse(root).netloc.lower()
                for root in self.root_urls
                if urlparse(root).netloc
            }
            current_domain = parsed.netloc.lower()
            if not current_domain:
                return False  # Invalid URL

            if current_domain not in root_domains:
                # Check if it's a subdomain of any root domain
                if not any(
                    rd and (current_domain == rd or current_domain.endswith("." + rd))
                    for rd in root_domains
                ):
                    logger.log(
                        logging.DEBUG,
                        f"Ignoring URL due to domain mismatch ({current_domain} vs {root_domains}): {url}",
                    )
                    return False

            # If all checks pass, the URL is considered valid
            return True
        except Exception as e:
            logger.debug(f"URL validation error for '{url}': {e}")
            return False
