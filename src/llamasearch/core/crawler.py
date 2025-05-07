# src/llamasearch/core/crawler.py

import asyncio
import hashlib
import inspect
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import unquote, urljoin, urlparse, urlunparse

from crawl4ai import (AsyncWebCrawler, BrowserConfig, CacheMode,
                      CrawlerRunConfig, CrawlResult)
from crawl4ai.async_crawler_strategy import AsyncPlaywrightCrawlerStrategy
from crawl4ai.models import CrawlResultContainer

from llamasearch.utils import setup_logging

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
DEFAULT_PAGE_TIMEOUT_MS = 30000
DEFAULT_FETCH_TIMEOUT_S = 45
DEFAULT_CRAWL_DELAY_S = 0.5
ALLOWED_CONTENT_TYPES = [
    "text/html",
    "application/xhtml+xml",
    "text/plain",
    "application/xml",
]


def sanitize_string(s: str, max_length: int = 40) -> str:
    """Sanitizes a string to be used as part of a filename or directory name."""
    s = unquote(s)
    scheme_match = re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", s)
    if scheme_match:
        s = s[len(scheme_match.group(0)) :]
    if s.lower().startswith("www."):
        s = s[4:]
    s = re.sub(r'[/:\\?*"<>|=]+', "_", s)
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"_+", "_", s)
    s = s.strip("_")  # Strip leading/trailing underscores
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
    """
    Fetches a single URL using the crawl4ai AsyncWebCrawler instance.
    Handles timeouts and shutdown signals.
    """
    if shutdown_event and shutdown_event.is_set():
        logger.info(f"Shutdown requested before fetching {url}")
        return None
    try:
        raw = await asyncio.wait_for(
            crawler.arun(url=url, config=cfg), timeout=timeout_seconds
        )
        if shutdown_event and shutdown_event.is_set():
            logger.info(f"Shutdown requested after fetching {url}")
            return None

        logger.debug(f"Arun result type for {url}: {type(raw)}")

        if isinstance(raw, CrawlResultContainer):
            if raw._results and isinstance(raw[0], CrawlResult):  # type: ignore
                return raw[0]  # type: ignore
            else:
                logger.warning(f"CrawlResultContainer empty or invalid for {url}.")
                return None
        elif isinstance(raw, CrawlResult):
            # This case was unexpected but seems to happen with single URL crawls sometimes
            logger.debug(f"arun returned direct CrawlResult for {url}.")
            return raw
        elif inspect.isasyncgen(raw):
            logger.warning(
                f"arun returned async generator for {url}. Consuming first item."
            )
            try:
                async for first_item in raw:
                    if isinstance(first_item, CrawlResult):
                        return first_item
                    break
            finally:
                if hasattr(raw, "aclose"):
                    await raw.aclose()
            return None
        else:
            logger.error(f"Unexpected arun result type for {url}: {type(raw)}.")
            return None
    except asyncio.TimeoutError:
        logger.error(f"Timeout ({timeout_seconds}s) fetching {url} with Playwright")
        return None
    except Exception as e:
        if not (shutdown_event and shutdown_event.is_set()):
            if "Target page, context or browser has been closed" in str(e):
                logger.warning(f"Playwright page closed unexpectedly for {url}: {e}")
            elif "Timeout" in str(e) and "ms exceeded" in str(e):
                logger.error(f"Playwright internal timeout fetching {url}: {e}")
            else:
                logger.error(f"Playwright exception fetching {url}: {e}", exc_info=True)
        return None


class Crawl4AICrawler:
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
        self.max_crawl_level = max_depth + 1
        self.base_crawl_dir = base_crawl_dir
        self.relevance_keywords = (
            [kw.lower() for kw in relevance_keywords]
            if relevance_keywords
            else DEFAULT_RELEVANCE_KEYWORDS
        )
        self.verbose_logging = verbose_logging
        logger.info(f"Using relevance keywords: {self.relevance_keywords}")

        self.raw_markdown_dir = self.base_crawl_dir / "raw"
        self.reverse_lookup_path = self.base_crawl_dir / "reverse_lookup.json"
        self.raw_markdown_dir.mkdir(parents=True, exist_ok=True)
        self._reverse_lookup: Dict[str, str] = {}
        self._load_reverse_lookup()

        self._user_abort = False
        self._shutdown_event = shutdown_event
        self.crawl_delay = max(0.1, crawl_delay)
        self._crawler: Optional[AsyncWebCrawler] = None

        logger.info(
            f"Initializing Crawl4AICrawler for root URLs: {', '.join(self.root_urls)}"
        )
        logger.info(f"Markdown output directory: {self.raw_markdown_dir}")
        logger.info(f"Reverse lookup file: {self.reverse_lookup_path}")
        logger.info(
            f"Target Links: {self.target_links}, Max Depth: {max_depth} => Max Level: {self.max_crawl_level}"
        )
        logger.info(f"Delay between requests: {self.crawl_delay}s")

        browser_config_args = {
            "browser_type": "chromium",
            "headless": headless,
            "verbose": self.verbose_logging,
        }
        if user_agent:
            browser_config_args["user_agent"] = user_agent
        self._browser_cfg = BrowserConfig(**browser_config_args)

        self._run_cfg = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            remove_overlay_elements=True,
            stream=False,
            page_timeout=DEFAULT_PAGE_TIMEOUT_MS + 10000,
        )

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

    async def close(self):
        logger.info("Closing Crawl4AICrawler Playwright resources...")
        if self._crawler:
            try:
                await self._crawler.close()
                logger.info("Closed underlying AsyncWebCrawler resources (Playwright).")
            except Exception as e:
                logger.error(f"Error closing underlying AsyncWebCrawler: {e}")
            finally:
                self._crawler = None
        else:
            logger.debug(
                "No underlying AsyncWebCrawler instance found or already closed."
            )

    def normalize_url(self, url: str) -> str:
        if not isinstance(url, str):
            return ""
        parsed_pre_check = urlparse(url)
        if parsed_pre_check.scheme and parsed_pre_check.scheme not in ["http", "https"]:
            logger.debug(
                f"Unsupported scheme '{parsed_pre_check.scheme}' in URL: {url}"
            )
            return ""
        if not parsed_pre_check.scheme:
            url = "https://" + url

        try:
            parsed = urlparse(url)
            path = re.sub(r"/+", "/", parsed.path)
            if not path:
                path = "/"

            query_params = parsed.query.split("&")
            query_params.sort()
            sorted_query = (
                "&".join(query_params) if query_params and query_params[0] else ""
            )

            normalized = urlunparse(
                parsed._replace(fragment="", path=path, query=sorted_query)
            )

            parsed_final = urlparse(normalized)
            if not parsed_final.query:
                path_segments = parsed_final.path.split("/")
                last_segment = path_segments[-1] if path_segments else ""
                has_file_extension = "." in last_segment and not last_segment.endswith(
                    "."
                )

                if (
                    parsed_final.path == "/" or not has_file_extension
                ) and not normalized.endswith("/"):
                    normalized += "/"
            return normalized
        except ValueError as e:
            logger.warning(f"URL normalization failed for '{url}': {e}")
            return ""

    def _generate_key(self, url: str) -> str:
        return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]

    def _load_reverse_lookup(self):
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
        temp_path = None
        try:
            self.reverse_lookup_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = self.reverse_lookup_path.with_suffix(".json.tmp")
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(self._reverse_lookup, f, indent=2, ensure_ascii=False)
            os.replace(temp_path, self.reverse_lookup_path)
            logger.info(
                f"Global reverse lookup saved ({len(self._reverse_lookup)} entries) to {self.reverse_lookup_path}."
            )
        except Exception as e:
            logger.error(f"Error saving reverse lookup file: {e}", exc_info=True)
            if temp_path and temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass

    def _score_link_priority(
        self, link_url: str, link_text: Optional[str] = None
    ) -> float:
        url_score = 0.0
        text_score = 0.0
        link_lower = link_url.lower()
        try:
            parsed = urlparse(link_lower)
            path_query = (parsed.path or "") + (
                "?" + parsed.query if parsed.query else ""
            )
            for keyword in self.relevance_keywords:
                url_score += path_query.count(keyword)
        except Exception as e:
            logger.warning(f"Could not parse/score URL {link_url}: {e}")
            return float("inf")
        if link_text:
            text_lower = link_text.lower()
            for keyword in self.relevance_keywords:
                text_score += text_lower.count(keyword)
        total_score = url_score + text_score

        # Use a simpler check for verbose logging
        if self.verbose_logging:
            logger.debug(
                f"Scoring '{link_url}' (Text: '{(link_text or '')[:30]}...'): "
                f"URL Score={url_score:.2f}, Text Score={text_score:.2f}, Total={total_score:.2f}"
            )
        total_score = max(0.0, total_score)
        priority = max(1.0 / (total_score + 0.01), 1e-6)
        if self.verbose_logging:
            logger.debug(f"  -> Priority: {priority:.4f}")
        return priority

    def abort(self):
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
        GLOBAL_TIMEOUT = 1800
        start_time = time.time()

        crawler_instance = None
        try:
            if not hasattr(self, "_strategy") or self._strategy is None:
                raise RuntimeError("Crawler strategy not initialized.")
            crawler_instance = AsyncWebCrawler(crawler_strategy=self._strategy)
            self._crawler = crawler_instance
            logger.debug("AsyncWebCrawler initialized.")
        except Exception as crawler_init_err:
            logger.error(
                f"Failed to initialize AsyncWebCrawler: {crawler_init_err}",
                exc_info=True,
            )
            return []

        for url in self.root_urls:
            norm_url = self.normalize_url(url)
            if norm_url and norm_url not in visited:
                initial_priority = self._score_link_priority(norm_url)
                await queue.put((initial_priority, 1, norm_url))
                visited.add(norm_url)
                logger.debug(
                    f"Added root URL to queue: {norm_url} (Pri: {initial_priority:.3f})"
                )
        collected_count = 0

        async def crawl_loop():
            nonlocal last_progress_time, collected_count
            while not queue.empty() and collected_count < self.target_links:
                if self._user_abort or (
                    self._shutdown_event and self._shutdown_event.is_set()
                ):
                    logger.warning("Crawl loop aborted by signal.")
                    break
                await asyncio.sleep(self.crawl_delay)
                priority, level, url = await queue.get()
                logger.debug(
                    f"--- Dequeued {url} (Pri:{priority:.3f}, Lvl:{level}) ---"
                )
                try:
                    if level > self.max_crawl_level:
                        logger.debug(
                            f"Max level {self.max_crawl_level} reached for {url}. Skipping."
                        )
                        continue
                    logger.info(
                        f"Crawling [L:{level}/{self.max_crawl_level} | C:{collected_count}/{self.target_links} | Q:{queue.qsize()}]: {url}"
                    )
                    result = await fetch_single(
                        crawler_instance,
                        url,
                        self._run_cfg,
                        self._shutdown_event,
                        timeout_seconds=DEFAULT_FETCH_TIMEOUT_S,
                    )
                    if result is None:
                        if not (self._shutdown_event and self._shutdown_event.is_set()):
                            logger.warning(
                                f"Playwright fetch failed or aborted for {url}."
                            )
                        failed_urls.append(f"{url} (fetch_error)")
                        continue
                    if (
                        getattr(result, "success", True)
                        and getattr(result, "markdown", None) is not None
                        and isinstance(result.markdown, str)
                        and len(result.markdown.strip()) > 10
                    ):
                        key = self._generate_key(url)
                        md_path = self.raw_markdown_dir / f"{key}.md"
                        try:
                            md_content = result.markdown
                            md_path.write_text(md_content, encoding="utf-8")
                            self._reverse_lookup[key] = url
                            if url not in collected_urls:
                                collected_urls.append(url)
                                collected_count += 1
                                last_progress_time = time.time()
                            logger.debug(
                                f"Saved MD: {md_path.name} ({len(md_content)} chars) for URL: {url}"
                            )
                        except OSError as e:
                            logger.error(f"Failed to write MD file {md_path}: {e}")
                            failed_urls.append(f"{url} (write_error)")
                            continue
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
                                link_text: Optional[str] = None
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
                        reason = (
                            "success=False"
                            if not getattr(result, "success", True)
                            else (
                                "no markdown"
                                if getattr(result, "markdown", None) is None
                                else (
                                    "markdown not string"
                                    if not isinstance(
                                        getattr(result, "markdown", ""), str
                                    )
                                    else (
                                        "markdown too short"
                                        if len(getattr(result, "markdown", "").strip())
                                        <= 10
                                        else "unknown reason"
                                    )
                                )
                            )
                        )
                        if not (self._shutdown_event and self._shutdown_event.is_set()):
                            logger.warning(
                                f"Skipping result for {url} (Reason: {reason})"
                            )
                        failed_urls.append(f"{url} ({reason})")
                except asyncio.CancelledError:
                    logger.info(f"Crawl task for {url} cancelled.")
                    break
                except Exception as e:
                    if not (self._shutdown_event and self._shutdown_event.is_set()):
                        logger.error(f"Error processing {url}: {e}", exc_info=True)
                    failed_urls.append(f"{url} (processing_error)")
                finally:
                    queue.task_done()
                if time.time() - last_progress_time > STALL_TIMEOUT:
                    logger.warning(
                        f"Crawl may be stalled? No new pages collected for {STALL_TIMEOUT}s. Queue size: {queue.qsize()}."
                    )
                    last_progress_time = time.time()
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

        try:
            await asyncio.wait_for(crawl_loop(), timeout=GLOBAL_TIMEOUT)
        except asyncio.TimeoutError:
            logger.error(f"Global crawl timeout ({GLOBAL_TIMEOUT}s) reached.")
            self.abort()
        except asyncio.CancelledError:
            logger.warning("Crawl main task cancelled externally.")
        except Exception as e:
            logger.error(f"Unexpected error during crawl run: {e}", exc_info=True)
            failed_urls.append(f"GENERAL_ERROR:{e}")
        finally:
            duration = time.time() - start_time
            await self.close()
            if failed_urls:
                logger.warning(
                    f"Crawl encountered issues for {len(failed_urls)} URLs (max 10 shown): {failed_urls[:10]}"
                )
            self._save_reverse_lookup()
            logger.info(
                f"Crawl finished in {duration:.2f}s. Collected: {len(collected_urls)}, Visited: {len(visited)}, Queue Left: {queue.qsize()}"
            )
        return collected_urls if collected_urls is not None else []

    def is_valid_content_url(self, url: str) -> bool:
        if not isinstance(url, str) or not url:
            return False
        try:
            parsed = urlparse(url)
            if parsed.scheme not in ["http", "https"]:
                return False

            url_lower = url.lower()

            # General ignore patterns
            ignore_patterns = [
                r"/__(.*?)__/",
                r"/(cdn|static|assets|media|images|img|css|js|fonts|font)/",
                r"/node_modules/",
                r"/\.git/",
                r"/\.well-known/",
                r"(/|\.)(login|signin|signup|register|auth|account|profile|user|admin|dashboard)($|/|\?|#)",
                r"/search(\?|$|/)",
                r"/(cart|checkout|order|wishlist)($|/|\?|#)",
                r"/(tag|tags|category|categories|author|authors)/",
                r"\?replytocom=",
                r"#",  # Fragment identifier itself means not a new page
                r"^(mailto|tel|javascript|ftp|irc):",
                r"/(feed|rss|atom|xmlrpc)(\.php)?($|/|\?|#)",  # Feeds
                r"/(wp-content|wp-admin|wp-includes|wp-json)/",
                r"/(_next|_nuxt|_svelte)/",
                r"/(cgi-bin|plesk-stat|webmail)/",
                r"/ajax/",
            ]

            # Special handling for /api/ - only ignore if it's likely a data endpoint, not documentation
            if "/api/" in url_lower:
                is_api_doc_page = (
                    any(
                        kw in url_lower
                        for kw in [
                            "doc",
                            "reference",
                            "guide",
                            "manual",
                            "example",
                            "spec",
                            "v1",
                            "v2",
                            "v3",
                        ]
                    )
                    or url_lower.endswith((".html", ".htm", "/"))
                    or re.search(
                        r"/api/.*?/(get|post|put|delete|patch|resource)", url_lower
                    )
                )

                if not is_api_doc_page:
                    if self.verbose_logging:
                        logger.debug(
                            f"Ignoring URL due to /api/ pattern match (not doc): {url}"
                        )
                    return False  # Moved return False out of verbose_logging check

            if any(re.search(p, url_lower) for p in ignore_patterns):
                if self.verbose_logging:
                    logger.debug(f"Ignoring URL due to general pattern match: {url}")
                return False

            ignore_extensions = (
                r"\.(jpg|jpeg|png|gif|bmp|tif|tiff|ico|webp|svg|css|js|json|xml|mp3|mp4|avi|mov|wmv|flv|mkv|webm|"
                r"woff|woff2|ttf|eot|otf|pdf|zip|tar|gz|rar|7z|exe|dmg|iso|bin|msi|doc|docx|xls|xlsx|ppt|pptx|"
                r"csv|rtf|odt|ods|odp|txt|log|ini|cfg|sql|db|bak|conf|yaml|yml|toml)$"
            )
            path_part = parsed.path.split("?")[0].lower()
            if re.search(ignore_extensions, path_part):
                if self.verbose_logging:
                    logger.debug(f"Ignoring URL due to extension match: {url}")
                return False

            root_netlocs = {
                urlparse(root).netloc.lower()
                for root in self.root_urls
                if urlparse(root).netloc
            }
            current_netloc = parsed.netloc.lower()
            if not current_netloc:
                return False
            if not any(
                rn and (current_netloc == rn or current_netloc.endswith("." + rn))
                for rn in root_netlocs
            ):
                if self.verbose_logging:
                    logger.debug(
                        f"Ignoring URL due to domain mismatch ({current_netloc} vs {root_netlocs}): {url}"
                    )
                return False
            return True
        except Exception as e:
            logger.debug(f"URL validation error for '{url}': {e}")
            return False
