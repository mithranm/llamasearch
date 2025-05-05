"""
crawler.py - Asynchronous crawler using a custom priority queue logic.
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
                    break
            finally:
                if hasattr(raw, "aclose"):
                    await raw.aclose()
            return None
        else:
            logger.error(f"Unexpected arun type for {url}: {type(raw)}.")
            return None
    except asyncio.TimeoutError:
        logger.error(f"Timeout ({timeout_seconds}s) fetching {url}")
        return None
    except Exception as e:
        if not (shutdown_event and shutdown_event.is_set()):
            logger.error(f"Exception fetching {url}: {e}", exc_info=True)
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
        headless: bool = True,
        verbose_logging: bool = False,
        shutdown_event: Optional[threading.Event] = None,
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
        logger.info(f"Init Crawl4AICrawler for: {', '.join(self.root_urls)}")
        logger.info(f"Output dir (MD): {self.raw_markdown_dir}")
        logger.info(f"Reverse lookup: {self.reverse_lookup_path}")
        logger.info(
            f"Target: {self.target_links}, Max Depth: {max_depth} => Max Level: {self.max_crawl_level}"
        )
        self._browser_cfg = BrowserConfig(
            browser_type="chromium", headless=headless, verbose=verbose_logging
        )
        self._strategy = AsyncPlaywrightCrawlerStrategy(config=self._browser_cfg)
        self._run_cfg = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            remove_overlay_elements=True,
            stream=False,
            page_timeout=DEFAULT_PAGE_TIMEOUT_MS,
        )

    def normalize_url(self, url: str) -> str:
        if not isinstance(url, str):
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
        return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]

    def _load_reverse_lookup(self):
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
        score = 0.0
        link_lower = link_url.lower()
        try:
            parsed = urlparse(link_lower)
            path_query = parsed.path + "?" + parsed.query
            for keyword in self.relevance_keywords:
                try:
                    score += len(
                        re.findall(r"\b" + re.escape(keyword) + r"\b", path_query)
                    )
                except re.error:
                    score += path_query.count(keyword)
        except Exception as e:
            logger.warning(f"Could not parse/score link {link_url}: {e}")
            return float("inf")
        return max(1.0 / (score + 0.01), 1e-6)

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

        for url in self.root_urls:
            norm_url = self.normalize_url(url)
            if norm_url and norm_url not in visited:
                await queue.put((self._score_link_priority(norm_url), 1, norm_url))
                visited.add(norm_url)

        crawler = AsyncWebCrawler(crawler_strategy=self._strategy)
        collected_count = 0

        async def crawl_loop():
            nonlocal last_progress_time, collected_count
            while not queue.empty() and collected_count < self.target_links:
                if self._user_abort or (
                    self._shutdown_event and self._shutdown_event.is_set()
                ):
                    logger.warning("Crawl loop aborted.")
                    break
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
                    logger.info(
                        f"Crawling [L:{level}/{self.max_crawl_level} | C:{collected_count}/{self.target_links} | Q:{queue.qsize()}]: {url}"
                    )
                    result = await fetch_single(
                        crawler, url, self._run_cfg, self._shutdown_event
                    )
                    if result is None:
                        if not (self._shutdown_event and self._shutdown_event.is_set()):
                            logger.warning(f"Fetch failed/aborted for {url}.")
                        failed_urls.append(url)
                        continue
                    if (
                        getattr(result, "success", True)
                        and getattr(result, "markdown", None) is not None
                    ):
                        key = self._generate_key(url)
                        md_path = self.raw_markdown_dir / f"{key}.md"
                        try:
                            md_content = result.markdown
                            if not isinstance(md_content, str):
                                logger.error(f"MD not string for {url}. Skip.")
                                failed_urls.append(url)
                                continue
                            md_path.write_text(md_content, encoding="utf-8")
                            self._reverse_lookup[key] = url
                            if url not in collected_urls:
                                collected_urls.append(url)
                                collected_count += 1
                                last_progress_time = time.time()
                            logger.debug(
                                f"Saved MD: {md_path.name} ({len(md_content)} chars)"
                            )
                        except OSError as e:
                            logger.error(f"Failed write MD {md_path}: {e}")
                            failed_urls.append(url)
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
                                    if (
                                        self.is_valid_content_url(norm_link)
                                        and norm_link not in visited
                                    ):
                                        visited.add(norm_link)
                                        await queue.put(
                                            (
                                                self._score_link_priority(norm_link),
                                                next_level,
                                                norm_link,
                                            )
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
                        if not (self._shutdown_event and self._shutdown_event.is_set()):
                            logger.warning(f"No MD/success=false for {url}")
                        failed_urls.append(url)
                except asyncio.CancelledError:
                    logger.info("Crawl task cancelled.")
                    break
                except Exception as e:
                    if not (self._shutdown_event and self._shutdown_event.is_set()):
                        logger.error(f"Error processing {url}: {e}", exc_info=True)
                    failed_urls.append(url)
                finally:
                    queue.task_done()
                if time.time() - last_progress_time > STALL_TIMEOUT:
                    logger.warning(
                        f"Crawl stalled? No progress {STALL_TIMEOUT}s. Q:{queue.qsize()}."
                    )
                    last_progress_time = time.time()
            if self._user_abort or (
                self._shutdown_event and self._shutdown_event.is_set()
            ):
                logger.info("--- DIAG: Crawl loop exit: abort/shutdown.")
            elif queue.empty():
                logger.info("--- DIAG: Crawl loop exit: queue empty.")
            elif collected_count >= self.target_links:
                logger.info("--- DIAG: Crawl loop exit: target reached.")
            else:
                logger.warning(
                    f"--- DIAG: Crawl loop exit: unknown. Q:{queue.qsize()}, C:{collected_count}"
                )

        try:
            await asyncio.wait_for(crawl_loop(), timeout=GLOBAL_TIMEOUT)
        except asyncio.TimeoutError:
            logger.error(f"Global crawl timeout ({GLOBAL_TIMEOUT}s) reached.")
            self.abort()
        except asyncio.CancelledError:
            logger.warning("Crawl main task cancelled.")
        except Exception as e:
            logger.error(f"Unexpected crawl error: {e}", exc_info=True)
            failed_urls.append(f"GENERAL_ERROR:{e}")
        finally:
            duration = time.time() - start_time
            if failed_urls:
                logger.warning(
                    f"Crawl issues for {len(failed_urls)} URLs (max 10 shown): {failed_urls[:10]}"
                )
            self._save_reverse_lookup()
            logger.info(
                f"Crawl finished in {duration:.2f}s. Collected: {len(collected_urls)}, Visited: {len(visited)}, Q Left: {queue.qsize()}"
            )
        return list(self._reverse_lookup.values())  # Return URLs actually saved

    def is_valid_content_url(self, url: str) -> bool:
        if not isinstance(url, str) or not url:
            return False
        try:
            parsed = urlparse(url)
            if parsed.scheme not in ["http", "https"]:
                return False
            # <<< Removed unused path_low >>>
            ignore_patterns = [
                r"/(cdn|static|assets|media|images|css|js|fonts)/",
                r"/__(.*)__/",
                r"(/|\.)(login|signup|register|auth|account|profile|admin|user)($|/|\?)",
                r"/search\?",
                r"/cart",
                r"/checkout",
                r"/order",
                r"/tag/",
                r"/category/",
                r"/author/",
                r"tel:",
                r"mailto:",
                r"javascript:",
                r"/api/",
                r"/rest/",
                r"/feed/",
                r"/rss/",
                r"/atom/",
                r"/(wp-content|wp-admin|wp-includes|wp-json)/",
                r"/ajax/",
                r"xmlrpc\.php",
            ]
            if any(re.search(p, url.lower()) for p in ignore_patterns):
                return False
            ignore_extensions = r"\.(jpg|jpeg|png|gif|bmp|tif|tiff|mp3|mp4|avi|mov|wmv|webp|svg|css|js|json|xml|ico|woff|woff2|ttf|eot|otf|pdf|zip|tar|gz|rar|7z|exe|dmg|iso|ppt|pptx|doc|docx|xls|xlsx|csv|txt|rtf)$"
            if re.search(ignore_extensions, parsed.path.split("?")[0].lower()):
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
                if not any(
                    rd and (current_domain == rd or current_domain.endswith("." + rd))
                    for rd in root_domains
                ):
                    return False
            return True
        except Exception as e:
            logger.debug(f"URL validation error '{url}': {e}")
            return False
