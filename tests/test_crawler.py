import unittest
from unittest.mock import patch, MagicMock, AsyncMock, call, mock_open, ANY
import asyncio
import hashlib
import json
import os # For os.replace mock
import re
import threading
from pathlib import Path
import logging # For spec in mocks and asserting log levels

# Patch setup_logging for the crawler module
MOCK_CRAWLER_SETUP_LOGGING_TARGET = 'llamasearch.utils.setup_logging'
mock_crawler_logger_instance = MagicMock(spec=logging.Logger)
crawler_logger_patcher = patch(MOCK_CRAWLER_SETUP_LOGGING_TARGET, return_value=mock_crawler_logger_instance)
crawler_logger_patcher.start()

# Import target module and its dependencies AFTER global mocks
from llamasearch.core.crawler import (
    sanitize_string,
    fetch_single,
    Crawl4AICrawler,
    DEFAULT_RELEVANCE_KEYWORDS,
    DEFAULT_PAGE_TIMEOUT_MS,
    DEFAULT_FETCH_TIMEOUT_S,
    DEFAULT_CRAWL_DELAY_S
)

# Import crawl4ai types for spec
from crawl4ai import (
    AsyncWebCrawler as RealAsyncWebCrawler,
    BrowserConfig as RealBrowserConfig,
    CrawlerRunConfig as RealCrawlerRunConfig,
    CrawlResult as RealCrawlResult,
    CacheMode
)
# --- CORRECTED IMPORT FOR CrawlResultContainer ---
from crawl4ai.models import CrawlResultContainer as RealCrawlResultContainer
# --- END CORRECTION ---
from crawl4ai.async_crawler_strategy import AsyncPlaywrightCrawlerStrategy as RealAsyncPlaywrightCrawlerStrategy


# Paths for patching within test methods
ASYNC_WEB_CRAWLER_PATH = 'llamasearch.core.crawler.AsyncWebCrawler'
BROWSER_CONFIG_PATH = 'llamasearch.core.crawler.BrowserConfig'
CRAWLER_RUN_CONFIG_PATH = 'llamasearch.core.crawler.CrawlerRunConfig'
ASYNC_PLAYWRIGHT_STRATEGY_PATH = 'llamasearch.core.crawler.AsyncPlaywrightCrawlerStrategy'
ASYNCIO_PATH = 'asyncio' # For patching asyncio.wait_for, asyncio.sleep, asyncio.PriorityQueue
OS_PATH = 'os' # For os.replace

# Module-level teardown for global patchers
def tearDownModule():
    crawler_logger_patcher.stop()

class TestCrawlerUtils(unittest.IsolatedAsyncioTestCase):

    def test_sanitize_string(self):
        self.assertEqual(sanitize_string("http://example.com/path/to/file?query=1"), "example.com_path_to_file_query_1")
        self.assertEqual(sanitize_string("https://www.example.com/a b c!@#$%^&*()=+[]{};':\",./<>?\\|`~"), "example.com_a_b_c_")
        self.assertEqual(sanitize_string("short"), "short")
        self.assertEqual(sanitize_string("a" * 50), "a" * 40)
        self.assertEqual(sanitize_string(" leading_trailing_underscores_ "), "leading_trailing_underscores")
        self.assertEqual(sanitize_string(""), "default")
        self.assertEqual(sanitize_string("https://domain.com/path_with_many___underscores"), "domain.com_path_with_many_underscores")
        self.assertEqual(sanitize_string("http://test.com/%E4%B8%AD%E6%96%87"), "test.com_中文") # Test unquote
        self.assertEqual(sanitize_string("http://test.com/", max_length=10), "test.com")
        self.assertEqual(sanitize_string("http://", max_length=10), "default")
        self.assertEqual(sanitize_string("/////"), "default")


    @patch(ASYNC_WEB_CRAWLER_PATH, spec=RealAsyncWebCrawler)
    async def test_fetch_single_success_crawl_result_container(self, MockAsyncWebCrawlerClass):
        mock_arun_result_container = MagicMock(spec=RealCrawlResultContainer)
        mock_crawl_result = MagicMock(spec=RealCrawlResult)
        mock_arun_result_container._results = [mock_crawl_result]
        mock_arun_result_container.__getitem__.return_value = mock_crawl_result

        mock_web_crawler_param_instance = MagicMock(spec=RealAsyncWebCrawler)
        mock_web_crawler_param_instance.arun = AsyncMock(return_value=mock_arun_result_container)

        mock_cfg = MagicMock(spec=RealCrawlerRunConfig)
        shutdown_event = threading.Event()

        result = await fetch_single(mock_web_crawler_param_instance, "http://example.com", mock_cfg, shutdown_event, timeout_seconds=1.0)
        self.assertEqual(result, mock_crawl_result)
        mock_web_crawler_param_instance.arun.assert_awaited_once_with(url="http://example.com", config=mock_cfg)

    @patch(ASYNC_WEB_CRAWLER_PATH, spec=RealAsyncWebCrawler)
    async def test_fetch_single_success_direct_crawl_result(self, MockAsyncWebCrawlerClass):
        mock_crawl_result = MagicMock(spec=RealCrawlResult)
        mock_web_crawler_param_instance = MagicMock(spec=RealAsyncWebCrawler)
        mock_web_crawler_param_instance.arun = AsyncMock(return_value=mock_crawl_result)
        mock_cfg = MagicMock(spec=RealCrawlerRunConfig)

        result = await fetch_single(mock_web_crawler_param_instance, "http://example.com", mock_cfg, timeout_seconds=1.0)
        self.assertEqual(result, mock_crawl_result)

    @patch(ASYNC_WEB_CRAWLER_PATH, spec=RealAsyncWebCrawler)
    async def test_fetch_single_success_async_generator(self, MockAsyncWebCrawlerClass):
        mock_crawl_result = MagicMock(spec=RealCrawlResult)

        async def mock_gen_func():
            yield mock_crawl_result

        mock_async_gen = mock_gen_func()

        mock_web_crawler_param_instance = MagicMock(spec=RealAsyncWebCrawler)
        mock_web_crawler_param_instance.arun = AsyncMock(return_value=mock_async_gen)
        mock_cfg = MagicMock(spec=RealCrawlerRunConfig)

        result = await fetch_single(mock_web_crawler_param_instance, "http://example.com", mock_cfg, timeout_seconds=1.0)
        self.assertEqual(result, mock_crawl_result)


    @patch(ASYNC_WEB_CRAWLER_PATH, spec=RealAsyncWebCrawler)
    @patch(f'{ASYNCIO_PATH}.wait_for', side_effect=asyncio.TimeoutError)
    async def test_fetch_single_timeout_error(self, mock_wait_for, MockAsyncWebCrawlerClass):
        mock_web_crawler_param_instance = MagicMock(spec=RealAsyncWebCrawler)
        mock_cfg = MagicMock(spec=RealCrawlerRunConfig)
        result = await fetch_single(mock_web_crawler_param_instance, "http://example.com", mock_cfg, timeout_seconds=0.01)
        self.assertIsNone(result)
        mock_crawler_logger_instance.error.assert_any_call(f"Timeout (0.01s) fetching http://example.com with Playwright")


    @patch(ASYNC_WEB_CRAWLER_PATH, spec=RealAsyncWebCrawler)
    async def test_fetch_single_generic_exception(self, MockAsyncWebCrawlerClass):
        mock_web_crawler_param_instance = MagicMock(spec=RealAsyncWebCrawler)
        mock_web_crawler_param_instance.arun = AsyncMock(side_effect=Exception("Playwright crash"))
        mock_cfg = MagicMock(spec=RealCrawlerRunConfig)
        result = await fetch_single(mock_web_crawler_param_instance, "http://example.com", mock_cfg, timeout_seconds=1.0)
        self.assertIsNone(result)
        mock_crawler_logger_instance.error.assert_any_call("Playwright exception fetching http://example.com: Playwright crash", exc_info=True)

    @patch(ASYNC_WEB_CRAWLER_PATH, spec=RealAsyncWebCrawler)
    async def test_fetch_single_shutdown_before_fetch(self, MockAsyncWebCrawlerClass):
        mock_web_crawler_param_instance = MagicMock(spec=RealAsyncWebCrawler)
        mock_cfg = MagicMock(spec=RealCrawlerRunConfig)
        shutdown_event = threading.Event()
        shutdown_event.set()
        result = await fetch_single(mock_web_crawler_param_instance, "http://example.com", mock_cfg, shutdown_event)
        self.assertIsNone(result)
        mock_crawler_logger_instance.info.assert_any_call("Shutdown requested before fetching http://example.com")
        mock_web_crawler_param_instance.arun.assert_not_awaited()

    @patch(ASYNC_WEB_CRAWLER_PATH, spec=RealAsyncWebCrawler)
    async def test_fetch_single_shutdown_after_fetch(self, MockAsyncWebCrawlerClass):
        mock_web_crawler_param_instance = MagicMock(spec=RealAsyncWebCrawler)
        shutdown_event = threading.Event()
        def arun_side_effect(*args, **kwargs):
            shutdown_event.set()
            return MagicMock(spec=RealCrawlResult) # Return a mock CrawlResult
        mock_web_crawler_param_instance.arun = AsyncMock(side_effect=arun_side_effect)
        mock_cfg = MagicMock(spec=RealCrawlerRunConfig)

        result = await fetch_single(mock_web_crawler_param_instance, "http://example.com", mock_cfg, shutdown_event, timeout_seconds=1.0)
        self.assertIsNone(result)
        mock_crawler_logger_instance.info.assert_any_call("Shutdown requested after fetching http://example.com")

    @patch(ASYNC_WEB_CRAWLER_PATH, spec=RealAsyncWebCrawler)
    async def test_fetch_single_unexpected_arun_result_type(self, MockAsyncWebCrawlerClass):
        mock_web_crawler_param_instance = MagicMock(spec=RealAsyncWebCrawler)
        mock_web_crawler_param_instance.arun = AsyncMock(return_value=123) # Non-CrawlResult type
        mock_cfg = MagicMock(spec=RealCrawlerRunConfig)
        result = await fetch_single(mock_web_crawler_param_instance, "http://example.com", mock_cfg, timeout_seconds=1.0)
        self.assertIsNone(result)
        mock_crawler_logger_instance.error.assert_any_call("Unexpected arun result type for http://example.com: <class 'int'>.")

    @patch(ASYNC_WEB_CRAWLER_PATH, spec=RealAsyncWebCrawler)
    async def test_fetch_single_crawl_result_container_empty(self, MockAsyncWebCrawlerClass):
        mock_arun_result_container = MagicMock(spec=RealCrawlResultContainer)
        mock_arun_result_container._results = [] # Empty results
        mock_web_crawler_param_instance = MagicMock(spec=RealAsyncWebCrawler)
        mock_web_crawler_param_instance.arun = AsyncMock(return_value=mock_arun_result_container)
        mock_cfg = MagicMock(spec=RealCrawlerRunConfig)

        result = await fetch_single(mock_web_crawler_param_instance, "http://example.com", mock_cfg, timeout_seconds=1.0)
        self.assertIsNone(result)
        mock_crawler_logger_instance.warning.assert_any_call("CrawlResultContainer empty or invalid for http://example.com.")

    @patch(ASYNC_WEB_CRAWLER_PATH, spec=RealAsyncWebCrawler)
    async def test_fetch_single_generator_yields_none_or_invalid(self, MockAsyncWebCrawlerClass):
        async def mock_gen_func_none(): yield None
        async def mock_gen_func_invalid(): yield "not a CrawlResult"

        mock_web_crawler_param_instance = MagicMock(spec=RealAsyncWebCrawler)
        mock_cfg = MagicMock(spec=RealCrawlerRunConfig)

        mock_web_crawler_param_instance.arun = AsyncMock(return_value=mock_gen_func_none())
        result_none = await fetch_single(mock_web_crawler_param_instance, "http://example.com/none", mock_cfg)
        self.assertIsNone(result_none)

        mock_web_crawler_param_instance.arun = AsyncMock(return_value=mock_gen_func_invalid())
        result_invalid = await fetch_single(mock_web_crawler_param_instance, "http://example.com/invalid", mock_cfg)
        self.assertIsNone(result_invalid)

    @patch(ASYNC_WEB_CRAWLER_PATH, spec=RealAsyncWebCrawler)
    async def test_fetch_single_playwright_specific_exceptions(self, MockAsyncWebCrawlerClass):
        mock_web_crawler_param_instance = MagicMock(spec=RealAsyncWebCrawler)
        mock_cfg = MagicMock(spec=RealCrawlerRunConfig)

        mock_web_crawler_param_instance.arun = AsyncMock(side_effect=Exception("Target page, context or browser has been closed"))
        await fetch_single(mock_web_crawler_param_instance, "http://example.com/closed", mock_cfg)
        mock_crawler_logger_instance.warning.assert_any_call("Playwright page closed unexpectedly for http://example.com/closed: Target page, context or browser has been closed")

        mock_web_crawler_param_instance.arun = AsyncMock(side_effect=Exception("Timeout 30000ms exceeded"))
        await fetch_single(mock_web_crawler_param_instance, "http://example.com/timeout", mock_cfg)
        mock_crawler_logger_instance.error.assert_any_call("Playwright internal timeout fetching http://example.com/timeout: Timeout 30000ms exceeded")


class TestCrawl4AICrawler(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.path_patcher = patch('llamasearch.core.crawler.Path', autospec=True)
        self.MockPathClass = self.path_patcher.start()

        self.mock_base_crawl_dir_path_obj = MagicMock(spec=Path, name="BaseCrawlDirPath")
        self.mock_raw_markdown_dir_path_obj = MagicMock(spec=Path, name="RawMarkdownDirPath")
        self.mock_reverse_lookup_path_obj = MagicMock(spec=Path, name="ReverseLookupPath")

        def path_constructor_side_effect(path_str_or_obj):
            path_str = str(path_str_or_obj) # Ensure it's a string for comparisons
            if path_str == "/fake/test_base_crawl_dir":
                return self.mock_base_crawl_dir_path_obj
            if path_str == "/fake/test_base_crawl_dir/raw":
                 return self.mock_raw_markdown_dir_path_obj
            if path_str == "/fake/test_base_crawl_dir/reverse_lookup.json":
                 return self.mock_reverse_lookup_path_obj
            # For other paths, return a generic new mock
            new_mock = MagicMock(spec=Path, name=f"Path({path_str})")
            new_mock.__str__.return_value = path_str
            new_mock.name = os.path.basename(path_str)
            new_mock.exists.return_value = False
            new_mock.parent = MagicMock(spec=Path, name=f"PathParent({os.path.dirname(path_str)})")
            new_mock.parent.mkdir = MagicMock()
            new_mock.__truediv__.side_effect = lambda other: path_constructor_side_effect(os.path.join(path_str, str(other)))
            new_mock.with_suffix.side_effect = lambda suffix: path_constructor_side_effect(path_str.rsplit('.',1)[0] + suffix if '.' in path_str else path_str + suffix)
            return new_mock
        self.MockPathClass.side_effect = path_constructor_side_effect

        self.mock_base_crawl_dir_path_obj.__str__.return_value = "/fake/test_base_crawl_dir"
        self.mock_raw_markdown_dir_path_obj.__str__.return_value = "/fake/test_base_crawl_dir/raw"
        self.mock_reverse_lookup_path_obj.__str__.return_value = "/fake/test_base_crawl_dir/reverse_lookup.json"

        self.mock_base_crawl_dir_path_obj.__truediv__.side_effect = lambda x: \
            self.mock_raw_markdown_dir_path_obj if x == "raw" else \
            self.mock_reverse_lookup_path_obj if x == "reverse_lookup.json" else \
            path_constructor_side_effect(os.path.join(str(self.mock_base_crawl_dir_path_obj), str(x)))

        self.mock_reverse_lookup_path_obj.parent = self.mock_base_crawl_dir_path_obj


        self.mock_raw_markdown_dir_path_obj.mkdir = MagicMock()
        self.mock_reverse_lookup_path_obj.exists = MagicMock(return_value=False)

        self.mock_shutdown_event = MagicMock(spec=threading.Event)
        self.mock_shutdown_event.is_set.return_value = False

        self.browser_config_patcher = patch(BROWSER_CONFIG_PATH, spec=RealBrowserConfig)
        self.MockBrowserConfig = self.browser_config_patcher.start()
        self.mock_browser_config_instance = self.MockBrowserConfig.return_value

        self.run_config_patcher = patch(CRAWLER_RUN_CONFIG_PATH, spec=RealCrawlerRunConfig)
        self.MockCrawlerRunConfig = self.run_config_patcher.start()
        self.mock_run_config_instance = self.MockCrawlerRunConfig.return_value

        self.strategy_patcher = patch(ASYNC_PLAYWRIGHT_STRATEGY_PATH, spec=RealAsyncPlaywrightCrawlerStrategy)
        self.MockStrategy = self.strategy_patcher.start()
        self.mock_strategy_instance = self.MockStrategy.return_value

        self.async_web_crawler_patcher = patch(ASYNC_WEB_CRAWLER_PATH, spec=RealAsyncWebCrawler)
        self.MockAsyncWebCrawler = self.async_web_crawler_patcher.start()
        self.mock_async_web_crawler_instance = self.MockAsyncWebCrawler.return_value
        self.mock_async_web_crawler_instance.close = AsyncMock()

        self.fetch_single_patcher = patch('llamasearch.core.crawler.fetch_single', new_callable=AsyncMock)
        self.mock_fetch_single = self.fetch_single_patcher.start()

        self.priority_queue_patcher = patch(f'{ASYNCIO_PATH}.PriorityQueue')
        self.MockPriorityQueue = self.priority_queue_patcher.start()
        self.mock_priority_queue_instance = self.MockPriorityQueue.return_value
        self.mock_priority_queue_instance.empty.return_value = True
        self.mock_priority_queue_instance.qsize.return_value = 0

        self.mock_open_patcher = patch('builtins.open', new_callable=mock_open)
        self.mock_open_file = self.mock_open_patcher.start()

        self.os_replace_patcher = patch(f'{OS_PATH}.replace')
        self.mock_os_replace = self.os_replace_patcher.start()

        self.hashlib_patcher = patch('hashlib.sha256')
        self.mock_sha256 = self.hashlib_patcher.start()
        self.mock_sha256_instance = self.mock_sha256.return_value
        self.mock_sha256_instance.hexdigest.return_value = "fixed16charhash0"

        mock_crawler_logger_instance.reset_mock()

    def tearDown(self):
        self.path_patcher.stop()
        self.browser_config_patcher.stop()
        self.run_config_patcher.stop()
        self.strategy_patcher.stop()
        self.async_web_crawler_patcher.stop()
        self.fetch_single_patcher.stop()
        self.priority_queue_patcher.stop()
        self.mock_open_patcher.stop()
        self.os_replace_patcher.stop()
        self.hashlib_patcher.stop()

    def _create_crawler(self, root_urls=None, **kwargs):
        if root_urls is None: root_urls = ["http://example.com"]
        default_kwargs = {
            "base_crawl_dir": Path("/fake/test_base_crawl_dir"),
            "shutdown_event": self.mock_shutdown_event,
        }
        default_kwargs.update(kwargs)
        return Crawl4AICrawler(root_urls=root_urls, **default_kwargs)

    def test_init_successful_default_params(self):
        crawler = self._create_crawler()
        self.assertEqual(crawler.root_urls, ["https://example.com/"])
        self.mock_raw_markdown_dir_path_obj.mkdir.assert_called_once_with(parents=True, exist_ok=True)
        self.MockBrowserConfig.assert_called_once_with(browser_type="chromium", headless=True, verbose=False)
        self.MockStrategy.assert_called_once_with(config=self.mock_browser_config_instance)
        self.mock_reverse_lookup_path_obj.exists.assert_called() # Called during _load_reverse_lookup

    def test_init_empty_root_urls_raises_value_error(self):
        with self.assertRaises(ValueError): self._create_crawler(root_urls=[])

    def test_init_strategy_creation_fails_raises_runtime_error(self):
        self.MockStrategy.side_effect = Exception("Strategy init failed")
        with self.assertRaisesRegex(RuntimeError, "Could not initialize Playwright crawler strategy."):
            self._create_crawler()

    async def test_close_with_crawler_instance(self):
        crawler = self._create_crawler()
        # Simulate that _crawler was set during run_crawl
        crawler._crawler = self.mock_async_web_crawler_instance
        await crawler.close()
        self.mock_async_web_crawler_instance.close.assert_awaited_once()
        self.assertIsNone(crawler._crawler)

    async def test_close_without_crawler_instance(self):
        crawler = self._create_crawler()
        crawler._crawler = None # Ensure it's None
        await crawler.close()
        self.mock_async_web_crawler_instance.close.assert_not_awaited() # Should not be called

    def test_normalize_url(self):
        crawler = self._create_crawler()
        self.assertEqual(crawler.normalize_url("example.com"), "https://example.com/")
        self.assertEqual(crawler.normalize_url("http://example.com/path#frag"), "http://example.com/path")
        self.assertEqual(crawler.normalize_url("https://foo.com"), "https://foo.com/")
        self.assertEqual(crawler.normalize_url("https://foo.com/bar"), "https://foo.com/bar")
        self.assertEqual(crawler.normalize_url("http://foo.com?q=1"), "http://foo.com/?q=1")
        self.assertEqual(crawler.normalize_url("not a url"), "")
        self.assertEqual(crawler.normalize_url(None), "") # Test with None input

    def test_generate_key(self):
        crawler = self._create_crawler()
        key = crawler._generate_key("https://example.com/")
        self.assertEqual(len(key), 16)
        self.mock_sha256.assert_called_once_with("https://example.com/".encode("utf-8"))
        self.mock_sha256_instance.hexdigest.assert_called_once()

    def test_load_reverse_lookup_file_exists_valid_json(self):
        # Init calls _load_reverse_lookup, so we test its behavior as part of init
        # or by calling it directly after ensuring the state
        self.mock_reverse_lookup_path_obj.exists.return_value = True
        dummy_data = {"key1": "url1"}
        # Configure mock_open to simulate reading JSON
        self.mock_open_file.return_value.__enter__.return_value.read.return_value = json.dumps(dummy_data)

        crawler = self._create_crawler() # This will call _load_reverse_lookup

        self.assertEqual(crawler._reverse_lookup, dummy_data)
        self.mock_open_file.assert_called_with(self.mock_reverse_lookup_path_obj, "r", encoding="utf-8")

    def test_load_reverse_lookup_file_exists_invalid_json(self):
        self.mock_reverse_lookup_path_obj.exists.return_value = True
        self.mock_open_file.return_value.__enter__.return_value.read.return_value = "invalid json"
        # Simulate json.load raising an error
        with patch('json.load', side_effect=json.JSONDecodeError("err", "doc", 0)):
            crawler = self._create_crawler()
        self.assertEqual(crawler._reverse_lookup, {}) # Should reset to empty
        mock_crawler_logger_instance.error.assert_any_call(
            f"Error loading reverse lookup file: err: line 1 column 1 (char 0). Starting fresh.",
            exc_info=True
        )

    def test_load_reverse_lookup_file_not_exist(self):
        self.mock_reverse_lookup_path_obj.exists.return_value = False
        crawler = self._create_crawler()
        self.assertEqual(crawler._reverse_lookup, {})
        mock_crawler_logger_instance.info.assert_any_call(
             f"No existing reverse lookup file found at {self.mock_reverse_lookup_path_obj}."
        )

    def test_save_reverse_lookup_successful(self):
        crawler = self._create_crawler()
        crawler._reverse_lookup = {"k1": "url1"}
        # Mock the temp path object that would be created
        mock_temp_path = MagicMock(spec=Path, name="TempPath")
        mock_temp_path.__str__.return_value = str(self.mock_reverse_lookup_path_obj) + ".tmp"
        mock_temp_path.exists.return_value = False # Assume it doesn't exist initially
        self.mock_reverse_lookup_path_obj.with_suffix.return_value = mock_temp_path
        self.mock_reverse_lookup_path_obj.parent.mkdir.assert_called_with(parents=True, exist_ok=True)

        with patch('json.dump') as mock_json_dump:
            crawler._save_reverse_lookup()
            # Check if mock_open was called for the temp path
            self.mock_open_file.assert_called_with(mock_temp_path, "w", encoding="utf-8")
            mock_json_dump.assert_called_once_with(crawler._reverse_lookup, ANY, indent=2, ensure_ascii=False)
        self.mock_os_replace.assert_called_once_with(mock_temp_path, self.mock_reverse_lookup_path_obj)

    def test_save_reverse_lookup_os_error_on_replace(self):
        crawler = self._create_crawler()
        crawler._reverse_lookup = {"k1": "url1"}
        mock_temp_path = MagicMock(spec=Path)
        self.mock_reverse_lookup_path_obj.with_suffix.return_value = mock_temp_path
        self.mock_os_replace.side_effect = OSError("Cannot replace")
        mock_temp_path.exists.return_value = True # Simulate temp file was created
        mock_temp_path.unlink = MagicMock()


        crawler._save_reverse_lookup()

        mock_crawler_logger_instance.error.assert_any_call(
            f"Error saving reverse lookup file: Cannot replace", exc_info=True
        )
        mock_temp_path.unlink.assert_called_once() # Ensure cleanup is attempted

    def test_score_link_priority(self):
        crawler = self._create_crawler(relevance_keywords=["doc", "api"])
        # URL and text match
        p_high = crawler._score_link_priority("http://example.com/doc/api_guide", "Official API Documentation")
        # Only URL matches
        p_mid_url = crawler._score_link_priority("http://example.com/doc/something", "Some other page")
        # Only text matches
        p_mid_text = crawler._score_link_priority("http://example.com/other_page", "Link to API reference")
        # No match
        p_low = crawler._score_link_priority("http://example.com/general_info", "About us")

        self.assertTrue(p_high < p_mid_url)
        self.assertTrue(p_high < p_mid_text)
        self.assertTrue(p_mid_url < p_low or p_mid_text < p_low) # At least one mid should be better than low
        self.assertGreater(p_low, 0) # Priority should be positive

        # Test with invalid URL for parsing
        bad_url_priority = crawler._score_link_priority("::not_a_url:::", "text")
        self.assertEqual(bad_url_priority, float('inf')) # Should get max priority (lowest relevance)

    def test_abort(self):
        crawler = self._create_crawler()
        self.assertFalse(crawler._user_abort)
        crawler.abort()
        self.assertTrue(crawler._user_abort)
        mock_crawler_logger_instance.info.assert_called_with("User abort requested for crawler.")

    def test_is_valid_content_url(self):
        crawler = self._create_crawler(root_urls=["http://example.com", "http://sub.example.org"])
        # Valid
        self.assertTrue(crawler.is_valid_content_url("https://example.com/docs/page1"))
        self.assertTrue(crawler.is_valid_content_url("http://www.example.com/another-page.html")) # www is fine
        self.assertTrue(crawler.is_valid_content_url("http://sub.example.org/blog/post"))
        self.assertTrue(crawler.is_valid_content_url("https://deep.sub.example.org/path")) # Sub-subdomain of a root

        # Invalid scheme
        self.assertFalse(crawler.is_valid_content_url("ftp://example.com/file.zip"))
        # Asset paths
        self.assertFalse(crawler.is_valid_content_url("https://example.com/assets/image.jpg"))
        self.assertFalse(crawler.is_valid_content_url("https://example.com/static/css/style.css"))
        # Auth paths
        self.assertFalse(crawler.is_valid_content_url("https://example.com/login"))
        self.assertFalse(crawler.is_valid_content_url("https://example.com/admin/dashboard"))
        # Asset extensions
        self.assertFalse(crawler.is_valid_content_url("https://example.com/download.pdf"))
        self.assertFalse(crawler.is_valid_content_url("https://example.com/archive.tar.gz"))
        # Different domain
        self.assertFalse(crawler.is_valid_content_url("https://another-domain.com/page"))
        # Empty or None URL
        self.assertFalse(crawler.is_valid_content_url(""))
        self.assertFalse(crawler.is_valid_content_url(None))
        # Invalid URL structure causing urlparse to fail (or return unexpected netloc)
        self.assertFalse(crawler.is_valid_content_url("http:///onlypath")) # No netloc


    @patch(f'{ASYNCIO_PATH}.sleep', new_callable=AsyncMock) # Mock asyncio.sleep
    async def test_run_crawl_happy_path_reaches_target(self, mock_async_sleep):
        root_url_normalized = "https://example.com/"
        crawler = self._create_crawler(root_urls=["http://example.com"], target_links=1)

        # Setup PriorityQueue mock behavior
        # Items to be "dequeued": (priority, level, url)
        queue_items_to_get = [(0.5, 1, root_url_normalized)]
        current_item_index = 0

        def mock_queue_empty_side_effect():
            return current_item_index >= len(queue_items_to_get)

        async def mock_queue_get_side_effect():
            nonlocal current_item_index
            if current_item_index < len(queue_items_to_get):
                item = queue_items_to_get[current_item_index]
                current_item_index += 1
                return item
            # This should not be reached if target_links logic is correct
            raise asyncio.QueueEmpty("Queue became empty unexpectedly in test")

        self.mock_priority_queue_instance.empty.side_effect = mock_queue_empty_side_effect
        self.mock_priority_queue_instance.get = AsyncMock(side_effect=mock_queue_get_side_effect)
        self.mock_priority_queue_instance.put = AsyncMock() # To verify items are added

        # Mock result from fetch_single
        mock_crawl_result = MagicMock(spec=RealCrawlResult)
        mock_crawl_result.success = True
        # Ensure markdown is long enough
        mock_crawl_result.markdown = "This is some sample markdown content more than ten characters long."
        # Provide some links to test link processing logic (though target_links=1 means it won't recurse here)
        mock_crawl_result.links = {"internal": [{"href": "/page2", "text": "Next Page"}]}
        self.mock_fetch_single.return_value = mock_crawl_result

        # Mock the path object for the markdown file that will be written
        mock_md_file_path = self.MockPathClass(f"/fake/test_base_crawl_dir/raw/fixed16charhash0.md")
        mock_md_file_path.write_text = MagicMock()
        # Ensure the __truediv__ for raw_markdown_dir returns this specific mock when key.md is accessed
        self.mock_raw_markdown_dir_path_obj.__truediv__.return_value = mock_md_file_path


        collected_urls = await crawler.run_crawl()

        self.assertEqual(len(collected_urls), 1)
        self.assertIn(root_url_normalized, collected_urls)
        self.mock_fetch_single.assert_called_once_with(
            self.mock_async_web_crawler_instance, # The crawler instance from AsyncWebCrawler
            root_url_normalized,
            self.mock_run_config_instance, # The _run_cfg attribute
            self.mock_shutdown_event,
            timeout_seconds=DEFAULT_FETCH_TIMEOUT_S
        )
        mock_md_file_path.write_text.assert_called_once_with(mock_crawl_result.markdown, encoding="utf-8")
        self.assertTrue(crawler._reverse_lookup["fixed16charhash0"] == root_url_normalized)
        self.mock_os_replace.assert_called_once() # _save_reverse_lookup called at the end
        self.mock_async_web_crawler_instance.close.assert_awaited_once() # Crawler closed at the end
        # Check if the initial root URL was put onto the queue
        self.mock_priority_queue_instance.put.assert_any_call(
             (ANY, 1, root_url_normalized) # priority, level, url
        )


    @patch(f'{ASYNCIO_PATH}.sleep', new_callable=AsyncMock)
    async def test_run_crawl_initial_crawler_init_fails(self, mock_async_sleep):
        # Make AsyncWebCrawler constructor fail
        self.MockAsyncWebCrawler.side_effect = RuntimeError("Test AsyncWebCrawler init fail")
        crawler = self._create_crawler()

        collected_urls = await crawler.run_crawl()

        self.assertEqual(len(collected_urls), 0)
        mock_crawler_logger_instance.error.assert_any_call(
            "Failed to initialize AsyncWebCrawler: Test AsyncWebCrawler init fail", exc_info=True
        )
        # close() is called in finally, it should handle self._crawler being None
        self.mock_async_web_crawler_instance.close.assert_not_awaited() # close on the instance shouldn't be called if instance creation failed
        self.mock_os_replace.assert_called_once() # Save lookup still called in finally

    @patch(f'{ASYNCIO_PATH}.sleep', new_callable=AsyncMock)
    async def test_run_crawl_aborts_on_signal(self, mock_async_sleep):
        crawler = self._create_crawler(target_links=10) # High target
        self.mock_shutdown_event.is_set.return_value = True # Simulate immediate shutdown

        # Queue needs to have something for the loop to start
        self.mock_priority_queue_instance.empty.return_value = False
        self.mock_priority_queue_instance.get = AsyncMock(return_value=(0.5,1,"https://example.com/"))


        collected_urls = await crawler.run_crawl()

        self.assertEqual(len(collected_urls), 0)
        mock_crawler_logger_instance.warning.assert_any_call("Crawl loop aborted by signal.")
        self.mock_fetch_single.assert_not_called() # Should not attempt to fetch

    @patch(f'{ASYNCIO_PATH}.sleep', new_callable=AsyncMock)
    async def test_run_crawl_global_timeout(self, mock_async_sleep):
        crawler = self._create_crawler(target_links=10)
        # Make queue always non-empty to simulate continuous work
        self.mock_priority_queue_instance.empty.return_value = False
        self.mock_priority_queue_instance.get = AsyncMock(return_value=(0.5, 1, "https://example.com/"))
        # Make fetch_single work but not add to collected_urls to prolong loop
        mock_crawl_result_no_md = MagicMock(spec=RealCrawlResult, success=True, markdown=None)
        self.mock_fetch_single.return_value = mock_crawl_result_no_md

        with patch(f'{ASYNCIO_PATH}.wait_for', side_effect=asyncio.TimeoutError("Global timeout test")):
            collected_urls = await crawler.run_crawl()

        self.assertEqual(len(collected_urls), 0)
        mock_crawler_logger_instance.error.assert_any_call("Global crawl timeout (1800s) reached.")
        self.assertTrue(crawler._user_abort) # Abort should be signaled

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)