# tests/test_crawler.py
import unittest
from unittest.mock import MagicMock, patch, AsyncMock
import asyncio
import tempfile
from pathlib import Path
import threading
import logging
import json

from llamasearch.core.crawler import (
    Crawl4AICrawler,
    fetch_single,
    sanitize_string,
    DEFAULT_PAGE_TIMEOUT_MS,
    DEFAULT_FETCH_TIMEOUT_S
)
from crawl4ai import (
    AsyncWebCrawler,
    CrawlResult,
    BrowserConfig,
    CrawlerRunConfig,
    CacheMode
)
from crawl4ai.models import CrawlResultContainer

# Patch the logger instance directly in the crawler module
MOCK_CRAWLER_LOGGER_INSTANCE_TARGET = "llamasearch.core.crawler.logger"


class TestCrawlerModuleStandalone(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.module_logger_patcher = patch(MOCK_CRAWLER_LOGGER_INSTANCE_TARGET, spec=logging.Logger)
        cls.mock_crawler_module_logger_direct = cls.module_logger_patcher.start()
        cls.test_class_verbose_logging = False # For tests that don't explicitly set it

    @classmethod
    def tearDownClass(cls):
        cls.module_logger_patcher.stop()

    def setUp(self):
        self.mock_crawler_module_logger_direct.reset_mock()
        for method_name in ['error', 'info', 'warning', 'debug', 'critical', 'log']: 
            if hasattr(self.mock_crawler_module_logger_direct, method_name):
                getattr(self.mock_crawler_module_logger_direct, method_name).reset_mock()

        self.temp_dir_obj = tempfile.TemporaryDirectory(prefix="test_crawler_s_")
        self.temp_dir = Path(self.temp_dir_obj.name)
        self.shutdown_event = threading.Event()

        self.strategy_patcher = patch('llamasearch.core.crawler.AsyncPlaywrightCrawlerStrategy')
        self.MockStrategyCls = self.strategy_patcher.start()


    def tearDown(self):
        self.strategy_patcher.stop()
        self.temp_dir_obj.cleanup()
        self.shutdown_event.clear()

    def test_sanitize_string(self):
        self.assertEqual(sanitize_string("http://example.com/path?query=1"), "example.com_path_query_1")
        self.assertEqual(sanitize_string("https://www.example.com/a b/c:d*e<f>g|h="), "example.com_a_b_c_d_e_f_g_h")
        self.assertEqual(sanitize_string("test" * 20, max_length=10), "testtestte")
        self.assertEqual(sanitize_string(""), "default")
        self.assertEqual(sanitize_string("///???==="), "default")
        self.assertEqual(sanitize_string("https://example.com/%E4%B8%AD%E6%96%87"), "example.com_中文")
        self.assertEqual(sanitize_string("www.example.com/path"), "example.com_path")
        self.assertEqual(sanitize_string("ftp://example.com/path"), "example.com_path")

    @patch('asyncio.wait_for')
    async def test_fetch_single_success_crawl_result_container(self, mock_wait_for):
        mock_crawler_instance = MagicMock(spec=AsyncWebCrawler)
        mock_crawl_result = MagicMock(spec=CrawlResult)
        mock_container = MagicMock(spec=CrawlResultContainer)
        mock_container._results = [mock_crawl_result]
        mock_container.__getitem__.return_value = mock_crawl_result

        mock_crawler_instance.arun = AsyncMock(return_value=mock_container)
        mock_wait_for.return_value = mock_container

        url = "http://example.com"
        cfg = MagicMock(spec=CrawlerRunConfig)

        result = await fetch_single(mock_crawler_instance, url, cfg, self.shutdown_event)

        self.assertIs(result, mock_crawl_result)
        mock_wait_for.assert_called_once()
        self.assertIs(mock_wait_for.call_args[0][0], mock_crawler_instance.arun.return_value)
        mock_crawler_instance.arun.assert_called_once_with(url=url, config=cfg)
        self.assertEqual(mock_wait_for.call_args[1]['timeout'], DEFAULT_FETCH_TIMEOUT_S)


    @patch('asyncio.wait_for')
    async def test_fetch_single_success_direct_crawl_result(self, mock_wait_for):
        mock_crawler_instance = MagicMock(spec=AsyncWebCrawler)
        mock_crawl_result = MagicMock(spec=CrawlResult)

        mock_crawler_instance.arun = AsyncMock(return_value=mock_crawl_result)
        mock_wait_for.return_value = mock_crawl_result

        url = "http://example.com"
        cfg = MagicMock(spec=CrawlerRunConfig)

        result = await fetch_single(mock_crawler_instance, url, cfg, self.shutdown_event)
        self.assertIs(result, mock_crawl_result)

    @patch('asyncio.wait_for')
    async def test_fetch_single_success_async_generator(self, mock_wait_for):
        mock_crawler_instance = MagicMock(spec=AsyncWebCrawler)
        mock_crawl_result = MagicMock(spec=CrawlResult)

        async def async_gen_mock_func():
            yield mock_crawl_result

        mock_async_gen_obj = async_gen_mock_func()
        mock_async_gen_obj.aclose = AsyncMock()

        mock_crawler_instance.arun = AsyncMock(return_value=mock_async_gen_obj)
        mock_wait_for.return_value = mock_async_gen_obj


        url = "http://example.com"
        cfg = MagicMock(spec=CrawlerRunConfig)

        result = await fetch_single(mock_crawler_instance, url, cfg, self.shutdown_event)
        self.assertIs(result, mock_crawl_result)
        mock_async_gen_obj.aclose.assert_called_once()


    async def test_fetch_single_shutdown_before_fetch(self):
        mock_crawler_instance = MagicMock(spec=AsyncWebCrawler)
        url = "http://example.com"
        cfg = MagicMock(spec=CrawlerRunConfig)
        self.shutdown_event.set()
        result = await fetch_single(mock_crawler_instance, url, cfg, self.shutdown_event)
        self.assertIsNone(result)
        self.mock_crawler_module_logger_direct.info.assert_called_with(f"Shutdown requested before fetching {url}")

    @patch('asyncio.wait_for', side_effect=asyncio.TimeoutError)
    async def test_fetch_single_timeout(self, mock_wait_for):
        mock_crawler_instance = MagicMock(spec=AsyncWebCrawler)
        url = "http://example.com"
        cfg = MagicMock(spec=CrawlerRunConfig)
        result = await fetch_single(mock_crawler_instance, url, cfg, self.shutdown_event)
        self.assertIsNone(result)
        self.mock_crawler_module_logger_direct.error.assert_called_with(f"Timeout ({DEFAULT_FETCH_TIMEOUT_S}s) fetching {url} with Playwright")

    @patch('asyncio.wait_for', side_effect=Exception("Playwright error"))
    async def test_fetch_single_playwright_error(self, mock_wait_for):
        mock_crawler_instance = MagicMock(spec=AsyncWebCrawler)
        url = "http://example.com"
        cfg = MagicMock(spec=CrawlerRunConfig)
        result = await fetch_single(mock_crawler_instance, url, cfg, self.shutdown_event)
        self.assertIsNone(result)
        self.mock_crawler_module_logger_direct.error.assert_called_with(f"Playwright exception fetching {url}: Playwright error", exc_info=True)


    def test_crawler_init_success(self):
        root_urls = ["http://example.com"]
        crawler = Crawl4AICrawler(root_urls, self.temp_dir, target_links=10, max_depth=1, shutdown_event=self.shutdown_event, verbose_logging=self.test_class_verbose_logging)

        self.assertEqual(crawler.root_urls, ["http://example.com/"])
        self.assertEqual(crawler.target_links, 10)
        self.assertEqual(crawler.max_crawl_level, 2)
        self.assertEqual(crawler.base_crawl_dir, self.temp_dir)
        self.assertTrue((self.temp_dir / "raw").is_dir())

        self.MockStrategyCls.assert_called_once()
        browser_config_arg = self.MockStrategyCls.call_args[1]['config']
        self.assertIsInstance(browser_config_arg, BrowserConfig)
        self.assertEqual(browser_config_arg.browser_type, "chromium")
        self.assertTrue(browser_config_arg.headless)

        self.assertIsInstance(crawler._run_cfg, CrawlerRunConfig)
        self.assertEqual(crawler._run_cfg.cache_mode, CacheMode.BYPASS)
        self.assertEqual(crawler._run_cfg.page_timeout, DEFAULT_PAGE_TIMEOUT_MS + 10000)


    def test_crawler_init_no_root_urls(self):
        with self.assertRaisesRegex(ValueError, "At least one root URL must be provided."):
            Crawl4AICrawler([], self.temp_dir, verbose_logging=self.test_class_verbose_logging)

    def test_crawler_init_strategy_fail(self):
        self.MockStrategyCls.side_effect = RuntimeError("Strategy init failed")
        with self.assertRaisesRegex(RuntimeError, "Could not initialize Playwright crawler strategy."):
            Crawl4AICrawler(["http://example.com"], self.temp_dir, verbose_logging=self.test_class_verbose_logging)

    def test_normalize_url(self):
        crawler = Crawl4AICrawler(["http://example.com"], self.temp_dir, verbose_logging=self.test_class_verbose_logging)

        self.assertEqual(crawler.normalize_url("example.com"), "https://example.com/")
        self.assertEqual(crawler.normalize_url("http://example.com/path#frag"), "http://example.com/path/")
        self.assertEqual(crawler.normalize_url("https://example.com/path?q=1"), "https://example.com/path?q=1")
        self.assertEqual(crawler.normalize_url("https://example.com//path//"), "https://example.com/path/")
        self.assertEqual(crawler.normalize_url("ftp://example.com"), "")
        self.assertEqual(crawler.normalize_url(123), "") #type: ignore
        self.assertEqual(crawler.normalize_url("http://example.com"), "http://example.com/")
        self.assertEqual(crawler.normalize_url("http://example.com/file.html"), "http://example.com/file.html")
        self.assertEqual(crawler.normalize_url("http://example.com/dir"), "http://example.com/dir/")
        self.assertEqual(crawler.normalize_url("http://example.com/path?b=2&a=1"), "http://example.com/path?a=1&b=2")


    def test_generate_key(self):
        crawler = Crawl4AICrawler(["http://example.com"], self.temp_dir, verbose_logging=self.test_class_verbose_logging)
        key1 = crawler._generate_key("https://example.com/")
        key2 = crawler._generate_key("https://example.com/")
        self.assertEqual(key1, key2)
        self.assertNotEqual(crawler._generate_key("https://example.org/"), key1)
        self.assertEqual(len(key1), 16)

    @patch('llamasearch.core.crawler.os.replace')
    def test_load_save_reverse_lookup(self, mock_os_replace):
        crawl_dir_for_lookup = self.temp_dir / "lookup_test_data"

        crawler = Crawl4AICrawler(["http://example.com"], crawl_dir_for_lookup, verbose_logging=self.test_class_verbose_logging)

        self.assertEqual(crawler._reverse_lookup, {})

        crawler._reverse_lookup = {"key1": "url1", "key2": "url2"}
        crawler._save_reverse_lookup()

        temp_file_path = crawl_dir_for_lookup / "reverse_lookup.json.tmp"
        final_file_path = crawl_dir_for_lookup / "reverse_lookup.json"
        mock_os_replace.assert_called_once_with(temp_file_path, final_file_path)

        final_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(final_file_path, 'w', encoding='utf-8') as f:
            json.dump({"key1": "url1", "key2": "url2"}, f, indent=2, ensure_ascii=False)
        self.assertTrue(final_file_path.exists())


        crawler2 = Crawl4AICrawler(["http://example.com"], crawl_dir_for_lookup, verbose_logging=self.test_class_verbose_logging)
        self.assertEqual(crawler2._reverse_lookup, {"key1": "url1", "key2": "url2"})

    def test_is_valid_content_url(self):
        root_urls = ["http://example.com", "http://sub.example.org"]
        crawler = Crawl4AICrawler(root_urls, self.temp_dir, verbose_logging=self.test_class_verbose_logging)

        self.assertTrue(crawler.is_valid_content_url("http://example.com/page"))
        self.assertTrue(crawler.is_valid_content_url("http://sub.example.org/another/page.html"))
        self.assertTrue(crawler.is_valid_content_url("http://docs.sub.example.org/api/reference")) 
        self.assertTrue(crawler.is_valid_content_url("http://example.com/api/docs")) 

        self.assertFalse(crawler.is_valid_content_url("http://otherdomain.com/page"))
        self.assertFalse(crawler.is_valid_content_url("http://example.com/image.jpg"))
        self.assertTrue(crawler.is_valid_content_url("http://example.com/api/data"))
        self.assertFalse(crawler.is_valid_content_url("http://example.com/login"))


    @patch('llamasearch.core.crawler.fetch_single', new_callable=AsyncMock)
    @patch('llamasearch.core.crawler.AsyncWebCrawler')
    async def test_run_crawl_success_flow(self, MockAsyncWebCrawlerCls, mock_fetch_single_func):
        mock_actual_crawler_instance = MockAsyncWebCrawlerCls.return_value
        mock_actual_crawler_instance.close = AsyncMock()

        root_urls = ["http://example.com/"]
        crawler = Crawl4AICrawler(root_urls, self.temp_dir, target_links=1, max_depth=0, shutdown_event=self.shutdown_event, verbose_logging=self.test_class_verbose_logging)

        crawler._crawler = mock_actual_crawler_instance

        result1_markdown = "# Page 1\nContent 1"
        crawl_res1 = MagicMock(spec=CrawlResult, success=True, markdown=result1_markdown, links={})
        mock_fetch_single_func.return_value = crawl_res1

        collected = await crawler.run_crawl()

        self.assertEqual(len(collected), 1)
        self.assertIn("http://example.com/", collected)
        mock_fetch_single_func.assert_called_once_with(
            mock_actual_crawler_instance, "http://example.com/", crawler._run_cfg, crawler._shutdown_event, DEFAULT_FETCH_TIMEOUT_S
        )
        mock_actual_crawler_instance.close.assert_called_once()

        key1 = crawler._generate_key("http://example.com/")
        md_file_path = crawler.raw_markdown_dir / f"{key1}.md"
        self.assertTrue(md_file_path.exists())
        self.assertEqual(md_file_path.read_text(encoding='utf-8'), result1_markdown)
        self.assertTrue(crawler.reverse_lookup_path.exists())

    @patch('llamasearch.core.crawler.fetch_single', new_callable=AsyncMock)
    @patch('llamasearch.core.crawler.AsyncWebCrawler')
    async def test_run_crawl_target_links_reached(self, MockAsyncWebCrawlerCls, mock_fetch_single_func):
        mock_actual_crawler_instance = MockAsyncWebCrawlerCls.return_value
        mock_actual_crawler_instance.close = AsyncMock()

        crawler = Crawl4AICrawler(["http://example.com"], self.temp_dir, target_links=1, max_depth=1, verbose_logging=self.test_class_verbose_logging)
        crawler._crawler = mock_actual_crawler_instance

        mock_fetch_single_func.return_value = MagicMock(spec=CrawlResult, success=True, markdown="content", links={})
        await crawler.run_crawl()
        self.assertEqual(mock_fetch_single_func.call_count, 1)

    @patch('llamasearch.core.crawler.fetch_single', new_callable=AsyncMock)
    @patch('llamasearch.core.crawler.AsyncWebCrawler')
    async def test_run_crawl_max_depth_reached(self, MockAsyncWebCrawlerCls, mock_fetch_single_func):
        mock_actual_crawler_instance = MockAsyncWebCrawlerCls.return_value
        mock_actual_crawler_instance.close = AsyncMock()

        crawler = Crawl4AICrawler(["http://example.com"], self.temp_dir, target_links=5, max_depth=0, verbose_logging=self.test_class_verbose_logging)
        crawler._crawler = mock_actual_crawler_instance

        crawl_res_root = MagicMock(spec=CrawlResult, success=True, markdown="root content", links={"internal": ["/link1"]})
        mock_fetch_single_func.return_value = crawl_res_root

        await crawler.run_crawl()
        self.assertEqual(mock_fetch_single_func.call_count, 1)

    @patch('llamasearch.core.crawler.fetch_single', new_callable=AsyncMock)
    @patch('llamasearch.core.crawler.AsyncWebCrawler')
    async def test_run_crawl_shutdown_event(self, MockAsyncWebCrawlerCls, mock_fetch_single_func):
        mock_actual_crawler_instance = MockAsyncWebCrawlerCls.return_value
        mock_actual_crawler_instance.close = AsyncMock()

        crawler = Crawl4AICrawler(["http://example.com"], self.temp_dir, target_links=5, max_depth=1, shutdown_event=self.shutdown_event, verbose_logging=self.test_class_verbose_logging)
        crawler._crawler = mock_actual_crawler_instance

        mock_fetch_single_func.side_effect = lambda *args, **kwargs: self.shutdown_event.set() or None

        await crawler.run_crawl()
        self.assertTrue(self.shutdown_event.is_set())
        self.mock_crawler_module_logger_direct.warning.assert_any_call("Crawl loop aborted by signal.")

    @patch('llamasearch.core.crawler.fetch_single', new_callable=AsyncMock)
    @patch('llamasearch.core.crawler.AsyncWebCrawler')
    async def test_run_crawl_fetch_error(self, MockAsyncWebCrawlerCls, mock_fetch_single_func):
        mock_actual_crawler_instance = MockAsyncWebCrawlerCls.return_value
        mock_actual_crawler_instance.close = AsyncMock()
        crawler = Crawl4AICrawler(["http://example.com"], self.temp_dir, verbose_logging=self.test_class_verbose_logging)
        crawler._crawler = mock_actual_crawler_instance
        mock_fetch_single_func.return_value = None

        collected_urls = await crawler.run_crawl()
        self.assertEqual(len(collected_urls), 0)
        self.mock_crawler_module_logger_direct.warning.assert_any_call("Playwright fetch failed or aborted for http://example.com/")

    @patch('llamasearch.core.crawler.fetch_single', new_callable=AsyncMock)
    @patch('llamasearch.core.crawler.AsyncWebCrawler')
    async def test_run_crawl_error_writing_markdown(self, MockAsyncWebCrawlerCls, mock_fetch_single_func):
        mock_actual_crawler_instance = MockAsyncWebCrawlerCls.return_value
        mock_actual_crawler_instance.close = AsyncMock()
        crawler = Crawl4AICrawler(["http://example.com"], self.temp_dir, verbose_logging=self.test_class_verbose_logging)
        crawler._crawler = mock_actual_crawler_instance

        mock_fetch_single_func.return_value = MagicMock(spec=CrawlResult, success=True, markdown="content", links={})

        with patch.object(Path, 'write_text', side_effect=OSError("Disk full")):
            collected_urls = await crawler.run_crawl()

        self.assertEqual(len(collected_urls), 0)
        error_found = False
        for call_args in self.mock_crawler_module_logger_direct.error.call_args_list:
            if f"Failed to write MD file {crawler.raw_markdown_dir / (crawler._generate_key('http://example.com/') + '.md')}: Disk full" in call_args[0][0]:
                error_found = True
                break
        self.assertTrue(error_found, "Expected error log for failing to write MD file not found.")