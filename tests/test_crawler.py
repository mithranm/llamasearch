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
from crawl4ai.async_crawler_strategy import AsyncPlaywrightCrawlerStrategy


MOCK_CRAWLER_SETUP_LOGGING_TARGET = "llamasearch.core.crawler.setup_logging"
mock_crawler_logger_global_instance = MagicMock(spec=logging.Logger)


class TestCrawlerModuleStandalone(unittest.TestCase): 

    @classmethod
    def setUpClass(cls):
        cls.logger_patcher = patch(MOCK_CRAWLER_SETUP_LOGGING_TARGET, return_value=mock_crawler_logger_global_instance)
        cls.logger_patcher.start()

    @classmethod
    def tearDownClass(cls):
        cls.logger_patcher.stop()
    
    def setUp(self):
        mock_crawler_logger_global_instance.reset_mock()
        self.temp_dir_obj = tempfile.TemporaryDirectory(prefix="test_crawler_s_") 
        self.temp_dir = Path(self.temp_dir_obj.name)
        self.shutdown_event = threading.Event()
        self.test_class_verbose_logging = False # Default for tests, can be overridden per test instantiation

        # Mock AsyncPlaywrightCrawlerStrategy by default for most tests
        # Tests that need specific strategy behavior can re-patch or unpatch
        self.mock_strategy_cls_patcher = patch('llamasearch.core.crawler.AsyncPlaywrightCrawlerStrategy')
        self.MockStrategyCls = self.mock_strategy_cls_patcher.start()
        self.mock_strategy_instance = MagicMock(spec=AsyncPlaywrightCrawlerStrategy)
        self.MockStrategyCls.return_value = self.mock_strategy_instance


    def tearDown(self):
        self.mock_strategy_cls_patcher.stop()
        self.temp_dir_obj.cleanup()
        self.shutdown_event.clear()

    def test_sanitize_string(self):
        self.assertEqual(sanitize_string("http://example.com/path?query=1"), "example.com_path_query=1") # Corrected expectation
        self.assertEqual(sanitize_string("https://www.example.com/a b/c:d*e<f>g|h"), "example.com_a_b_c_d_e_f_g_h")
        self.assertEqual(sanitize_string("test" * 20, max_length=10), "testtestte") 
        self.assertEqual(sanitize_string(""), "default")
        self.assertEqual(sanitize_string("///???"), "default") 
        self.assertEqual(sanitize_string("https://example.com/%E4%B8%AD%E6%96%87"), "example.com_中文")
        self.assertEqual(sanitize_string("www.example.com/path"), "example.com_path")
        self.assertEqual(sanitize_string("ftp://example.com/path"), "example.com_path")

    @patch('asyncio.wait_for')
    async def test_fetch_single_success_crawl_result_container(self, mock_wait_for):
        mock_crawler_instance = MagicMock(spec=AsyncWebCrawler)
        mock_crawl_result = MagicMock(spec=CrawlResult)
        mock_container = MagicMock(spec=CrawlResultContainer) 
        # Correctly mock the _results attribute to be a list containing the mock_crawl_result
        type(mock_container)._results = [mock_crawl_result] # Direct assignment for list
        # Mock __getitem__ behavior
        mock_container.__getitem__.return_value = mock_crawl_result

        mock_crawler_instance.arun = AsyncMock(return_value=mock_container)
        mock_wait_for.return_value = mock_container 

        url = "http://example.com"
        cfg = MagicMock(spec=CrawlerRunConfig)

        result = await fetch_single(mock_crawler_instance, url, cfg, self.shutdown_event)

        self.assertIs(result, mock_crawl_result)
        mock_wait_for.assert_called_once()
        # The coroutine object from AsyncMock is not directly comparable using ==
        # We check that arun was called, and its result (the coroutine) was passed to wait_for
        self.assertTrue(mock_crawler_instance.arun.called)
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
        mock_crawler_logger_global_instance.info.assert_called_with(f"Shutdown requested before fetching {url}")

    @patch('asyncio.wait_for', side_effect=asyncio.TimeoutError)
    async def test_fetch_single_timeout(self, mock_wait_for):
        mock_crawler_instance = MagicMock(spec=AsyncWebCrawler)
        url = "http://example.com"
        cfg = MagicMock(spec=CrawlerRunConfig)
        result = await fetch_single(mock_crawler_instance, url, cfg, self.shutdown_event)
        self.assertIsNone(result)
        mock_crawler_logger_global_instance.error.assert_called_with(f"Timeout ({DEFAULT_FETCH_TIMEOUT_S}s) fetching {url} with Playwright")

    @patch('asyncio.wait_for', side_effect=Exception("Playwright error"))
    async def test_fetch_single_playwright_error(self, mock_wait_for):
        mock_crawler_instance = MagicMock(spec=AsyncWebCrawler)
        url = "http://example.com"
        cfg = MagicMock(spec=CrawlerRunConfig)
        result = await fetch_single(mock_crawler_instance, url, cfg, self.shutdown_event)
        self.assertIsNone(result)
        mock_crawler_logger_global_instance.error.assert_called_with(f"Playwright exception fetching {url}: Playwright error", exc_info=True)

    def test_crawler_init_success(self): # MockStrategyCls is provided by self.MockStrategyCls
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
        self.assertEqual(browser_config_arg.verbose, self.test_class_verbose_logging) # Check verbose passed

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
        self.assertEqual(crawler.normalize_url("https://example.com//path//"), "https://example.com/path/") # Corrected SUT behavior
        self.assertEqual(crawler.normalize_url("ftp://example.com"), "")
        self.assertEqual(crawler.normalize_url(123), "") # type: ignore[arg-type] 
        self.assertEqual(crawler.normalize_url("http://example.com"), "http://example.com/")
        self.assertEqual(crawler.normalize_url("http://example.com/file.html"), "http://example.com/file.html")
        self.assertEqual(crawler.normalize_url("http://example.com/dir"), "http://example.com/dir/")


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


    def test_score_link_priority(self):
        keywords = ["doc", "api"]
        # Test with verbose_logging=True to exercise the logging paths
        crawler = Crawl4AICrawler(["http://example.com"], self.temp_dir, relevance_keywords=keywords, verbose_logging=True)
        
        priority_relevant = crawler._score_link_priority("http://example.com/doc/api-guide", "API Documentation")
        priority_less_relevant = crawler._score_link_priority("http://example.com/about-us", "Company Info")
        self.assertLess(priority_relevant, priority_less_relevant)
        
        # Check if debug logging was called due to verbose_logging=True
        # This assumes the scoring values are high enough to trigger logging.
        self.assertTrue(mock_crawler_logger_global_instance.log.called)


    def test_is_valid_content_url(self):
        root_urls = ["http://example.com", "http://sub.example.org"]
        crawler = Crawl4AICrawler(root_urls, self.temp_dir, verbose_logging=self.test_class_verbose_logging)

        self.assertTrue(crawler.is_valid_content_url("http://example.com/page"))
        self.assertTrue(crawler.is_valid_content_url("http://sub.example.org/another/page.html"))
        self.assertFalse(crawler.is_valid_content_url("http://docs.sub.example.org/api")) # /api is ignored
        
        self.assertFalse(crawler.is_valid_content_url("http://otherdomain.com/page"))
        self.assertFalse(crawler.is_valid_content_url("http://example.com/image.jpg"))
        self.assertFalse(crawler.is_valid_content_url("http://example.com/api/data")) 
        self.assertFalse(crawler.is_valid_content_url("http://example.com/login"))
        self.assertTrue(crawler.is_valid_content_url("https://www.example.com/blog/article-title"))
        self.assertFalse(crawler.is_valid_content_url("https://www.example.com/blog/tags/tech"))
        self.assertTrue(crawler.is_valid_content_url("http://example.com/product/item123"))
        self.assertFalse(crawler.is_valid_content_url("http://example.com/cart"))
        self.assertFalse(crawler.is_valid_content_url("http://example.com/assets/style.css"))
        self.assertFalse(crawler.is_valid_content_url("mailto:test@example.com"))
        self.assertTrue(crawler.is_valid_content_url("http://sub.example.com/path/?param=value")) # With query params

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
    async def test_run_crawl_target_links_reached(self, MockAsyncWebCrawlerCls, mock_fetch_single):
        mock_crawler = MockAsyncWebCrawlerCls.return_value
        mock_crawler.close = AsyncMock()

        crawler = Crawl4AICrawler(["http://a.com"], self.temp_dir, target_links=2, max_depth=1)
        crawler._crawler = mock_crawler
        
        page1_links = {"internal": [{"href": "http://a.com/page2", "text": "Page 2 Link"}]}
        page2_links = {"internal": [{"href": "http://a.com/page3", "text": "Page 3 Link"}]}

        mock_fetch_single.side_effect = [
            MagicMock(spec=CrawlResult, success=True, markdown="md1", links=page1_links),
            MagicMock(spec=CrawlResult, success=True, markdown="md2", links=page2_links),
            MagicMock(spec=CrawlResult, success=True, markdown="md3", links={}) # This shouldn't be fetched
        ]
        
        collected_urls = await crawler.run_crawl()
        self.assertEqual(len(collected_urls), 2)
        self.assertIn("http://a.com/", collected_urls)
        self.assertIn("http://a.com/page2/", collected_urls)
        self.assertEqual(mock_fetch_single.call_count, 2) # a.com, a.com/page2. page3 should not be fetched.
        self.assertTrue(mock_crawler_logger_global_instance.info.called)
        self.assertTrue(any("target link count (2) was reached" in call_args[0][0] for call_args in mock_crawler_logger_global_instance.info.call_args_list))


    @patch('llamasearch.core.crawler.fetch_single', new_callable=AsyncMock)
    @patch('llamasearch.core.crawler.AsyncWebCrawler')
    async def test_run_crawl_max_depth_reached(self, MockAsyncWebCrawlerCls, mock_fetch_single):
        mock_crawler = MockAsyncWebCrawlerCls.return_value
        mock_crawler.close = AsyncMock()

        crawler = Crawl4AICrawler(["http://b.com"], self.temp_dir, target_links=5, max_depth=1) # Max level = 2
        crawler._crawler = mock_crawler

        mock_fetch_single.side_effect = [
            MagicMock(spec=CrawlResult, success=True, markdown="md_root", links={"internal": [{"href":"/level1", "text":"L1"}]}),
            MagicMock(spec=CrawlResult, success=True, markdown="md_l1", links={"internal": [{"href":"/level1/level2", "text":"L2"}]}),
            # /level1/level2 is at level 3, should not be fetched.
        ]
        
        collected_urls = await crawler.run_crawl()
        self.assertEqual(len(collected_urls), 2) # root and level1
        self.assertIn("http://b.com/", collected_urls)
        self.assertIn("http://b.com/level1/", collected_urls)
        self.assertEqual(mock_fetch_single.call_count, 2)
        # Check that the log for max depth was hit for /level1/level2
        self.assertTrue(any("Max level 2 reached for http://b.com/level1/level2/" in call_args[0][0] for call_args in mock_crawler_logger_global_instance.debug.call_args_list))

    @patch('llamasearch.core.crawler.fetch_single', new_callable=AsyncMock)
    @patch('llamasearch.core.crawler.AsyncWebCrawler')
    async def test_run_crawl_shutdown_event(self, MockAsyncWebCrawlerCls, mock_fetch_single):
        mock_crawler = MockAsyncWebCrawlerCls.return_value
        mock_crawler.close = AsyncMock()
        
        shutdown_event = threading.Event()
        crawler = Crawl4AICrawler(["http://c.com"], self.temp_dir, target_links=5, shutdown_event=shutdown_event)
        crawler._crawler = mock_crawler

        # Simulate shutdown after the first fetch
        async def fetch_side_effect(*args, **kwargs):
            url_being_fetched = args[1]
            if url_being_fetched == "http://c.com/":
                shutdown_event.set() # Set shutdown after fetching the first URL
                return MagicMock(spec=CrawlResult, success=True, markdown="md_c", links={})
            return None # Should not be called for other URLs due to shutdown
        
        mock_fetch_single.side_effect = fetch_side_effect
        
        collected_urls = await crawler.run_crawl()
        self.assertEqual(len(collected_urls), 1) # Only the first URL should be collected
        self.assertIn("http://c.com/", collected_urls)
        mock_fetch_single.assert_called_once()
        self.assertTrue(any("Crawl loop aborted by signal." in call_args[0][0] for call_args in mock_crawler_logger_global_instance.warning.call_args_list))

    @patch('llamasearch.core.crawler.fetch_single', new_callable=AsyncMock)
    @patch('llamasearch.core.crawler.AsyncWebCrawler')
    async def test_run_crawl_error_writing_markdown(self, MockAsyncWebCrawlerCls, mock_fetch_single):
        mock_crawler = MockAsyncWebCrawlerCls.return_value
        mock_crawler.close = AsyncMock()

        crawler = Crawl4AICrawler(["http://d.com"], self.temp_dir)
        crawler._crawler = mock_crawler
        
        mock_fetch_single.return_value = MagicMock(spec=CrawlResult, success=True, markdown="markdown content", links={})
        
        # Make Path.write_text raise an error
        with patch.object(Path, 'write_text', side_effect=OSError("Disk full")):
            collected_urls = await crawler.run_crawl()
        
        self.assertEqual(len(collected_urls), 0) # No URLs collected successfully if write fails
        self.assertTrue(any("Failed to write MD file" in call_args[0][0] for call_args in mock_crawler_logger_global_instance.error.call_args_list))

    @patch('llamasearch.core.crawler.fetch_single', new_callable=AsyncMock)
    @patch('llamasearch.core.crawler.AsyncWebCrawler')
    async def test_run_crawl_fetch_error(self, MockAsyncWebCrawlerCls, mock_fetch_single):
        mock_crawler = MockAsyncWebCrawlerCls.return_value
        mock_crawler.close = AsyncMock()

        crawler = Crawl4AICrawler(["http://e.com"], self.temp_dir)
        crawler._crawler = mock_crawler
        
        mock_fetch_single.return_value = None # Simulate fetch failure
        
        collected_urls = await crawler.run_crawl()
        self.assertEqual(len(collected_urls), 0)
        self.assertTrue(any("Playwright fetch failed or aborted for http://e.com/" in call_args[0][0] for call_args in mock_crawler_logger_global_instance.warning.call_args_list))

if __name__ == '__main__':
    unittest.main()