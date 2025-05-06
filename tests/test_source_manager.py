# tests/test_source_manager.py
import unittest
from unittest.mock import MagicMock, patch, mock_open
from pathlib import Path as RealPath
import json
import time
import logging
import tempfile
from typing import Optional, List # Import Optional and List

from llamasearch.core.source_manager import (
    _SourceManagementMixin,
    DEFAULT_MAX_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_MIN_CHUNK_SIZE_FILTER,
)
from llamasearch.core.embedder import EnhancedEmbedder
from chromadb import Collection as ChromaCollection
# Corrected type for GetResult items, and GetResult itself
from chromadb.api.types import Metadata, GetResult as ChromaGetResultType 
# Removed unused Embedding and Document imports
from llamasearch.core.bm25 import WhooshBM25Retriever
from whoosh import index as whoosh_index

# Import the module containing the logger to be patched
from llamasearch.core import source_manager as source_manager_module


class DummyLLMSearchWithMixin(_SourceManagementMixin):
    def __init__(self):
        self.embedder: MagicMock = MagicMock(spec=EnhancedEmbedder)
        self.chroma_collection: MagicMock = MagicMock(spec=ChromaCollection)
        self.bm25: MagicMock = MagicMock(spec=WhooshBM25Retriever)
        self.bm25.ix = MagicMock(spec=whoosh_index.FileIndex)
        
        # Setup for BM25 writer mocking
        self.mock_bm25_actual_writer = MagicMock(name="actual_bm25_writer")
        mock_writer_cm = MagicMock(name="bm25_writer_context_manager")
        mock_writer_cm.__enter__.return_value = self.mock_bm25_actual_writer
        mock_writer_cm.__exit__.return_value = None
        self.bm25.ix.writer.return_value = mock_writer_cm

        self._shutdown_event: MagicMock = MagicMock()
        self._shutdown_event.is_set.return_value = False
        self._reverse_lookup: dict = {}
        self.max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE
        self.chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
        self.min_chunk_size_filter: int = DEFAULT_MIN_CHUNK_SIZE_FILTER
        self.verbose: bool = False
        self.debug: bool = False
        self.storage_dir: RealPath = RealPath("/fake/storage_dir_for_mixin")


class TestSourceManagementMixin(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.logger_patcher = patch.object(source_manager_module, 'logger', spec=logging.Logger)
        cls.mock_sut_logger_object = cls.logger_patcher.start()

    @classmethod
    def tearDownClass(cls):
        cls.logger_patcher.stop()

    def setUp(self):
        self.mock_sut_logger_object.reset_mock()

        self.mixin_instance = DummyLLMSearchWithMixin()
        
        self.mixin_instance.mock_bm25_actual_writer.reset_mock()
        self.mixin_instance.bm25.ix.writer.reset_mock() 
        mock_writer_cm = MagicMock(name="bm25_writer_context_manager_setup")
        mock_writer_cm.__enter__.return_value = self.mixin_instance.mock_bm25_actual_writer
        mock_writer_cm.__exit__.return_value = None
        self.mixin_instance.bm25.ix.writer.return_value = mock_writer_cm

        self.mixin_instance.embedder.reset_mock()
        self.mixin_instance.chroma_collection.reset_mock()
        if self.mixin_instance.bm25.ix:
            self.mixin_instance.bm25.ix.reset_mock()

        self.mixin_instance._shutdown_event.reset_mock()
        self.mixin_instance._shutdown_event.is_set.return_value = False
        self.mixin_instance._reverse_lookup = {}

        self.mock_data_manager_patcher = patch("llamasearch.core.source_manager.data_manager")
        self.mock_data_manager = self.mock_data_manager_patcher.start()
        self.mock_data_manager.get_data_paths.return_value = {"crawl_data": "/fake/crawl_data_path_default"}

        self.path_mocks_created: dict[str, MagicMock] = {}
        self.mock_path_patcher = patch("llamasearch.core.source_manager.Path")
        self.MockPathClass = self.mock_path_patcher.start()
        self.MockPathClass.side_effect = self._path_constructor_side_effect

        self.mock_open_patcher = patch("builtins.open", new_callable=mock_open)
        self.mock_open_builtin = self.mock_open_patcher.start()
        self.mock_json_load_patcher = patch("json.load")
        self.mock_json_load = self.mock_json_load_patcher.start()
        self.mock_json_dump_patcher = patch("json.dump")
        self.mock_json_dump = self.mock_json_dump_patcher.start()
        self.mock_chunker_patcher = patch("llamasearch.core.source_manager.chunk_markdown_text")
        self.mock_chunk_markdown_text = self.mock_chunker_patcher.start()
        self.mock_hashlib_patcher = patch("hashlib.sha1")
        self.mock_sha1_constructor = self.mock_hashlib_patcher.start()
        mock_sha1_instance = MagicMock()
        mock_sha1_instance.hexdigest.return_value = "fakehash"
        self.mock_sha1_constructor.return_value = mock_sha1_instance
        self.mock_os_remove_patcher = patch("os.remove")
        self.mock_os_remove = self.mock_os_remove_patcher.start()
        self.temp_dir_obj = tempfile.TemporaryDirectory(prefix="test_source_manager_")
        self.temp_dir_path = RealPath(self.temp_dir_obj.name)

    def tearDown(self):
        self.mock_data_manager_patcher.stop()
        self.mock_path_patcher.stop()
        self.mock_open_patcher.stop()
        self.mock_json_load_patcher.stop()
        self.mock_json_dump_patcher.stop()
        self.mock_chunker_patcher.stop()
        self.mock_hashlib_patcher.stop()
        self.mock_os_remove_patcher.stop()
        self.temp_dir_obj.cleanup()

    def _path_constructor_side_effect(self, path_arg: str | RealPath) -> MagicMock:
        path_str = str(RealPath(str(path_arg)).resolve())
        if path_str not in self.path_mocks_created:
            mp = MagicMock(spec=RealPath, name=f"PathMock({path_str})")
            mp.__str__.return_value = path_str
            mp.as_posix.return_value = RealPath(path_str).as_posix()
            mp.resolve.return_value = mp
            mp.exists.return_value = False
            mp.is_file.return_value = False
            mp.is_dir.return_value = False
            mp.name = RealPath(path_str).name
            mp.stem = RealPath(path_str).stem
            mp.suffix = RealPath(path_str).suffix
            mp.parents = [self._path_constructor_side_effect(str(p)) for p in RealPath(path_str).parents]
            mp.__truediv__.side_effect = lambda other: self._path_constructor_side_effect(RealPath(path_str) / str(other))
            mp.read_text.side_effect = FileNotFoundError(f"[Mock] File not found: {path_str}")
            mock_stat_result = MagicMock()
            mock_stat_result.st_mtime = time.time()
            mp.stat.return_value = mock_stat_result
            mp.stat.side_effect = None
            mp.mkdir.return_value = None
            mp.rglob.return_value = []
            self.path_mocks_created[path_str] = mp
        return self.path_mocks_created[path_str]

    def _configure_path_mock(
        self, path_str: str, exists: bool = False, is_file: bool = False, is_dir: bool = False,
        read_text_content: str | None = None, read_text_error: Exception | None = None,
        mtime: float | None = None, rglob_return: list[MagicMock] | None = None,
        parents_override: list[str] | None = None,
    ) -> MagicMock:
        p_mock = self._path_constructor_side_effect(path_str)
        p_mock.exists.return_value = exists
        p_mock.is_file.return_value = is_file
        p_mock.is_dir.return_value = is_dir
        if read_text_content is not None:
            p_mock.read_text.return_value = read_text_content
            p_mock.read_text.side_effect = None
        elif read_text_error is not None:
            p_mock.read_text.side_effect = read_text_error
            p_mock.read_text.return_value = None
        if mtime is not None:
            p_mock.stat.return_value.st_mtime = mtime
        elif exists:
            p_mock.stat.side_effect = None
            if not hasattr(p_mock.stat.return_value, "st_mtime"):
                p_mock.stat.return_value.st_mtime = time.time()
        if rglob_return is not None:
            p_mock.rglob.return_value = rglob_return
        if parents_override is not None:
            p_mock.parents = [self._path_constructor_side_effect(str(parent_path)) for parent_path in parents_override]
        return p_mock

    def test_load_reverse_lookup_path_not_configured(self):
        self.mock_data_manager.get_data_paths.return_value = {}
        self.mixin_instance._load_reverse_lookup()
        self.assertEqual(self.mixin_instance._reverse_lookup, {})
        self.mock_sut_logger_object.warning.assert_called_with(
            "Crawl data path not configured, cannot load reverse lookup."
        )

    def test_load_reverse_lookup_file_exists(self):
        crawl_data_dir_str = str(self.temp_dir_path / "crawl_data_test_load")
        lookup_file_str = str(RealPath(crawl_data_dir_str) / "reverse_lookup.json")
        self.mock_data_manager.get_data_paths.return_value = {"crawl_data": crawl_data_dir_str}
        self._configure_path_mock(lookup_file_str, exists=True, is_file=True)
        self.mock_json_load.return_value = {"hash1": "url1"}
        self.mixin_instance._load_reverse_lookup()
        self.assertEqual(self.mixin_instance._reverse_lookup, {"hash1": "url1"})
        mock_lookup_file_path_obj = self._path_constructor_side_effect(lookup_file_str)
        self.mock_open_builtin.assert_called_once_with(mock_lookup_file_path_obj, "r", encoding="utf-8")
        self.mock_json_load.assert_called_once()
        self.mock_sut_logger_object.info.assert_any_call("Loaded URL reverse lookup (1 entries).")

    def test_load_reverse_lookup_file_not_exist(self):
        crawl_data_dir_str = str(self.temp_dir_path / "crawl_data_test_load_no_file")
        lookup_file_str = str(RealPath(crawl_data_dir_str) / "reverse_lookup.json")
        self.mock_data_manager.get_data_paths.return_value = {"crawl_data": crawl_data_dir_str}
        self._configure_path_mock(lookup_file_str, exists=False)
        self.mixin_instance._load_reverse_lookup()
        self.assertEqual(self.mixin_instance._reverse_lookup, {})
        self.mock_sut_logger_object.info.assert_any_call("URL reverse lookup file not found.")

    def test_load_reverse_lookup_json_error(self):
        crawl_data_dir_str = str(self.temp_dir_path / "crawl_data_test_load_json_err")
        lookup_file_str = str(RealPath(crawl_data_dir_str) / "reverse_lookup.json")
        self.mock_data_manager.get_data_paths.return_value = {"crawl_data": crawl_data_dir_str}
        self._configure_path_mock(lookup_file_str, exists=True, is_file=True)
        self.mock_json_load.side_effect = json.JSONDecodeError("err", "doc", 0)
        self.mixin_instance.debug = True
        self.mixin_instance._load_reverse_lookup()
        self.assertEqual(self.mixin_instance._reverse_lookup, {})
        self.mock_sut_logger_object.error.assert_called_with(
            "Error loading reverse lookup: err: line 1 column 1 (char 0)", exc_info=True
        )

    def test_save_reverse_lookup_path_not_configured(self):
        self.mock_data_manager.get_data_paths.return_value = {}
        self.mixin_instance._save_reverse_lookup()
        self.mock_json_dump.assert_not_called()
        self.mock_sut_logger_object.warning.assert_called_with(
            "Crawl data path not configured, cannot save reverse lookup."
        )

    def test_save_reverse_lookup_success(self):
        crawl_data_dir_str = str(self.temp_dir_path / "crawl_data_save_ok")
        lookup_file_str = str(RealPath(crawl_data_dir_str) / "reverse_lookup.json")
        self.mock_data_manager.get_data_paths.return_value = {"crawl_data": crawl_data_dir_str}
        mock_lookup_file = self._configure_path_mock(lookup_file_str)
        mock_parent_dir = self._configure_path_mock(crawl_data_dir_str)
        mock_lookup_file.parent = mock_parent_dir
        self.mixin_instance._reverse_lookup = {"hash1": "url1_saved"}
        self.mixin_instance._save_reverse_lookup()
        mock_parent_dir.mkdir.assert_called_once_with(parents=True, exist_ok=True)
        self.mock_open_builtin.assert_called_once_with(mock_lookup_file, "w", encoding="utf-8")
        self.mock_json_dump.assert_called_once()
        args, kwargs = self.mock_json_dump.call_args
        self.assertEqual(args[0], {"hash1": "url1_saved"})
        self.mock_sut_logger_object.info.assert_any_call("Saved URL reverse lookup (1 entries).")

    def test_save_reverse_lookup_io_error(self):
        crawl_data_dir_str = str(self.temp_dir_path / "crawl_data_save_err")
        lookup_file_str = str(RealPath(crawl_data_dir_str) / "reverse_lookup.json")
        self.mock_data_manager.get_data_paths.return_value = {"crawl_data": crawl_data_dir_str}
        self._configure_path_mock(lookup_file_str)
        self.mock_json_dump.side_effect = IOError("Disk full")
        self.mixin_instance.debug = True
        self.mixin_instance._save_reverse_lookup()
        self.mock_sut_logger_object.error.assert_called_with(
            "Error saving reverse lookup: Disk full", exc_info=True
        )

    def test_add_source_shutdown_event_set(self):
        self.mixin_instance._shutdown_event.is_set.return_value = True
        count, blocked = self.mixin_instance.add_source("/fake/path.md")
        self.assertEqual(count, 0)
        self.assertFalse(blocked)
        self.mock_sut_logger_object.warning.assert_called_with("Add source cancelled due to shutdown.")

    def test_add_source_path_not_found(self):
        source_path_str = "/fake/nonexistent.md"
        self._configure_path_mock(source_path_str, exists=False)
        count, blocked = self.mixin_instance.add_source(source_path_str)
        self.assertEqual(count, 0)
        self.assertFalse(blocked)
        mocked_path_obj = self._path_constructor_side_effect(source_path_str)
        self.mock_sut_logger_object.error.assert_called_with(f"Source path not found: {mocked_path_obj}")

    def test_add_source_unsupported_extension(self):
        source_path_str = "/fake/file.exe"
        self._configure_path_mock(source_path_str, exists=True, is_file=True)
        count, blocked = self.mixin_instance.add_source(source_path_str)
        self.assertEqual(count, 0)
        self.assertFalse(blocked)
        self.mock_sut_logger_object.info.assert_called_with("Skipping file file.exe: Unsupported extension '.exe'.")

    def test_add_source_in_crawl_dir_blocked_external_call(self):
        managed_crawl_path = str(self.temp_dir_path / "managed_crawl_data")
        managed_raw_path = str(RealPath(managed_crawl_path) / "raw")
        source_in_crawl_dir_str = str(RealPath(managed_raw_path) / "blocked_source.md")
        self.mock_data_manager.get_data_paths.return_value = {"crawl_data": managed_crawl_path}
        self._configure_path_mock(managed_raw_path, exists=True, is_dir=True)
        self._configure_path_mock(source_in_crawl_dir_str, exists=True, is_file=True, parents_override=[managed_raw_path, managed_crawl_path])
        count, blocked = self.mixin_instance.add_source(source_in_crawl_dir_str, internal_call=False)
        self.assertEqual(count, 0)
        self.assertTrue(blocked)
        mock_source_path_obj = self._path_constructor_side_effect(source_in_crawl_dir_str)
        self.mock_sut_logger_object.warning.assert_called_with(
            f"Cannot manually add source from managed crawl directory: {mock_source_path_obj}. Use crawl feature or move the file."
        )

    @patch.object(_SourceManagementMixin, "_is_source_unchanged", return_value=True)
    def test_add_source_unchanged_file_skipped(self, mock_is_unchanged: MagicMock):
        source_path_str = str(self.temp_dir_path / "unchanged.md")
        self._configure_path_mock(source_path_str, exists=True, is_file=True)
        count, blocked = self.mixin_instance.add_source(source_path_str)
        self.assertEqual(count, 0)
        self.assertFalse(blocked)
        mock_is_unchanged.assert_called_once_with(source_path_str, None)
        self.mock_sut_logger_object.info.assert_any_call("File 'unchanged.md' is unchanged. Skipping.")

    @patch.object(_SourceManagementMixin, "_is_source_unchanged", return_value=False)
    @patch.object(_SourceManagementMixin, "remove_source", return_value=(True, False))
    def test_add_source_file_processing_success(self, mock_remove_source: MagicMock, mock_is_unchanged: MagicMock):
        source_path_str = str(self.temp_dir_path / "test_proc.md")
        self._configure_path_mock(source_path_str, exists=True, is_file=True, read_text_content="Test content.", mtime=12345.678)
        self.mock_chunk_markdown_text.return_value = [{"chunk": "Test content.", "metadata": {"chunk_index_in_doc": 0, "length": 13, "effective_length": 13, "processing_mode": "text"}}]
        mock_embeddings_array = MagicMock(name="embeddings_array_mock")
        mock_embeddings_array.shape = (1, 10)
        mock_embeddings_array.tolist.return_value = [[0.1] * 10]
        self.mixin_instance.embedder.embed_strings.return_value = mock_embeddings_array
        self.mixin_instance.bm25.add_document.return_value = True
        count, blocked = self.mixin_instance.add_source(source_path_str)
        self.assertEqual(count, 1)
        self.assertFalse(blocked)
        mock_is_unchanged.assert_called_once_with(source_path_str, None)
        mock_remove_source.assert_called_once_with(source_path_str)
        self.mock_chunk_markdown_text.assert_called_once_with(
            markdown_text="Test content.", source=source_path_str,
            chunk_size=self.mixin_instance.max_chunk_size, chunk_overlap=self.mixin_instance.chunk_overlap,
            min_chunk_char_length=self.mixin_instance.min_chunk_size_filter,
        )
        self.mixin_instance.embedder.embed_strings.assert_called_once_with(
            ["Test content."], input_type="document", show_progress=self.mixin_instance.verbose
        )
        self.mixin_instance.chroma_collection.upsert.assert_called_once()
        chroma_kwargs = self.mixin_instance.chroma_collection.upsert.call_args.kwargs
        self.assertEqual(len(chroma_kwargs["ids"]), 1)
        self.assertTrue(chroma_kwargs["ids"][0].startswith("fakehash_0_"))
        self.assertEqual(chroma_kwargs["embeddings"], [[0.1] * 10])
        self.assertEqual(chroma_kwargs["documents"], ["Test content."])
        self.assertEqual(chroma_kwargs["metadatas"][0]["source_path"], source_path_str)
        self.mixin_instance.bm25.add_document.assert_called_once_with("Test content.", chroma_kwargs["ids"][0])
        self.mock_sut_logger_object.info.assert_any_call("Successfully processed 1 chunks from test_proc.md.")

    @patch.object(_SourceManagementMixin, "_is_source_unchanged")
    @patch.object(_SourceManagementMixin, "remove_source")
    def test_add_source_directory_processing(self, mock_remove_source, mock_is_unchanged):
        dir_path_str = str(self.temp_dir_path / "my_docs_dir")
        file1_path_str = str(RealPath(dir_path_str) / "doc1.md")
        file2_path_str = str(RealPath(dir_path_str) / "notes.txt")
        hidden_file_path_str = str(RealPath(dir_path_str) / ".config")

        mock_dir_obj = self._configure_path_mock(dir_path_str, exists=True, is_dir=True)
        mock_file1 = self._configure_path_mock(file1_path_str, exists=True, is_file=True, read_text_content="File 1 content", parents_override=[dir_path_str])
        mock_file2 = self._configure_path_mock(file2_path_str, exists=True, is_file=True, read_text_content="File 2 content", parents_override=[dir_path_str])
        mock_hidden_file = self._configure_path_mock(hidden_file_path_str, exists=True, is_file=True, read_text_content="Hidden", parents_override=[dir_path_str])
        mock_dir_obj.rglob.return_value = [mock_file1, mock_file2, mock_hidden_file]

        mock_is_unchanged.return_value = False
        mock_remove_source.return_value = (True, False)

        def chunk_side_effect(markdown_text, **kwargs):
            if markdown_text == "File 1 content":
                return [{"chunk": "chunk_f1", "metadata": {}}]
            if markdown_text == "File 2 content": # Ruff E701 fix
                return [{"chunk": "chunk_f2a", "metadata": {}}, {"chunk": "chunk_f2b", "metadata": {}}]
            return []
        self.mock_chunk_markdown_text.side_effect = chunk_side_effect
        
        mock_embeddings_array_f1 = MagicMock()
        mock_embeddings_array_f1.shape = (1,10) 
        mock_embeddings_array_f1.tolist.return_value = [[0.1]*10]
        mock_embeddings_array_f2 = MagicMock()
        mock_embeddings_array_f2.shape = (2,10) 
        mock_embeddings_array_f2.tolist.return_value = [[0.2]*10, [0.3]*10]

        def embed_side_effect(texts, **kwargs):
            if texts == ["chunk_f1"]:
                return mock_embeddings_array_f1
            if texts == ["chunk_f2a", "chunk_f2b"]: # Ruff E701 fix
                return mock_embeddings_array_f2
            return MagicMock(shape=(0,10)) 
        self.mixin_instance.embedder.embed_strings.side_effect = embed_side_effect
        self.mixin_instance.bm25.add_document.return_value = True

        count, blocked = self.mixin_instance.add_source(dir_path_str)
        
        self.assertEqual(count, 3) 
        self.assertFalse(blocked)
        mock_dir_obj.rglob.assert_called_once_with("*")
        
        mock_is_unchanged.assert_any_call(file1_path_str, None)
        mock_is_unchanged.assert_any_call(file2_path_str, None)
        mock_remove_source.assert_any_call(file1_path_str)
        mock_remove_source.assert_any_call(file2_path_str)
        
        self.mock_chunk_markdown_text.assert_any_call(markdown_text="File 1 content", source=file1_path_str, chunk_size=DEFAULT_MAX_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP, min_chunk_char_length=DEFAULT_MIN_CHUNK_SIZE_FILTER)
        self.mock_chunk_markdown_text.assert_any_call(markdown_text="File 2 content", source=file2_path_str, chunk_size=DEFAULT_MAX_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP, min_chunk_char_length=DEFAULT_MIN_CHUNK_SIZE_FILTER)
        self.mixin_instance.embedder.embed_strings.assert_any_call(["chunk_f1"], input_type="document", show_progress=False)
        self.mixin_instance.embedder.embed_strings.assert_any_call(["chunk_f2a", "chunk_f2b"], input_type="document", show_progress=False)

        self.assertEqual(self.mixin_instance.chroma_collection.upsert.call_count, 2)
        self.assertEqual(self.mixin_instance.bm25.add_document.call_count, 3)

        self.mock_sut_logger_object.info.assert_any_call("Dir scan 'my_docs_dir': Added 3 chunks from 2 files. Skipped 1. Failed 0.")

    def test_is_source_unchanged_true(self):
        source_path_str = str(self.temp_dir_path / "unchanged_check.md")
        self._configure_path_mock(source_path_str, exists=True, is_file=True, mtime=100.0)
        # Corrected GetResult: metadatas is Optional[List[Optional[Metadata]]]
        mock_metadata_list: List[Optional[Metadata]] = [{"mtime": 100.0, "source_path": source_path_str}]
        self.mixin_instance.chroma_collection.get.return_value = ChromaGetResultType(
            ids=["id1"], metadatas=mock_metadata_list,
            embeddings=None, documents=None, uris=None, data=None
        ) # type: ignore[assignment]
        self.assertTrue(self.mixin_instance._is_source_unchanged(source_path_str))
        self.mock_sut_logger_object.debug.assert_any_call(f"Source '{source_path_str}' is unchanged (mtime).")

    def test_is_source_unchanged_mtime_differs(self):
        source_path_str = str(self.temp_dir_path / "changed_check.md")
        self._configure_path_mock(source_path_str, exists=True, is_file=True, mtime=200.0)
        mock_metadata_list: List[Optional[Metadata]] = [{"mtime": 100.0, "source_path": source_path_str}]
        self.mixin_instance.chroma_collection.get.return_value = ChromaGetResultType(
            ids=["id1"], metadatas=mock_metadata_list,
            embeddings=None, documents=None, uris=None, data=None
        ) # type: ignore[assignment]
        self.assertFalse(self.mixin_instance._is_source_unchanged(source_path_str))
        self.mock_sut_logger_object.debug.assert_any_call(f"Source '{source_path_str}' mtime differs.")

    def test_get_indexed_sources_empty(self):
        self.mixin_instance.chroma_collection.count.return_value = 0
        self.mixin_instance.chroma_collection.get.return_value = ChromaGetResultType(
            ids=[], metadatas=[], embeddings=None, documents=None, uris=None, data=None
        ) # type: ignore[assignment]
        self.assertEqual(self.mixin_instance.get_indexed_sources(), [])
        self.mock_sut_logger_object.debug.assert_any_call("Fetching metadata for 0 total chunks...")

    def test_get_indexed_sources_aggregates_correctly(self):
        crawl_dir_str = str(self.temp_dir_path / "test_crawl_data_get")
        raw_dir = RealPath(crawl_dir_str) / "raw"
        self.mock_data_manager.get_data_paths.return_value = {"crawl_data": crawl_dir_str}
        self._configure_path_mock(str(raw_dir), exists=True, is_dir=True)
        local_file_path_str = str(self.temp_dir_path / "local_get" / "doc1.md")
        self._configure_path_mock(local_file_path_str, is_file=True)
        crawled_file_hash = "0123456789abcdef"
        crawled_file_name = f"{crawled_file_hash}.md"
        crawled_file_path_str_in_raw = str(raw_dir / crawled_file_name)
        self.mixin_instance._reverse_lookup = {crawled_file_hash: "http://example.com/crawled_page_get"}
        self._configure_path_mock(crawled_file_path_str_in_raw, is_file=True, parents_override=[str(raw_dir), crawl_dir_str])
        self.mixin_instance.chroma_collection.count.return_value = 3
        
        metadata_list_for_get: List[Optional[Metadata]] = [
            {"source_path": local_file_path_str, "filename": "doc1.md", "mtime": 100.0},
            {"source_path": crawled_file_path_str_in_raw, "filename": crawled_file_name, "mtime": 200.0},
            {"source_path": crawled_file_path_str_in_raw, "filename": crawled_file_name, "mtime": 200.0},
        ]
        # For get_indexed_sources, chroma_collection.get is called iteratively.
        # We mock the `get` method to return a GetResult for each batch.
        # Since batch_size is 5000 and total_docs is 3, it will be called once.
        self.mixin_instance.chroma_collection.get.return_value = ChromaGetResultType(
            ids=["l1", "c1", "c2"], metadatas=metadata_list_for_get, 
            embeddings=None, documents=None, uris=None, data=None
        ) # type: ignore[assignment]
        sources = self.mixin_instance.get_indexed_sources()
        self.assertEqual(len(sources), 2)
        local_source_found = any(s["identifier"] == local_file_path_str for s in sources)
        crawled_source_found = any(s["identifier"] == "http://example.com/crawled_page_get" for s in sources)
        self.assertTrue(local_source_found, "Local source not aggregated")
        self.assertTrue(crawled_source_found, "Crawled source not aggregated")
        for s_info in sources:
            if s_info["identifier"] == local_file_path_str:
                self.assertFalse(s_info["is_url_source"])
                self.assertEqual(s_info["chunk_count"], 1)
            elif s_info["identifier"] == "http://example.com/crawled_page_get":
                self.assertTrue(s_info["is_url_source"])
                self.assertEqual(s_info["chunk_count"], 2)

    def test_remove_source_no_chunks_found(self):
        source_id_to_remove = str(self.temp_dir_path / "file_to_remove_notfound.md")
        self._configure_path_mock(source_id_to_remove)
        self.mixin_instance.chroma_collection.get.return_value = ChromaGetResultType(
            ids=[], metadatas=[], embeddings=None, documents=None, uris=None, data=None
        ) # type: ignore[assignment]
        removed, blocked = self.mixin_instance.remove_source(source_id_to_remove)
        self.assertFalse(removed)
        self.assertFalse(blocked)
        self.mock_sut_logger_object.info.assert_any_call(f"No chunks found for identifier '{source_id_to_remove}'.")
        
if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)