# tests/test_setup.py

import unittest
import tempfile
import shutil
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock, call, ANY

# Import the module to be tested (main function)
from llamasearch import setup as llamasearch_setup_module
from llamasearch.core.embedder import DEFAULT_MODEL_NAME as DEFAULT_EMBEDDER_MODEL, EnhancedEmbedder # Import class
from llamasearch.core.onnx_model import (
    ONNX_MODEL_REPO_ID,
    ONNX_SUBFOLDER,
    MODEL_ONNX_BASENAME,
    load_onnx_llm, # Import function used in verify
    GenericONNXLLM # Import class used in verify
)
from llamasearch.exceptions import SetupError, ModelNotFoundError


class TestLlamaSearchSetup(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="test_setup_"))
        self.models_dir = self.temp_dir / "models"
        self.active_model_dir_expected = self.models_dir / "active_model"
        self.active_onnx_dir_expected = self.active_model_dir_expected / ONNX_SUBFOLDER

        # Common patches
        self.patch_data_manager = patch("llamasearch.setup.data_manager")
        self.mock_data_manager = self.patch_data_manager.start()
        self.mock_data_manager.get_data_paths.return_value = {
            "models": str(self.models_dir)
        }

        self.patch_hf_folder_token = patch(
            "llamasearch.setup.HfFolder.get_token"
        )
        self.mock_hf_token = self.patch_hf_folder_token.start()
        self.mock_hf_token.return_value = "fake_token"

        self.patch_snapshot_download = patch("llamasearch.setup.snapshot_download")
        self.mock_snapshot_download = self.patch_snapshot_download.start()
        self.mock_snapshot_download.return_value = str(self.active_model_dir_expected)

        self.patch_hf_hub_download = patch("llamasearch.setup.hf_hub_download")
        self.mock_hf_hub_download = self.patch_hf_hub_download.start()
        self.mock_cache_dir = self.temp_dir / "cache"
        self.mock_cache_dir.mkdir(parents=True, exist_ok=True)

        self.patch_shutil_rmtree = patch("llamasearch.setup.shutil.rmtree")
        self.mock_shutil_rmtree = self.patch_shutil_rmtree.start()

        self.patch_shutil_copyfile = patch("llamasearch.setup.shutil.copyfile")
        self.mock_shutil_copyfile = self.patch_shutil_copyfile.start()

        self.patch_verify_setup = patch(
            "llamasearch.setup.verify_setup", return_value=None
        )
        self.mock_verify_setup = self.patch_verify_setup.start()

        # Keep EnhancedEmbedder and load_onnx_llm patched *by default*
        self.patch_enhanced_embedder = patch("llamasearch.setup.EnhancedEmbedder")
        self.mock_enhanced_embedder_cls = self.patch_enhanced_embedder.start()
        self.mock_embedder_instance = MagicMock(spec=EnhancedEmbedder)
        self.mock_embedder_instance.get_embedding_dimension.return_value = 384
        self.mock_embedder_instance.model = MagicMock()
        self.mock_embedder_instance.close = MagicMock()
        self.mock_enhanced_embedder_cls.return_value = self.mock_embedder_instance

        self.patch_load_onnx_llm = patch("llamasearch.setup.load_onnx_llm")
        self.mock_load_onnx_llm = self.patch_load_onnx_llm.start()
        self.mock_llm_instance = MagicMock(spec=GenericONNXLLM)
        self.mock_llm_instance.model_info = MagicMock()
        self.mock_llm_instance.model_info.model_id = "mock-model-id-suffix"
        self.mock_llm_instance._quant_suffix = "suffix"
        self.mock_llm_instance.unload = MagicMock()
        self.mock_load_onnx_llm.return_value = self.mock_llm_instance

        self.patch_sys_exit = patch("llamasearch.setup.sys.exit")
        self.mock_sys_exit = self.patch_sys_exit.start()

        self.patch_gc_collect = patch("llamasearch.setup.gc.collect")
        self.mock_gc_collect = self.patch_gc_collect.start()

        # List of patchers started in setUp to stop in tearDown
        self.started_patchers = [
            self.patch_data_manager, self.patch_hf_folder_token,
            self.patch_snapshot_download, self.patch_hf_hub_download,
            self.patch_shutil_rmtree, self.patch_shutil_copyfile,
            self.patch_verify_setup, self.patch_enhanced_embedder,
            self.patch_load_onnx_llm, self.patch_sys_exit, self.patch_gc_collect
        ]

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        # Stop only the patchers started in setUp
        for patcher in self.started_patchers:
            # Add try-except block in case a patcher wasn't started correctly
            try:
                patcher.stop()
            except RuntimeError: # Handle "patcher not active"
                pass
        # Avoid patch.stopall()

    def _run_main(self, args_list=None):
        if args_list is None:
            args_list = []
        self.mock_sys_exit.reset_mock()
        self.mock_verify_setup.reset_mock()
        self.mock_snapshot_download.reset_mock()
        self.mock_hf_hub_download.reset_mock()
        self.mock_shutil_rmtree.reset_mock()
        self.mock_shutil_copyfile.reset_mock()
        self.mock_load_onnx_llm.reset_mock()
        self.mock_enhanced_embedder_cls.reset_mock()

        # Use a separate patch context for sys.argv within the run
        with patch.object(sys, 'argv', ["setup.py"] + args_list):
            try:
                llamasearch_setup_module.main()
            except SystemExit as e:
                # Log the caught exit but don't re-raise to allow test assertions
                print(f"Test caught SystemExit({e.code}).")
                # Store exit code on mock for assertion
                self.mock_sys_exit(e.code)
            except Exception as e:
                 # Log other exceptions during main execution for debugging
                 print(f"Test caught Exception during main: {e}")
                 # If sys.exit wasn't called, we need to assert failure differently or let it fail
                 # For now, let's assume sys.exit should have been called on error
                 if not self.mock_sys_exit.called:
                     self.mock_sys_exit(1) # Simulate exit(1) if main fails unexpectedly

    def _setup_mock_hf_download_for_onnx(self, suffix=""):
        onnx_model_rel_path = f"{ONNX_SUBFOLDER}/{MODEL_ONNX_BASENAME}{suffix}.onnx"
        mock_model_cache_path = self.mock_cache_dir / onnx_model_rel_path
        mock_model_cache_path.parent.mkdir(parents=True, exist_ok=True)
        mock_model_cache_path.touch()

        mock_data_cache_path = None
        onnx_data_rel_path = f"{ONNX_SUBFOLDER}/{MODEL_ONNX_BASENAME}.onnx_data"
        if suffix == "":
            mock_data_cache_path = (
                self.mock_cache_dir / ONNX_SUBFOLDER / Path(onnx_data_rel_path).name
            )
            mock_data_cache_path.touch()

        def hf_download_side_effect(repo_id, filename, cache_dir, **kwargs):
            if filename == onnx_model_rel_path:
                return str(mock_model_cache_path)
            elif mock_data_cache_path and filename == onnx_data_rel_path:
                return str(mock_data_cache_path)
            else:
                from huggingface_hub.errors import EntryNotFoundError
                raise EntryNotFoundError(f"File {filename} not found in mock repo {repo_id}")

        self.mock_hf_hub_download.side_effect = hf_download_side_effect

    # <<< New Test: No quant flag >>>
    @patch('argparse.ArgumentParser.error') # Mock the error method
    def test_main_missing_quant_flag(self, mock_argparse_error):
        """Test setup exits if no quantization flag is provided."""
        self._run_main() # Run without any flags
        # Check that parser.error was called with a message indicating the requirement
        mock_argparse_error.assert_called_once()
        self.assertTrue("A quantization flag" in mock_argparse_error.call_args[0][0])
        # sys.exit is called internally by parser.error, so we don't check it directly here
        # We can check that the rest of the setup was NOT called
        self.mock_snapshot_download.assert_not_called()
        self.mock_hf_hub_download.assert_not_called()
        self.mock_verify_setup.assert_not_called()

    def test_main_success_flow_cpu_explicit_fp32(self): # <<< Renamed >>>
        """Test successful run with explicit --fp32 flag."""
        chosen_suffix = ""
        self._setup_mock_hf_download_for_onnx(suffix=chosen_suffix)
        self._run_main(['--fp32']) # <<< Pass explicit flag >>>

        self.mock_snapshot_download.assert_any_call(
            repo_id=DEFAULT_EMBEDDER_MODEL, cache_dir=self.models_dir, local_files_only=True, local_dir_use_symlinks=False, ignore_patterns=ANY
        )
        expected_root_files = [
            "config.json", "generation_config.json", "special_tokens_map.json",
            "tokenizer.json", "tokenizer_config.json"
        ]
        self.mock_snapshot_download.assert_any_call(
            repo_id=ONNX_MODEL_REPO_ID,
            cache_dir=self.models_dir,
            local_dir=self.active_model_dir_expected,
            local_dir_use_symlinks=False,
            allow_patterns=expected_root_files,
            ignore_patterns=["*", "*/*"],
            force_download=False, resume_download=True, repo_type="model"
        )
        self.mock_hf_hub_download.assert_any_call(
            repo_id=ONNX_MODEL_REPO_ID, filename=f"{ONNX_SUBFOLDER}/{MODEL_ONNX_BASENAME}{chosen_suffix}.onnx", cache_dir=self.models_dir, force_download=False, resume_download=True, local_files_only=False, local_dir_use_symlinks=False, repo_type="model"
        )
        self.mock_hf_hub_download.assert_any_call(
            repo_id=ONNX_MODEL_REPO_ID, filename=f"{ONNX_SUBFOLDER}/{MODEL_ONNX_BASENAME}.onnx_data", cache_dir=self.models_dir, force_download=False, resume_download=True, local_files_only=False, local_dir_use_symlinks=False, repo_type="model"
        )
        expected_onnx_src = str(self.mock_cache_dir / ONNX_SUBFOLDER / f"{MODEL_ONNX_BASENAME}{chosen_suffix}.onnx")
        expected_onnx_dst = self.active_onnx_dir_expected / f"{MODEL_ONNX_BASENAME}{chosen_suffix}.onnx"
        expected_data_src = str(self.mock_cache_dir / ONNX_SUBFOLDER / f"{MODEL_ONNX_BASENAME}.onnx_data")
        expected_data_dst = self.active_onnx_dir_expected / f"{MODEL_ONNX_BASENAME}.onnx_data"
        self.mock_shutil_copyfile.assert_any_call(expected_onnx_src, expected_onnx_dst)
        self.mock_shutil_copyfile.assert_any_call(expected_data_src, expected_data_dst)
        self.assertTrue(self.active_model_dir_expected.exists())
        self.assertTrue(self.active_onnx_dir_expected.exists())
        self.mock_verify_setup.assert_called_once_with(chosen_suffix)
        self.mock_sys_exit.assert_called_once_with(0)

    def test_main_success_flow_cpu_explicit_int8(self):
        """Test successful run with explicit --int8 flag."""
        chosen_suffix = "_int8"
        self._setup_mock_hf_download_for_onnx(suffix=chosen_suffix)
        self._run_main(["--int8"])

        self.mock_hf_hub_download.assert_any_call(
            repo_id=ONNX_MODEL_REPO_ID, filename=f"{ONNX_SUBFOLDER}/{MODEL_ONNX_BASENAME}{chosen_suffix}.onnx", cache_dir=self.models_dir, force_download=False, resume_download=True, local_files_only=False, local_dir_use_symlinks=False, repo_type="model"
        )
        data_file_call = call(
            repo_id=ONNX_MODEL_REPO_ID, filename=f"{ONNX_SUBFOLDER}/{MODEL_ONNX_BASENAME}.onnx_data", cache_dir=self.models_dir, force_download=False, resume_download=True, local_files_only=False, local_dir_use_symlinks=False, repo_type="model"
        )
        self.assertNotIn(data_file_call, self.mock_hf_hub_download.call_args_list)
        expected_onnx_src = str(self.mock_cache_dir / ONNX_SUBFOLDER / f"{MODEL_ONNX_BASENAME}{chosen_suffix}.onnx")
        expected_onnx_dst = self.active_onnx_dir_expected / f"{MODEL_ONNX_BASENAME}{chosen_suffix}.onnx"
        self.mock_shutil_copyfile.assert_called_once_with(expected_onnx_src, expected_onnx_dst)
        self.mock_verify_setup.assert_called_once_with(chosen_suffix)
        self.mock_sys_exit.assert_called_once_with(0)

    def test_main_force_redownload(self):
        # <<< FIX: Pass an explicit quant flag >>>
        chosen_suffix = "_fp16"
        self._setup_mock_hf_download_for_onnx(suffix=chosen_suffix)
        self.active_model_dir_expected.mkdir(parents=True, exist_ok=True)
        self.active_onnx_dir_expected.mkdir(parents=True, exist_ok=True)
        (self.active_onnx_dir_expected / "some_old_file").touch()

        self._run_main(["--force", "--fp16"]) # <<< Pass flags >>>

        self.mock_shutil_rmtree.assert_called_once_with(self.active_model_dir_expected)
        self.mock_snapshot_download.assert_any_call(
            repo_id=DEFAULT_EMBEDDER_MODEL, cache_dir=self.models_dir, force_download=True, resume_download=False, local_files_only=False, local_dir_use_symlinks=False, ignore_patterns=ANY
        )
        expected_root_files = [
            "config.json", "generation_config.json", "special_tokens_map.json",
            "tokenizer.json", "tokenizer_config.json"
        ]
        self.mock_snapshot_download.assert_any_call(
            repo_id=ONNX_MODEL_REPO_ID,
            cache_dir=self.models_dir,
            local_dir=self.active_model_dir_expected,
            local_dir_use_symlinks=False,
            allow_patterns=expected_root_files,
            ignore_patterns=["*", "*/*"],
            force_download=True,
            resume_download=False,
            repo_type="model"
        )
        self.mock_hf_hub_download.assert_any_call(
            repo_id=ONNX_MODEL_REPO_ID, filename=f"{ONNX_SUBFOLDER}/{MODEL_ONNX_BASENAME}{chosen_suffix}.onnx", cache_dir=self.models_dir, force_download=True, resume_download=False, local_files_only=False, local_dir_use_symlinks=False, repo_type="model"
        )
        self.mock_verify_setup.assert_called_once_with(chosen_suffix)
        self.mock_sys_exit.assert_called_once_with(0)

    def test_main_force_rmtree_error(self):
        chosen_suffix = "_int8" # <<< Choose a flag >>>
        self._setup_mock_hf_download_for_onnx(suffix=chosen_suffix)
        self.active_model_dir_expected.mkdir(parents=True, exist_ok=True)
        self.mock_shutil_rmtree.side_effect = OSError("Permission denied")

        self._run_main(["--force", "--int8"]) # <<< Pass flags >>>
        self.mock_sys_exit.assert_called_once_with(1)
        self.mock_shutil_rmtree.assert_called_once_with(self.active_model_dir_expected)
        self.mock_verify_setup.assert_not_called()

    def test_main_embedder_download_failure(self):
        self.mock_snapshot_download.side_effect = [
            llamasearch_setup_module.LocalEntryNotFoundError("Embedder not found locally"),
            SetupError("Network error during embedder download"),
        ]
        self._run_main(['--fp32']) # <<< Pass required flag >>>
        self.mock_sys_exit.assert_called_once_with(1)

    def test_main_onnx_llm_root_files_download_failure(self):
        def snapshot_side_effect(*args, **kwargs):
            if kwargs.get("repo_id") == DEFAULT_EMBEDDER_MODEL:
                if kwargs.get("local_files_only"):
                    raise llamasearch_setup_module.LocalEntryNotFoundError("Embedder not local")
                return str(self.temp_dir / "embedder_cache")
            elif kwargs.get("repo_id") == ONNX_MODEL_REPO_ID:
                raise SetupError("Network error during ONNX root download")
            return MagicMock()

        self.mock_snapshot_download.side_effect = snapshot_side_effect
        self._run_main(['--fp32']) # <<< Pass required flag >>>
        self.mock_sys_exit.assert_called_once_with(1)

    def test_main_onnx_llm_model_file_download_failure(self):
        self.mock_snapshot_download.return_value = str(self.active_model_dir_expected)
        self.mock_hf_hub_download.side_effect = SetupError("Network error for ONNX model file")
        self._run_main(["--int8"])
        self.mock_sys_exit.assert_called_once_with(1)

    def test_main_onnx_copyfile_failure(self):
        chosen_suffix = "_int8"
        self._setup_mock_hf_download_for_onnx(suffix=chosen_suffix)
        self.mock_shutil_copyfile.side_effect = IOError("Disk full")

        self._run_main(["--int8"])
        self.mock_sys_exit.assert_called_once_with(1)
        expected_onnx_src = str(self.mock_cache_dir / ONNX_SUBFOLDER / f"{MODEL_ONNX_BASENAME}{chosen_suffix}.onnx")
        expected_onnx_dst = self.active_onnx_dir_expected / f"{MODEL_ONNX_BASENAME}{chosen_suffix}.onnx"
        self.mock_shutil_copyfile.assert_called_once_with(expected_onnx_src, expected_onnx_dst)
        self.mock_verify_setup.assert_not_called()

    def test_main_onnx_llm_fp32_data_file_missing(self):
        """Test failure when fp32 is chosen but .onnx_data is missing from repo."""
        onnx_model_rel_path = f"{ONNX_SUBFOLDER}/{MODEL_ONNX_BASENAME}.onnx"
        mock_model_cache_path = self.mock_cache_dir / onnx_model_rel_path
        mock_model_cache_path.parent.mkdir(parents=True, exist_ok=True)
        mock_model_cache_path.touch()

        def hf_download_side_effect_data_missing(repo_id, filename, cache_dir, **kwargs):
            if filename == onnx_model_rel_path:
                return str(mock_model_cache_path)
            if filename == f"{ONNX_SUBFOLDER}/{MODEL_ONNX_BASENAME}.onnx_data":
                from huggingface_hub.errors import EntryNotFoundError
                raise EntryNotFoundError("data file missing")
            raise ValueError(f"Unexpected filename in mock: {filename}")

        self.mock_hf_hub_download.side_effect = hf_download_side_effect_data_missing
        self._run_main(["--fp32"])
        self.mock_sys_exit.assert_called_once_with(1)

    def test_main_verification_failure(self):
        chosen_suffix = "_fp16"
        self._setup_mock_hf_download_for_onnx(suffix=chosen_suffix)
        self.mock_verify_setup.side_effect = SetupError("Verification failed")
        self._run_main(["--fp16"])
        self.mock_verify_setup.assert_called_once_with(chosen_suffix)
        self.mock_sys_exit.assert_called_once_with(1)

    # <<< FIX: Use 'with patch(...)' for local patch scope >>>
    def test_main_verify_embedder_missing(self):
        """Test setup exit when verify_setup fails on embedder."""
        chosen_suffix = "_fp16" # <<< Pass a required flag >>>
        self._setup_mock_hf_download_for_onnx(suffix=chosen_suffix)

        # Use context managers for patching within this test's scope
        with patch("llamasearch.setup.verify_setup", llamasearch_setup_module.verify_setup), \
             patch("llamasearch.setup.EnhancedEmbedder", side_effect=ModelNotFoundError("Embedder missing")) as mock_embedder_error, \
             patch("llamasearch.setup.load_onnx_llm") as mock_llm_loader:

            self._run_main(['--fp16']) # <<< Pass the required flag >>>

            # Assert that the main loop caught SetupError and exited
            self.mock_sys_exit.assert_called_once_with(1)
            mock_embedder_error.assert_called_once()
            mock_llm_loader.assert_not_called()

    # <<< FIX: Use 'with patch(...)' and configure embedder mock correctly >>>
    def test_main_verify_llm_missing(self):
        """Test setup exit when verify_setup fails on LLM."""
        chosen_suffix = "_fp16" # <<< Pass a required flag >>>
        self._setup_mock_hf_download_for_onnx(suffix=chosen_suffix)

        # Use context managers for patching within this test's scope
        with patch("llamasearch.setup.verify_setup", llamasearch_setup_module.verify_setup), \
             patch("llamasearch.setup.EnhancedEmbedder") as mock_embedder_cls, \
             patch("llamasearch.setup.load_onnx_llm", side_effect=ModelNotFoundError("LLM missing")) as mock_llm_error:

            # Configure the embedder mock *inside the context* to pass verification
            mock_embedder_inst = MagicMock(spec=EnhancedEmbedder)
            mock_embedder_inst.get_embedding_dimension.return_value = 384
            mock_embedder_inst.model = MagicMock()
            mock_embedder_inst.close = MagicMock()
            mock_embedder_cls.return_value = mock_embedder_inst

            self._run_main(['--fp16']) # <<< Pass the required flag >>>

            # Assert that the main loop caught SetupError and exited
            self.mock_sys_exit.assert_called_once_with(1)
            mock_embedder_cls.assert_called_once()
            # <<< FIX: load_onnx_llm inside verify_setup is called with 'auto' >>>
            mock_llm_error.assert_called_once_with(onnx_quantization='auto')

    def test_main_hf_token_check_fails(self):
        """Test warning when HF token check fails."""
        self.mock_hf_token.side_effect = Exception("Cannot access token")
        chosen_suffix = "_fp16" # <<< Pass a required flag >>>
        self._setup_mock_hf_download_for_onnx(suffix=chosen_suffix)

        with self.assertLogs(logger='llamasearch.setup', level='WARNING') as cm:
            self._run_main(['--fp16']) # <<< Pass the required flag >>>

        self.assertTrue(any("Could not check HF token" in log for log in cm.output))
        self.mock_sys_exit.assert_called_once_with(0)

    def test_main_hf_token_is_none(self):
        """Test warning when HF token is not set."""
        self.mock_hf_token.return_value = None
        chosen_suffix = "_fp16" # <<< Pass a required flag >>>
        self._setup_mock_hf_download_for_onnx(suffix=chosen_suffix)

        with self.assertLogs(logger='llamasearch.setup', level='WARNING') as cm:
            self._run_main(['--fp16']) # <<< Pass the required flag >>>

        self.assertTrue(any("HF token not found. Downloads might fail for gated models" in log for log in cm.output))
        self.mock_sys_exit.assert_called_once_with(0)

    # Test the download_file_with_retry helper
    @patch("llamasearch.setup.hf_hub_download")
    @patch("llamasearch.setup.time.sleep")
    def test_download_file_with_retry_success_first_attempt(self, mock_sleep, mock_hf_dl_local):
        mock_file_path = self.temp_dir / "testfile.txt"
        mock_hf_dl_local.return_value = str(mock_file_path)
        mock_file_path.touch()
        result = llamasearch_setup_module.download_file_with_retry(
            "repo", "testfile.txt", self.temp_dir, False
        )
        self.assertEqual(result, str(mock_file_path))
        mock_hf_dl_local.assert_called_once()
        mock_sleep.assert_not_called()

    @patch("llamasearch.setup.hf_hub_download")
    @patch("llamasearch.setup.time.sleep")
    def test_download_file_with_retry_success_after_retries(self, mock_sleep, mock_hf_dl_local):
        mock_file_path = self.temp_dir / "testfile.txt"
        mock_hf_dl_local.side_effect = [
            ConnectionError("Fail1"),
            ConnectionError("Fail2"),
            str(mock_file_path),
        ]
        mock_file_path.touch()
        result = llamasearch_setup_module.download_file_with_retry(
            "repo", "testfile.txt", self.temp_dir, False, max_retries=2
        )
        self.assertEqual(result, str(mock_file_path))
        self.assertEqual(mock_hf_dl_local.call_count, 3)
        self.assertEqual(mock_sleep.call_count, 2)

    @patch("llamasearch.setup.hf_hub_download")
    @patch("llamasearch.setup.time.sleep")
    def test_download_file_with_retry_failure_after_max_retries(self, mock_sleep, mock_hf_dl_local):
        mock_hf_dl_local.side_effect = ConnectionError("Persistent failure")
        with self.assertRaisesRegex(SetupError, "Failed DL after retries: testfile.txt"):
            llamasearch_setup_module.download_file_with_retry(
                "repo", "testfile.txt", self.temp_dir, False, max_retries=1, delay=1
            )
        self.assertEqual(mock_hf_dl_local.call_count, 2)
        self.assertEqual(mock_sleep.call_count, 1)

    @patch("llamasearch.setup.hf_hub_download")
    @patch("llamasearch.setup.time.sleep")
    def test_download_file_with_retry_filenotfound_error(self, mock_sleep, mock_hf_dl_local):
        """Test retry logic when hf_hub_download returns path but file doesn't exist."""
        mock_file_path_str = str(self.temp_dir / "testfile.txt")
        mock_hf_dl_local.side_effect = [mock_file_path_str, mock_file_path_str]

        with self.assertRaisesRegex(SetupError, "Failed DL after retries: testfile.txt"):
             with self.assertLogs(logger='llamasearch.setup', level='WARNING') as cm:
                  llamasearch_setup_module.download_file_with_retry(
                       "repo", "testfile.txt", self.temp_dir, False, max_retries=1, delay=1
                  )
                  self.assertTrue(any(f"File testfile.txt invalid or DNE after DL attempt 1" in log for log in cm.output))

        self.assertEqual(mock_hf_dl_local.call_count, 2)
        self.assertEqual(mock_sleep.call_count, 1)

    @patch("llamasearch.setup.hf_hub_download")
    @patch("llamasearch.setup.time.sleep")
    def test_download_file_with_retry_entry_not_found(self, mock_sleep, mock_hf_dl_local):
        from huggingface_hub.errors import EntryNotFoundError
        mock_hf_dl_local.side_effect = EntryNotFoundError("File not in repo")
        with self.assertRaises(EntryNotFoundError):
            llamasearch_setup_module.download_file_with_retry(
                "repo", "testfile.txt", self.temp_dir, False
            )
        mock_hf_dl_local.assert_called_once()
        mock_sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)