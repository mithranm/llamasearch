# tests/test_setup.py
import unittest
import sys
import logging
from pathlib import Path
import tempfile
import shutil # For cleaning active_model_dir if necessary
from unittest.mock import patch, MagicMock, call, ANY

# Import the main function and other components from the setup script
from llamasearch.setup import (
    main as setup_main,
    check_or_download_embedder,
    check_or_download_onnx_llm,
    verify_setup,
    REQUIRED_ROOT_FILES,
    DEFAULT_EMBEDDER_MODEL,
    ONNX_MODEL_REPO_ID,
    ONNX_SUBFOLDER,
    MODEL_ONNX_BASENAME,
    download_file_with_retry, # Import helper for direct testing
)
from llamasearch.core.onnx_model import GenericONNXLLM, GenericONNXModelInfo # For spec
from llamasearch.exceptions import SetupError, ModelNotFoundError


# Mock data_manager used by setup.py
MOCK_SETUP_DATA_MANAGER_TARGET = "llamasearch.setup.data_manager"

# No global logger patcher for setup_logging function itself needed if we patch llamasearch.setup.logger

class TestSetupScript(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory for test isolation
        self.temp_dir_obj = tempfile.TemporaryDirectory(prefix="test_setup_")
        self.temp_dir = Path(self.temp_dir_obj.name)
        self.models_dir = self.temp_dir / "models" # Consistent with DataManager default structure
        self.active_model_dir = self.models_dir / "active_model"
        self.active_onnx_dir = self.active_model_dir / ONNX_SUBFOLDER

        # Patch data_manager for all tests in this class
        self.data_manager_patcher = patch(MOCK_SETUP_DATA_MANAGER_TARGET)
        self.mock_data_manager = self.data_manager_patcher.start()
        self.mock_data_manager.get_data_paths.return_value = {
            "models": str(self.models_dir)
        }

        # Patch sys.argv and sys.exit by default
        self.argv_patcher = patch.object(sys, 'argv', ['setup.py'])
        self.mock_argv = self.argv_patcher.start()
        self.exit_patcher = patch.object(sys, 'exit')
        self.mock_exit = self.exit_patcher.start()

    def tearDown(self):
        self.exit_patcher.stop()
        self.argv_patcher.stop()
        self.data_manager_patcher.stop()
        self.temp_dir_obj.cleanup()

    def _create_dummy_cached_file(self, repo_path: Path, filename: str):
        """Helper to create a dummy file simulating a cached download."""
        file_path = repo_path / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(f"content for {filename}")
        return str(file_path)

    @patch("llamasearch.setup.logger") # Patch the module-level logger in setup.py
    @patch("llamasearch.setup.check_or_download_embedder")
    @patch("llamasearch.setup.check_or_download_onnx_llm")
    @patch("llamasearch.setup.verify_setup")
    @patch("llamasearch.setup.HfFolder.get_token", return_value="fake-token")
    def test_main_success_no_force(
        self, mock_get_token, mock_verify, mock_download_llm, mock_download_embedder, mock_sut_logger
    ):
        """Test successful main execution without --force."""
        setup_main()

        self.mock_data_manager.get_data_paths.assert_called_once()
        mock_get_token.assert_called_once()
        mock_download_embedder.assert_called_once_with(self.models_dir, False)
        mock_download_llm.assert_called_once_with(self.models_dir, False)
        mock_verify.assert_called_once_with()
        self.mock_exit.assert_called_once_with(0)
        mock_sut_logger.info.assert_any_call(
            f"Using models directory: {self.models_dir}"
        )
        mock_sut_logger.info.assert_any_call(
            "--- LlamaSearch Model Setup Completed Successfully (FP32) ---"
        )

    @patch("llamasearch.setup.logger")
    @patch("llamasearch.setup.check_or_download_embedder")
    @patch("llamasearch.setup.check_or_download_onnx_llm")
    @patch("llamasearch.setup.verify_setup")
    @patch("llamasearch.setup.HfFolder.get_token", return_value="fake-token")
    def test_main_success_force(
        self, mock_get_token, mock_verify, mock_download_llm, mock_download_embedder, mock_sut_logger
    ):
        """Test successful main execution with --force."""
        self.mock_argv[:] = ['setup.py', '--force'] # Simulate --force flag
        setup_main()

        self.mock_data_manager.get_data_paths.assert_called_once()
        mock_get_token.assert_called_once()
        mock_download_embedder.assert_called_once_with(self.models_dir, True) # Force=True
        mock_download_llm.assert_called_once_with(self.models_dir, True) # Force=True
        mock_verify.assert_called_once_with()
        self.mock_exit.assert_called_once_with(0)
        mock_sut_logger.info.assert_any_call(
            "Force mode enabled: Active directory will be recreated."
        )

    @patch("llamasearch.setup.logger")
    @patch("llamasearch.setup.check_or_download_embedder", side_effect=SetupError("Embedder fail"))
    @patch("llamasearch.setup.check_or_download_onnx_llm")
    @patch("llamasearch.setup.verify_setup")
    def test_main_fail_embedder_download(
        self, mock_verify, mock_download_llm, mock_download_embedder, mock_sut_logger
    ):
        """Test main exits if embedder download fails."""
        setup_main()
        mock_download_embedder.assert_called_once_with(self.models_dir, False)
        mock_download_llm.assert_not_called() # Should fail before LLM download
        mock_verify.assert_not_called()
        mock_sut_logger.error.assert_any_call("Setup failed: Embedder fail")
        self.mock_exit.assert_called_once_with(1)

    @patch("llamasearch.setup.logger")
    @patch("llamasearch.setup.check_or_download_embedder")
    @patch("llamasearch.setup.check_or_download_onnx_llm", side_effect=SetupError("LLM fail"))
    @patch("llamasearch.setup.verify_setup")
    def test_main_fail_llm_download(
        self, mock_verify, mock_download_llm, mock_download_embedder, mock_sut_logger
    ):
        """Test main exits if LLM download fails."""
        setup_main()
        mock_download_embedder.assert_called_once_with(self.models_dir, False)
        mock_download_llm.assert_called_once_with(self.models_dir, False)
        mock_verify.assert_not_called() # Should fail before verification
        mock_sut_logger.error.assert_any_call("Setup failed: LLM fail")
        self.mock_exit.assert_called_once_with(1)

    @patch("llamasearch.setup.logger")
    @patch("llamasearch.setup.check_or_download_embedder")
    @patch("llamasearch.setup.check_or_download_onnx_llm")
    @patch("llamasearch.setup.verify_setup", side_effect=SetupError("Verify fail"))
    def test_main_fail_verification(
        self, mock_verify, mock_download_llm, mock_download_embedder, mock_sut_logger
    ):
        """Test main exits if verification fails."""
        setup_main()
        mock_download_embedder.assert_called_once_with(self.models_dir, False)
        mock_download_llm.assert_called_once_with(self.models_dir, False)
        mock_verify.assert_called_once_with()
        mock_sut_logger.error.assert_any_call("Setup failed: Verify fail")
        self.mock_exit.assert_called_once_with(1)

    @patch("llamasearch.setup.logger")
    def test_main_fail_no_models_path(self, mock_sut_logger):
        """Test main exits if models path is not configured."""
        self.mock_data_manager.get_data_paths.return_value = {} # Simulate missing path
        setup_main()
        mock_sut_logger.error.assert_any_call("Setup failed: Models directory path not configured.")
        self.mock_exit.assert_called_once_with(1)

    @patch("llamasearch.setup.logger")
    @patch("llamasearch.setup.snapshot_download")
    def test_check_or_download_embedder_exists_locally(self, mock_snapshot_dl, mock_sut_logger):
        """Test embedder download skips if found locally (no force)."""
        mock_snapshot_dl.side_effect = [None] 
        check_or_download_embedder(self.models_dir, force=False)
        mock_snapshot_dl.assert_called_once_with(
            repo_id=DEFAULT_EMBEDDER_MODEL,
            cache_dir=self.models_dir,
            local_files_only=True,
            local_dir_use_symlinks=False,
            ignore_patterns=ANY,
        )
        mock_sut_logger.info.assert_any_call(
             f"Embedder '{DEFAULT_EMBEDDER_MODEL}' (PyTorch) found locally."
        )

    @patch("llamasearch.setup.logger")
    @patch("llamasearch.setup.snapshot_download")
    def test_check_or_download_embedder_downloads_if_missing(self, mock_snapshot_dl, mock_sut_logger):
        """Test embedder download proceeds if not found locally."""
        mock_snapshot_dl.side_effect = [FileNotFoundError("Not found locally"), None]
        check_or_download_embedder(self.models_dir, force=False)
        self.assertEqual(mock_snapshot_dl.call_count, 2)
        call_args_list = mock_snapshot_dl.call_args_list
        self.assertEqual(call_args_list[1], call(
            repo_id=DEFAULT_EMBEDDER_MODEL,
            cache_dir=self.models_dir,
            force_download=False,
            resume_download=True,
            local_files_only=False,
            local_dir_use_symlinks=False,
            ignore_patterns=ANY,
        ))
        mock_sut_logger.info.assert_any_call(
            f"Embedder '{DEFAULT_EMBEDDER_MODEL}' (PyTorch) cache verified/downloaded."
        )

    @patch("llamasearch.setup.snapshot_download")
    def test_check_or_download_embedder_force_downloads(self, mock_snapshot_dl):
        """Test embedder download proceeds with force=True."""
        check_or_download_embedder(self.models_dir, force=True)
        mock_snapshot_dl.assert_called_once_with(
            repo_id=DEFAULT_EMBEDDER_MODEL,
            cache_dir=self.models_dir,
            force_download=True, 
            resume_download=False, 
            local_files_only=False,
            local_dir_use_symlinks=False,
            ignore_patterns=ANY,
        )

    @patch("llamasearch.setup.logger")
    @patch("llamasearch.setup.download_file_with_retry")
    @patch("llamasearch.setup.shutil.copyfile")
    @patch("llamasearch.setup.shutil.rmtree")
    # Removed Path.exists and Path.is_file mocks here to test real file operations
    def test_check_or_download_onnx_llm_success(
        self, mock_rmtree, mock_copy, mock_dl_helper, mock_sut_logger
    ):
        """Test successful download and assembly of ONNX files."""
        # Ensure active_model_dir and active_onnx_dir are created for the test
        # This simulates them existing from a previous (or this) run.
        self.active_onnx_dir.mkdir(parents=True, exist_ok=True)
        # Simulate the necessary ONNX files being present to avoid cleaning
        (self.active_onnx_dir / f"{MODEL_ONNX_BASENAME}.onnx").touch()
        (self.active_onnx_dir / f"{MODEL_ONNX_BASENAME}.onnx_data").touch()

        def dl_side_effect(repo_id, filename, **kwargs):
             return self._create_dummy_cached_file(self.models_dir / repo_id, filename)
        mock_dl_helper.side_effect = dl_side_effect

        check_or_download_onnx_llm(self.models_dir, force=False)

        mock_rmtree.assert_not_called() 
        self.assertTrue(self.active_model_dir.exists())
        self.assertTrue(self.active_onnx_dir.exists())

        expected_dl_calls = []
        for fname in REQUIRED_ROOT_FILES:
            expected_dl_calls.append(call(
                repo_id=ONNX_MODEL_REPO_ID, filename=fname, cache_dir=self.models_dir, force=False, repo_type='model'
            ))
        onnx_model_rel_path = f"{ONNX_SUBFOLDER}/{MODEL_ONNX_BASENAME}.onnx"
        onnx_data_rel_path = f"{ONNX_SUBFOLDER}/{MODEL_ONNX_BASENAME}.onnx_data"
        expected_dl_calls.append(call(
            repo_id=ONNX_MODEL_REPO_ID, filename=onnx_model_rel_path, cache_dir=self.models_dir, force=False, repo_type='model'
        ))
        expected_dl_calls.append(call(
            repo_id=ONNX_MODEL_REPO_ID, filename=onnx_data_rel_path, cache_dir=self.models_dir, force=False, repo_type='model'
        ))
        mock_dl_helper.assert_has_calls(expected_dl_calls, any_order=True)
        self.assertEqual(mock_dl_helper.call_count, len(REQUIRED_ROOT_FILES) + 2)

        expected_copy_calls = []
        for fname in REQUIRED_ROOT_FILES:
            src_path = str(self.models_dir / ONNX_MODEL_REPO_ID / fname)
            dest_path = self.active_model_dir / fname
            expected_copy_calls.append(call(src_path, dest_path))

        src_model_path = str(self.models_dir / ONNX_MODEL_REPO_ID / onnx_model_rel_path)
        dest_model_path = self.active_onnx_dir / Path(onnx_model_rel_path).name
        expected_copy_calls.append(call(src_model_path, dest_model_path))

        src_data_path = str(self.models_dir / ONNX_MODEL_REPO_ID / onnx_data_rel_path)
        dest_data_path = self.active_onnx_dir / Path(onnx_data_rel_path).name
        expected_copy_calls.append(call(src_data_path, dest_data_path))

        mock_copy.assert_has_calls(expected_copy_calls, any_order=True)
        self.assertEqual(mock_copy.call_count, len(REQUIRED_ROOT_FILES) + 2)

    @patch("llamasearch.setup.logger")
    @patch("llamasearch.setup.download_file_with_retry")
    @patch("llamasearch.setup.shutil.copyfile")
    @patch("llamasearch.setup.shutil.rmtree")
    # Removed Path.exists and Path.is_file mocks
    def test_check_or_download_onnx_llm_force_cleans_dir(
        self, mock_rmtree, mock_copy, mock_dl_helper, mock_sut_logger
    ):
        """Test --force correctly cleans the active_model directory."""
        mock_dl_helper.side_effect = lambda repo_id, filename, **kwargs: self._create_dummy_cached_file(self.models_dir/repo_id, filename)

        self.active_model_dir.mkdir(parents=True, exist_ok=True)
        (self.active_model_dir / "old_file.txt").touch()
        self.assertTrue((self.active_model_dir / "old_file.txt").exists())


        check_or_download_onnx_llm(self.models_dir, force=True)

        mock_rmtree.assert_called_once_with(self.active_model_dir)
        self.assertTrue(self.active_model_dir.exists()) 
        self.assertFalse((self.active_model_dir / "old_file.txt").exists()) 
        self.assertEqual(mock_copy.call_count, len(REQUIRED_ROOT_FILES) + 2)

    @patch("llamasearch.setup.logger")
    @patch("llamasearch.setup.download_file_with_retry")
    @patch("llamasearch.setup.shutil.rmtree")
    # Removed Path.exists and Path.is_file mocks
    def test_check_or_download_onnx_llm_cleans_if_file_missing(
        self, mock_rmtree, mock_dl_helper, mock_sut_logger
    ):
        """Test active_model is cleaned if a required ONNX file is missing (no force)."""
        # Setup: active_model_dir exists, but model.onnx does not.
        self.active_model_dir.mkdir(parents=True, exist_ok=True)
        self.active_onnx_dir.mkdir(parents=True, exist_ok=True) # Ensure onnx subfolder exists
        # (model.onnx is not created, so it's "missing")
        (self.active_onnx_dir / f"{MODEL_ONNX_BASENAME}.onnx_data").touch() # data file exists

        mock_dl_helper.side_effect = lambda repo_id, filename, **kwargs: self._create_dummy_cached_file(self.models_dir/repo_id, filename)

        check_or_download_onnx_llm(self.models_dir, force=False)

        mock_rmtree.assert_called_once_with(self.active_model_dir)
        mock_sut_logger.warning.assert_any_call(
            f"Target ONNX file '{MODEL_ONNX_BASENAME}.onnx' missing in {self.active_onnx_dir}. Forcing clean assembly."
        )

    @patch("llamasearch.setup.logger")
    @patch("llamasearch.setup.download_file_with_retry", side_effect=SetupError("Download failed"))
    @patch("llamasearch.setup.shutil.rmtree")
    # Removed Path.exists mock
    def test_check_or_download_onnx_llm_fail_download(
        self, mock_rmtree, mock_dl_helper, mock_sut_logger
    ):
        """Test failure during download_file_with_retry call."""
        # active_model_dir might not exist yet, so no rmtree
        if self.active_model_dir.exists():
             shutil.rmtree(self.active_model_dir)
        self.assertFalse(self.active_model_dir.exists())

        with self.assertRaisesRegex(SetupError, "Failed to process root files"):
            check_or_download_onnx_llm(self.models_dir, force=False)
        # rmtree should not be called if active_model_dir wasn't created or if it failed before cleaning
        # The logic in SUT creates active_model_dir *after* rmtree if needs_clean.
        # If needs_clean is false (initial state) and no active_model_dir exists, no rmtree.
        # If needs_clean becomes true due to missing files, then it tries rmtree.
        # Here, it fails at download_file_with_retry for root files, which is before full active_model_dir structure is potentially cleaned.
        # The check for `needs_clean` happens. If `active_model_dir` does not exist `needs_clean` might not be set to true
        # based on missing onnx files. If `force` is False, and active_model_dir doesn't exist, it's not cleaned.
        # It will proceed to download, and if that fails, it raises.
        # So, rmtree not being called is correct here if active_model_dir didn't exist.
        # If it *did* exist but files were fine, and DL failed later, then also no rmtree.
        # The most direct path to this test condition is active_model_dir DNE.
        mock_rmtree.assert_not_called()


    @patch("llamasearch.setup.logger")
    @patch("llamasearch.setup.EnhancedEmbedder")
    @patch("llamasearch.setup.load_onnx_llm")
    def test_verify_setup_success(self, mock_load_llm, mock_embedder_cls, mock_sut_logger):
        """Test successful verification."""
        mock_embedder_instance = mock_embedder_cls.return_value
        mock_embedder_instance.get_embedding_dimension.return_value = 384
        mock_embedder_instance.model = MagicMock() 

        mock_llm_instance = MagicMock(spec=GenericONNXLLM)
        mock_llm_instance.model_info = MagicMock(spec=GenericONNXModelInfo)
        mock_llm_instance.model_info.model_id = "test-onnx-model"
        mock_load_llm.return_value = mock_llm_instance

        verify_setup()

        mock_embedder_cls.assert_called_once_with(batch_size=0)
        mock_embedder_instance.get_embedding_dimension.assert_called_once()
        mock_embedder_instance.close.assert_called_once()

        mock_load_llm.assert_called_once_with()
        mock_llm_instance.unload.assert_called_once()
        mock_sut_logger.info.assert_any_call("--- Model Verification Successful (CPU-Only, FP32) ---")

    @patch("llamasearch.setup.logger")
    @patch("llamasearch.setup.EnhancedEmbedder", side_effect=ModelNotFoundError("Embedder model missing"))
    @patch("llamasearch.setup.load_onnx_llm")
    def test_verify_setup_fail_embedder_load(self, mock_load_llm, mock_embedder_cls, mock_sut_logger):
        """Test verification fails if embedder loading raises ModelNotFoundError."""
        with self.assertRaisesRegex(SetupError, "Embedder model files not found"):
            verify_setup()
        mock_load_llm.assert_not_called() 
        mock_sut_logger.error.assert_any_call(
            "FAIL: Embedder model not found. Embedder model missing"
        )

    @patch("llamasearch.setup.logger")
    @patch("llamasearch.setup.EnhancedEmbedder")
    @patch("llamasearch.setup.load_onnx_llm", side_effect=ModelNotFoundError("LLM files missing"))
    def test_verify_setup_fail_llm_load(self, mock_load_llm, mock_embedder_cls, mock_sut_logger):
        """Test verification fails if LLM loading raises ModelNotFoundError."""
        mock_embedder_instance = mock_embedder_cls.return_value
        mock_embedder_instance.get_embedding_dimension.return_value = 384
        mock_embedder_instance.model = MagicMock()

        with self.assertRaisesRegex(SetupError, "ONNX LLM model files not found"):
            verify_setup()

        mock_embedder_instance.close.assert_called_once() 
        mock_load_llm.assert_called_once()
        mock_sut_logger.error.assert_any_call(
            "FAIL: ONNX LLM files missing or incomplete in active_model. LLM files missing"
        )

    @patch("llamasearch.setup.logger")
    @patch("llamasearch.setup.hf_hub_download")
    def test_download_helper_success(self, mock_hf_dl, mock_sut_logger): # Added mock_sut_logger
        """Test download_file_with_retry success on first attempt."""
        expected_path = "/fake/path/model.safetensors"
        mock_hf_dl.return_value = expected_path
        with patch("pathlib.Path.is_file", return_value=True):
            result = download_file_with_retry("repo", "model.safetensors", self.models_dir, False)
        self.assertEqual(result, expected_path)
        mock_hf_dl.assert_called_once_with(
            repo_id="repo", filename="model.safetensors", cache_dir=self.models_dir,
            force_download=False, resume_download=True, local_files_only=False,
            local_dir_use_symlinks=False, repo_type="model"
        )

    @patch("llamasearch.setup.logger") # Added mock_sut_logger
    @patch("llamasearch.setup.hf_hub_download")
    @patch("llamasearch.setup.time.sleep", return_value=None) 
    def test_download_helper_retry_success(self, mock_sleep, mock_hf_dl, mock_sut_logger):
        """Test download_file_with_retry succeeds after retrying."""
        expected_path = "/fake/path/model.safetensors"
        mock_hf_dl.side_effect = [
            ConnectionError("Network flaky"), 
            expected_path 
        ]
        with patch("pathlib.Path.is_file", return_value=True): # Assume is_file is True on successful download
             result = download_file_with_retry("repo", "model.safetensors", self.models_dir, False, max_retries=1, delay=1)

        self.assertEqual(result, expected_path)
        self.assertEqual(mock_hf_dl.call_count, 2)
        mock_sleep.assert_called_once_with(1)
        mock_sut_logger.warning.assert_any_call(
            "DL attempt 1 for model.safetensors failed: Network flaky"
        )

    @patch("llamasearch.setup.logger") # Added mock_sut_logger
    @patch("llamasearch.setup.hf_hub_download")
    @patch("llamasearch.setup.time.sleep", return_value=None)
    def test_download_helper_retry_fail(self, mock_sleep, mock_hf_dl, mock_sut_logger): # Added mock_sut_logger
        """Test download_file_with_retry fails after all retries."""
        mock_hf_dl.side_effect = ConnectionError("Network down") 
        with self.assertRaisesRegex(SetupError, "Failed DL after retries: config.json"):
             download_file_with_retry("repo", "config.json", self.models_dir, False, max_retries=1, delay=1)
        self.assertEqual(mock_hf_dl.call_count, 2) 
        self.assertEqual(mock_sleep.call_count, 1)


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)