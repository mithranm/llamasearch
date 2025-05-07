# tests/test_data_manager.py
import json
# import shutil # Not strictly needed if self.temp_dir handles cleanup
import tarfile
import tempfile
import time
import unittest
from pathlib import Path

# Import the module under test
import llamasearch.data_manager as dm_module

# Pull in the classes and constants
DataManager = dm_module.DataManager
SETTINGS_FILENAME = dm_module.SETTINGS_FILENAME
# Store original defaults to restore them after tests
ORIGINAL_DEFAULT_SETTINGS_BACKUP = dm_module.DEFAULT_SETTINGS.copy()


class TestDataManager(unittest.TestCase):

    def setUp(self):
        """Create a temporary directory for each test and patch DEFAULT_SETTINGS."""
        self.temp_dir_obj = tempfile.TemporaryDirectory(prefix="test_llamasearch_")
        self.temp_dir = Path(self.temp_dir_obj.name)

        # Monkey‐patch DEFAULT_SETTINGS so that all defaults land under our temp dir
        # Use a deep copy of the original defaults for patching
        current_original_defaults = ORIGINAL_DEFAULT_SETTINGS_BACKUP.copy()
        dm_module.DEFAULT_SETTINGS = {
            key: str(
                self.temp_dir / Path(path).name
            )  # Ensure paths are relative to temp_dir
            for key, path in current_original_defaults.items()
        }
        # Initialize a fresh DataManager (it will use our patched DEFAULT_SETTINGS)
        # Pass base_dir to ensure it uses the temp dir
        self.dm = DataManager(base_dir=self.temp_dir)

    def tearDown(self):
        """Remove the temporary directory and restore original DEFAULT_SETTINGS."""
        self.temp_dir_obj.cleanup()
        # Restore original DEFAULT_SETTINGS to avoid interference between test files/runs
        dm_module.DEFAULT_SETTINGS = ORIGINAL_DEFAULT_SETTINGS_BACKUP.copy()

    def test_ensure_directories(self):
        """Test ensure_directories creates missing directories from settings."""
        # Add an extra key that shouldn't be auto‐created by default logic
        # (unless explicitly called for that key if set_data_path supported unknown keys)
        # self.dm.settings["extra"] = str(self.temp_dir / "extra_stuff")

        # Remove one of the default dirs to simulate "missing"
        models_path = Path(self.dm.settings["models"])
        if models_path.exists():
            # Use shutil from pathlib for rmtree
            import shutil

            shutil.rmtree(models_path)
        self.assertFalse(models_path.exists())

        # Re‐create via ensure_directories
        self.dm.ensure_directories()
        self.assertTrue(models_path.exists())

        # Test that set_data_path also ensures directories
        new_index_path_str = str(self.temp_dir / "specific_index")
        self.dm.set_data_path("index", new_index_path_str)
        self.assertTrue(Path(new_index_path_str).exists())

    def test_get_data_paths(self):
        """Test getting the dictionary of data paths."""
        paths = self.dm.get_data_paths()
        self.assertEqual(paths["base"], str(self.temp_dir))
        for (
            key
        ) in (
            dm_module.DEFAULT_SETTINGS.keys()
        ):  # Iterate over the patched DEFAULT_SETTINGS
            self.assertEqual(paths[key], self.dm.settings[key])
            self.assertIsInstance(paths[key], str)

    def test_set_data_path_valid_key(self):
        """Test setting a data path for a valid key updates settings and saves."""
        new_crawl_path_str = str(self.temp_dir / "crawled_web_data")
        self.dm.set_data_path("crawl_data", new_crawl_path_str)

        # In‐memory update
        self.assertEqual(self.dm.settings["crawl_data"], new_crawl_path_str)
        # Directory was created
        self.assertTrue(
            Path(new_crawl_path_str).exists() and Path(new_crawl_path_str).is_dir()
        )
        # Settings file was written
        settings_file = self.temp_dir / SETTINGS_FILENAME
        self.assertTrue(settings_file.exists())
        with open(settings_file, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        self.assertEqual(loaded.get("crawl_data"), new_crawl_path_str)

    def test_set_data_path_invalid_key(self):
        """Test setting a data path for an invalid key raises ValueError."""
        with self.assertRaisesRegex(ValueError, "Unknown data path key: not_a_key"):
            self.dm.set_data_path("not_a_key", str(self.temp_dir / "dummy"))

    def _create_dummy_data(self, key):
        """Helper to create dummy file in a data directory."""
        data_dir = Path(self.dm.settings[key])
        data_dir.mkdir(parents=True, exist_ok=True)
        filename = f"dummy_{key}.txt"
        (data_dir / filename).write_text(f"content for {key}", encoding="utf-8")
        return filename

    def test_export_data_default_filename(self):
        """Test exporting data with a default generated filename."""
        keys = ["crawl_data", "logs"]
        dummy_files = {k: self._create_dummy_data(k) for k in keys}

        # Ensure timestamp moves forward
        time.sleep(0.01)
        start_ts_obj = time.localtime()  # Get struct_time
        # Format specifically to match the SUT's strftime format
        start_ts_str = time.strftime("%Y%m%d_%H%M%S", start_ts_obj)

        archive_str_path = self.dm.export_data(keys)
        archive_path = Path(archive_str_path)

        # Filename checks
        self.assertTrue(archive_path.name.startswith("llamasearch_export_"))
        self.assertTrue(archive_path.name.endswith(".tar.gz"))
        # Compare the timestamp part more carefully
        archive_ts_str = archive_path.name.replace("llamasearch_export_", "").replace(
            ".tar.gz", ""
        )
        self.assertGreaterEqual(archive_ts_str, start_ts_str)

        self.assertEqual(
            archive_path.parent, self.temp_dir
        )  # Export should be in base_dir
        self.assertTrue(archive_path.exists())

        # Inspect contents
        with tarfile.open(archive_path, "r:gz") as tar:
            members = tar.getnames()
            # Expected directory names inside the tar (arcname=exp_path.name)
            expected_arc_dirs = [Path(self.dm.settings[k]).name for k in keys]
            # Expected file paths inside the tar
            expected_arc_files = [
                f"{Path(self.dm.settings[k]).name}/{dummy_files[k]}" for k in keys
            ]
            for d_arc in expected_arc_dirs:
                self.assertIn(d_arc, members)
            for f_arc in expected_arc_files:
                self.assertIn(f_arc, members)

    def test_export_data_empty_keys(self):
        """Test exporting with no keys raises ValueError."""
        with self.assertRaisesRegex(ValueError, "No keys specified for export."):
            self.dm.export_data([])

    def test_export_data_invalid_keys(self):
        """Test exporting with only bad keys raises ValueError."""
        # Corrected regex to match the actual error message
        with self.assertRaisesRegex(
            ValueError,
            "No valid, existing directories found for export based on provided keys.",
        ):
            self.dm.export_data(["bad1", "bad2"])

    def test_export_data_missing_source_dir(self):
        """Test exporting when a source dir doesn't exist raises ValueError."""
        missing_dir_path = self.temp_dir / "i_dont_exist"
        # Point logs at a nonexistent path
        self.dm.settings["logs"] = str(missing_dir_path)
        if missing_dir_path.exists():
            # Use shutil from pathlib for rmtree
            import shutil

            shutil.rmtree(missing_dir_path)
        self.assertFalse(missing_dir_path.exists())

        # Expect ValueError as no valid paths will be found
        with self.assertRaisesRegex(
            ValueError,
            "No valid, existing directories found for export based on provided keys.",
        ):
            self.dm.export_data(["logs"])


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
