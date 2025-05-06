import unittest
import os
import json
import tarfile
import tempfile
import shutil
from pathlib import Path
import time

# Import the module under test
import llamasearch.data_manager as dm_module

# Pull in the classes and constants
DataManager = dm_module.DataManager
SETTINGS_FILENAME = dm_module.SETTINGS_FILENAME

class TestDataManager(unittest.TestCase):

    def setUp(self):
        """Create a temporary directory for each test and patch DEFAULT_SETTINGS."""
        # 1. Make a clean temp dir
        self.temp_dir = Path(tempfile.mkdtemp(prefix="test_llamasearch_"))

        # 2. Monkey‐patch DEFAULT_SETTINGS so that all defaults land under our temp dir
        original_defaults = dm_module.DEFAULT_SETTINGS.copy()
        dm_module.DEFAULT_SETTINGS = {
            key: str(self.temp_dir / Path(path).name)
            for key, path in original_defaults.items()
        }

        # 3. Initialize a fresh DataManager (it will use our patched DEFAULT_SETTINGS)
        self.dm = DataManager(base_dir=self.temp_dir)

    def tearDown(self):
        """Remove the temporary directory after each test."""
        shutil.rmtree(self.temp_dir)

    def test_ensure_directories(self):
        """Test ensure_directories creates missing directories from settings."""
        # Add an extra key that shouldn't be auto‐created
        extra_dir = self.temp_dir / "extra_stuff"
        self.dm.settings["extra"] = str(extra_dir)

        # Remove one of the default dirs to simulate "missing"
        models_path = Path(self.dm.settings["models"])
        if models_path.exists():
            shutil.rmtree(models_path)
        self.assertFalse(models_path.exists())

        # Re‐create via ensure_directories
        self.dm.ensure_directories()
        self.assertTrue(models_path.exists())

        # Test that set_data_path also ensures directories
        new_index = self.temp_dir / "specific_index"
        self.dm.set_data_path("index", str(new_index))
        self.assertTrue(new_index.exists())

    def test_get_data_paths(self):
        """Test getting the dictionary of data paths."""
        paths = self.dm.get_data_paths()
        self.assertEqual(paths["base"], str(self.temp_dir))
        for key in dm_module.DEFAULT_SETTINGS.keys():
            self.assertEqual(paths[key], self.dm.settings[key])
            self.assertIsInstance(paths[key], str)

    def test_set_data_path_valid_key(self):
        """Test setting a data path for a valid key updates settings and saves."""
        new_crawl = self.temp_dir / "crawled_web_data"
        self.dm.set_data_path("crawl_data", str(new_crawl))

        # In‐memory update
        self.assertEqual(self.dm.settings["crawl_data"], str(new_crawl))
        # Directory was created
        self.assertTrue(new_crawl.exists() and new_crawl.is_dir())
        # Settings file was written
        settings_file = self.temp_dir / SETTINGS_FILENAME
        self.assertTrue(settings_file.exists())
        with open(settings_file, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        self.assertEqual(loaded.get("crawl_data"), str(new_crawl))

    def test_set_data_path_invalid_key(self):
        """Test setting a data path for an invalid key raises ValueError."""
        with self.assertRaises(ValueError):
            self.dm.set_data_path("not_a_key", str(self.temp_dir / "dummy"))

    def _create_dummy_data(self, key):
        """Helper to create dummy file in a data directory."""
        data_dir = Path(self.dm.settings[key])
        data_dir.mkdir(parents=True, exist_ok=True)
        filename = f"dummy_{key}.txt"
        with open(data_dir / filename, "w") as f:
            f.write(f"content for {key}")
        return filename

    def test_export_data_default_filename(self):
        """Test exporting data with a default generated filename."""
        keys = ["crawl_data", "logs"]
        dummy_files = {k: self._create_dummy_data(k) for k in keys}

        # Ensure timestamp moves forward
        time.sleep(0.01)
        start_ts = time.strftime("%Y%m%d_%H%M%S")

        archive_str = self.dm.export_data(keys)
        archive = Path(archive_str)

        # Filename checks
        self.assertTrue(archive.name.startswith("llamasearch_export_"))
        self.assertTrue(archive.name.endswith(".tar.gz"))
        self.assertGreaterEqual(archive.name, f"llamasearch_export_{start_ts}.tar.gz")
        self.assertEqual(archive.parent, self.temp_dir)
        self.assertTrue(archive.exists())

        # Inspect contents
        with tarfile.open(archive, "r:gz") as tar:
            members = tar.getnames()
            expected_dirs = [Path(self.dm.settings[k]).name for k in keys]
            expected_files = [
                f"{Path(self.dm.settings[k]).name}/{dummy_files[k]}"
                for k in keys
            ]
            for d in expected_dirs:
                self.assertIn(d, members)
            for f in expected_files:
                self.assertIn(f, members)

    def test_export_data_empty_keys(self):
        """Test exporting with no keys raises ValueError."""
        with self.assertRaisesRegex(ValueError, "No keys specified"):
            self.dm.export_data([])

    def test_export_data_invalid_keys(self):
        """Test exporting with only bad keys raises ValueError."""
        with self.assertRaisesRegex(ValueError, "No valid paths found"):
            self.dm.export_data(["bad1", "bad2"])

    def test_export_data_missing_source_dir(self):
        """Test exporting when a source dir doesn't exist raises FileNotFoundError."""
        missing = self.temp_dir / "i_dont_exist"
        # Point logs at a nonexistent path
        self.dm.settings["logs"] = str(missing)
        if missing.exists():
            shutil.rmtree(missing)
        self.assertFalse(missing.exists())

        with self.assertRaises(FileNotFoundError):
            self.dm.export_data(["logs"])


if __name__ == "__main__":
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
