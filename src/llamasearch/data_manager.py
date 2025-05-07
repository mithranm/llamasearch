# src/llamasearch/data_manager.py
"""
data_manager.py - Dynamic configuration and export utilities for LlamaSearch.

This module stores paths for crawl data, index, models, and logs in a settings
dictionary that is loaded from (and saved to) a JSON file in the base directory.
Users can change these settings at runtime without exiting LlamaSearch.
Additionally, the module provides an export method that packages specified
directories into a tar.gz archive.
"""

import json
import tarfile
import time
from pathlib import Path
from typing import Optional

# Define the fixed base directory
_DEFAULT_BASE_DIR = Path.home() / ".llamasearch"

DEFAULT_SETTINGS = {
    "crawl_data": str(_DEFAULT_BASE_DIR / "crawl_data"),
    "index": str(_DEFAULT_BASE_DIR / "index"),
    "models": str(_DEFAULT_BASE_DIR / "models"),
    "logs": str(_DEFAULT_BASE_DIR / "logs"),
}

SETTINGS_FILENAME = "settings.json"


class DataManager:
    def __init__(self, base_dir: Optional[Path] = None):
        # Use a base_dir if given (primarily for testing), otherwise default to ~/.llamasearch
        # Removed environment variable check
        self.base_dir = base_dir if base_dir else _DEFAULT_BASE_DIR
        self.settings_file = self.base_dir / SETTINGS_FILENAME
        self.settings = DEFAULT_SETTINGS.copy()

        # Ensure base dir uses the fixed default if not provided for testing
        if not base_dir:
            for key in DEFAULT_SETTINGS:
                # Ensure default paths are relative to the actual default base dir
                self.settings[key] = str(
                    _DEFAULT_BASE_DIR / Path(DEFAULT_SETTINGS[key]).name
                )

        self._load_settings()
        self.ensure_directories()

    def _load_settings(self):
        if self.settings_file.exists():
            try:
                with open(self.settings_file, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    # Update only known keys to avoid unexpected settings
                    for key in DEFAULT_SETTINGS:
                        if key in loaded:
                            self.settings[key] = loaded[key]
            except Exception as e:
                print(f"Warning: could not load settings file: {e}")
        else:
            # If settings file doesn't exist, ensure settings dict reflects the default base
            for key in DEFAULT_SETTINGS:
                self.settings[key] = str(
                    self.base_dir / Path(DEFAULT_SETTINGS[key]).name
                )

    def save_settings(self):
        self.base_dir.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.settings_file, "w", encoding="utf-8") as f:
                json.dump(self.settings, f, indent=2)
        except Exception as e:
            print(f"Error saving settings: {e}")

    def ensure_directories(self):
        # Ensure the base directory itself exists first
        self.base_dir.mkdir(parents=True, exist_ok=True)
        for key in DEFAULT_SETTINGS.keys():  # Iterate through known keys
            dir_path_str = self.settings.get(key)
            if dir_path_str:
                dir_path = Path(dir_path_str)
                if not dir_path.exists():
                    dir_path.mkdir(parents=True, exist_ok=True)
            else:
                # If somehow a default key is missing from settings, log warning
                print(f"Warning: Path for default setting key '{key}' is missing.")

    def get_data_paths(self) -> dict:
        """Return the current data paths for all key directories."""
        # Ensure paths returned are consistent with the DataManager's base directory
        return {
            "base": str(self.base_dir),
            "crawl_data": self.settings.get(
                "crawl_data", str(self.base_dir / "crawl_data")
            ),
            "index": self.settings.get("index", str(self.base_dir / "index")),
            "models": self.settings.get("models", str(self.base_dir / "models")),
            "logs": self.settings.get("logs", str(self.base_dir / "logs")),
        }

    def set_data_path(self, key: str, path: str):
        """
        Update a given path (e.g., "crawl_data", "index", etc.).
        This change is saved immediately.
        """
        if key in DEFAULT_SETTINGS:
            new_path = Path(path).resolve()  # Store absolute path
            self.settings[key] = str(new_path)
            # Ensure the newly set directory exists
            if not new_path.exists():
                new_path.mkdir(parents=True, exist_ok=True)
            self.save_settings()
        else:
            raise ValueError(f"Unknown data path key: {key}")

    def export_data(self, keys: list, output_file: Optional[str] = None) -> str:
        """
        Export the directories specified in keys (e.g., ["crawl_data", "index"]) into a tar.gz archive.
        If output_file is not provided, creates one with a timestamp in the base directory.
        Returns the path to the archive.
        """
        if not keys:
            raise ValueError("No keys specified for export.")
        export_paths = []
        for key in keys:
            path_str = self.settings.get(key, "")
            if path_str:
                p = Path(path_str)
                if p.exists() and p.is_dir():  # Check existence before adding
                    export_paths.append(p)
                else:
                    print(
                        f"Warning: Directory for key '{key}' ({path_str}) does not exist. Skipping export."
                    )

        if not export_paths:
            raise ValueError(
                "No valid, existing directories found for export based on provided keys."
            )

        if not output_file:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            # Save export in the base LlamaSearch directory
            output_file = str(self.base_dir / f"llamasearch_export_{timestamp}.tar.gz")
        else:
            # Ensure output path is absolute
            output_file_path = Path(output_file).resolve()
            # Create parent directory if it doesn't exist
            output_file_path.parent.mkdir(parents=True, exist_ok=True)
            output_file = str(output_file_path)

        try:
            with tarfile.open(output_file, "w:gz") as tar:
                for exp_path in export_paths:
                    # arcname=exp_path.name stores the directory with its name as root in the archive
                    tar.add(str(exp_path), arcname=exp_path.name)
            print(f"Data exported successfully to: {output_file}")
            return output_file
        except Exception as e:
            print(f"Error during data export: {e}")
            raise  # Re-raise the exception


# Singleton instance for convenience
data_manager = DataManager()
