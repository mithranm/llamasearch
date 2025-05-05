#!/usr/bin/env python3
"""
data_manager.py – Dynamic configuration and export utilities for LlamaSearch.

This module stores paths for crawl data, index, models, and logs in a settings
dictionary that is loaded from (and saved to) a JSON file in the base directory.
Users can change these settings at runtime without exiting LlamaSearch.
Additionally, the module provides an export method that packages specified
directories into a tar.gz archive.
"""

import json
import os
import tarfile
import time
from pathlib import Path
from typing import Optional

DEFAULT_SETTINGS = {
    "crawl_data": os.path.join(os.path.expanduser("~"), ".llamasearch", "crawl_data"),
    "index": os.path.join(os.path.expanduser("~"), ".llamasearch", "index"),
    "models": os.path.join(os.path.expanduser("~"), ".llamasearch", "models"),
    "logs": os.path.join(os.path.expanduser("~"), ".llamasearch", "logs")
}

SETTINGS_FILENAME = "settings.json"

class DataManager:
    def __init__(self, base_dir: Optional[Path] = None):
        # Use a base_dir if given, otherwise default to ~/.llamasearch
        self.base_dir = base_dir if base_dir else Path(os.environ.get("LLAMASEARCH_DATA_DIR",
                                                                       os.path.join(os.path.expanduser("~"), ".llamasearch")))
        self.settings_file = self.base_dir / SETTINGS_FILENAME
        self.settings = DEFAULT_SETTINGS.copy()
        self._load_settings()
        self.ensure_directories()

    def _load_settings(self):
        if self.settings_file.exists():
            try:
                with open(self.settings_file, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    self.settings.update(loaded)
            except Exception as e:
                print(f"Warning: could not load settings file: {e}")

    def save_settings(self):
        self.base_dir.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.settings_file, "w", encoding="utf-8") as f:
                json.dump(self.settings, f, indent=2)
        except Exception as e:
            print(f"Error saving settings: {e}")

    def ensure_directories(self):
        for key in ["crawl_data", "index", "models", "logs"]:
            dir_path = Path(self.settings.get(key, ""))
            if not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)

    def get_data_paths(self) -> dict:
        """Return the current data paths for all key directories."""
        return {
            "base": str(self.base_dir),
            "crawl_data": self.settings.get("crawl_data", ""),
            "index": self.settings.get("index", ""),
            "models": self.settings.get("models", ""),
            "logs": self.settings.get("logs", "")
        }

    def set_data_path(self, key: str, path: str):
        """
        Update a given path (e.g., "crawl_data", "index", etc.).
        This change is saved immediately.
        """
        if key in DEFAULT_SETTINGS:
            self.settings[key] = path
            self.ensure_directories()
            self.save_settings()
        else:
            raise ValueError(f"Unknown data path key: {key}")

    def export_data(self, keys: list, output_file: Optional[str] = None) -> str:
        """
        Export the directories specified in keys (e.g., ["crawl_data", "index"]) into a tar.gz archive.
        If output_file is not provided, creates one with a timestamp.
        Returns the path to the archive.
        """
        if not keys:
            raise ValueError("No keys specified for export.")
        export_paths = []
        for key in keys:
            path_str = self.settings.get(key, "")
            if path_str:
                export_paths.append(Path(path_str))
        if not export_paths:
            raise ValueError("No valid paths found for export.")

        if not output_file:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_file = str(self.base_dir / f"llamasearch_export_{timestamp}.tar.gz")
        else:
            output_file = str(Path(output_file).resolve())

        with tarfile.open(output_file, "w:gz") as tar:
            for exp_path in export_paths:
                tar.add(exp_path, arcname=exp_path.name)
        return output_file

# Singleton instance for convenience
data_manager = DataManager()
