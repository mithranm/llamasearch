#!/usr/bin/env python3
"""
setup.py - Downloads and verifies LlamaSearch models (CPU-Only, FP32).

Configured for the generic ONNX Causal LM (Llama 3.2 1B - FP32) and embedder.
Downloads necessary files and assembles the active model directory.
"""

import argparse
import gc
import shutil
import sys
import time
from pathlib import Path
from typing import List, Optional

from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.errors import (EntryNotFoundError, HfHubHTTPError,
                                    LocalEntryNotFoundError)
from huggingface_hub.utils._hf_folder import HfFolder

from llamasearch.core.embedder import \
    DEFAULT_MODEL_NAME as DEFAULT_EMBEDDER_MODEL
from llamasearch.core.embedder import EnhancedEmbedder
from llamasearch.core.onnx_model import \
    _ONNX_ORIGINAL_BASE_FILES_  # Import the correct list
from llamasearch.core.onnx_model import (MODEL_ONNX_BASENAME,
                                         ONNX_MODEL_REPO_ID, ONNX_SUBFOLDER,
                                         GenericONNXLLM, load_onnx_llm)
from llamasearch.data_manager import data_manager
from llamasearch.exceptions import ModelNotFoundError, SetupError
from llamasearch.utils import setup_logging

logger = setup_logging("llamasearch.setup")

# These files are typically needed for AutoTokenizer and model loading config
# Use the list defined in onnx_model.py
REQUIRED_ROOT_FILES = _ONNX_ORIGINAL_BASE_FILES_


# Helper: Download with Retries
def download_file_with_retry(
    repo_id: str,
    filename: str,
    cache_dir: Path,
    force: bool,
    max_retries: int = 2,
    delay: int = 5,
    repo_type: Optional[str] = "model",
    **kwargs,
) -> str:
    """Attempts to download a file with retries on failure."""
    assert isinstance(cache_dir, Path)
    last_error : Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            logger.debug(
                f"Attempt {attempt + 1} DL: {filename} from {repo_id} ({repo_type})"
            )
            file_path_str = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=cache_dir,
                force_download=force,
                resume_download=not force,
                local_files_only=False,
                local_dir_use_symlinks=False,
                repo_type=repo_type,
                **kwargs,
            )
            if not file_path_str:
                 raise FileNotFoundError(f"hf_hub_download returned invalid path: {file_path_str}")

            fpath = Path(file_path_str)
            if not fpath.is_file():
                raise FileNotFoundError(
                    f"File {filename} invalid or DNE after DL attempt {attempt + 1} at path {fpath}."
                )
            logger.debug(f"OK DL {filename} -> {file_path_str}")
            return file_path_str
        except EntryNotFoundError as e:
            logger.debug(
                f"File {filename} not in repo {repo_id} (attempt {attempt + 1})."
            )
            last_error = e
            raise # Reraise immediately, no retry for 404
        except (ConnectionError, TimeoutError, HfHubHTTPError, FileNotFoundError) as e:
            logger.warning(f"DL attempt {attempt + 1} for {filename} failed: {e}")
            last_error = e
            if attempt < max_retries:
                logger.info(f"Retrying download of {filename} in {delay} seconds...")
                time.sleep(delay)
            else:
                raise SetupError(f"Failed DL after retries: {filename}") from last_error
        except Exception as e:
            logger.error(
                f"Unexpected DL error {filename} attempt {attempt + 1}: {e}",
                exc_info=True,
            )
            last_error = e
            raise SetupError(f"Failed DL unexpected: {filename}") from last_error
    # Should not be reachable
    raise SetupError(f"Download logic error for {filename}.")


# --- Embedder Check/Download ---
def check_or_download_embedder(models_dir: Path, force: bool = False) -> None:
    model_name = DEFAULT_EMBEDDER_MODEL
    logger.info(f"Checking/Downloading Embedder: {model_name}")
    ignore_patterns: List[str] = ["*.onnx", "onnx/*", "*.gguf", "gguf/*", "openvino/*"]
    logger.info(f"Ignoring patterns: {ignore_patterns}")
    try:
        if not force:
            try:
                logger.debug(f"Checking locally for {model_name} in {models_dir}")
                snapshot_download(
                    repo_id=model_name,
                    cache_dir=models_dir,
                    local_files_only=True,
                    local_dir_use_symlinks=False,
                    ignore_patterns=ignore_patterns,
                )
                logger.info(f"Embedder '{model_name}' (PyTorch) found locally.")
                return
            except (EntryNotFoundError, LocalEntryNotFoundError, FileNotFoundError):
                logger.info(
                    f"Embedder '{model_name}' (PyTorch) not found/incomplete locally. Proceeding to download/verify..."
                )
            except Exception as local_check_err:
                 logger.warning(f"Local check for embedder failed ({local_check_err}), proceeding to download/verify...")

        logger.info(f"Downloading/Verifying embedder {model_name} from Hub...")
        snapshot_download(
            repo_id=model_name,
            cache_dir=models_dir,
            force_download=force,
            resume_download=not force,
            local_files_only=False,
            local_dir_use_symlinks=False,
            ignore_patterns=ignore_patterns,
        )
        logger.info(f"Embedder '{model_name}' (PyTorch) cache verified/downloaded.")
    except Exception as e:
        raise SetupError(f"Failed get/verify embedder {model_name}") from e


# --- Generic ONNX LLM Check/Download ---
def check_or_download_onnx_llm(models_dir: Path, force: bool = False) -> None:
    """Downloads required generic ONNX LLM files (FP32 only)."""
    logger.info("Checking/Downloading Generic ONNX LLM (FP32)")

    active_model_dir = models_dir / "active_model"
    active_onnx_dir = active_model_dir / ONNX_SUBFOLDER
    needs_clean = force

    if not needs_clean and active_model_dir.exists():
        target_onnx_file = active_onnx_dir / f"{MODEL_ONNX_BASENAME}.onnx"
        target_data_file = active_onnx_dir / f"{MODEL_ONNX_BASENAME}.onnx_data"

        if not target_onnx_file.is_file():
            needs_clean = True
            logger.warning(
                f"Target ONNX file '{target_onnx_file.name}' missing in {active_onnx_dir}. Forcing clean assembly."
            )
        elif not target_data_file.is_file():
            needs_clean = True
            logger.warning(
                f"Target ONNX data file '{target_data_file.name}' missing in {active_onnx_dir}. Forcing clean assembly."
            )

    if needs_clean and active_model_dir.exists():
        logger.info(f"Cleaning existing active model directory: {active_model_dir}")
        try:
            shutil.rmtree(active_model_dir)
        except OSError as e:
            raise SetupError(f"Failed clear active dir {active_model_dir}: {e}") from e

    active_model_dir.mkdir(parents=True, exist_ok=True)
    active_onnx_dir.mkdir(parents=True, exist_ok=True)

    cache_location = models_dir
    repo_id = ONNX_MODEL_REPO_ID

    # 1. Download/Verify required root files
    logger.info(f"Downloading/Verifying root files from {repo_id}...")
    try:
        for filename in REQUIRED_ROOT_FILES:
            logger.debug(f"Processing root file: {filename}")
            cached_path_str = download_file_with_retry(
                repo_id=repo_id,
                filename=filename,
                cache_dir=cache_location,
                force=force,
                repo_type="model",
            )
            target_path = active_model_dir / filename
            logger.debug(f"Copying {cached_path_str} to {target_path}")
            shutil.copyfile(cached_path_str, target_path)
        logger.info(f"Root config/tokenizer files processed into {active_model_dir}")
    except EntryNotFoundError as e:
         logger.error(f"A required configuration file is missing from the repository: {e}")
         raise SetupError(f"Required config file missing from repo: {e.filename}") from e
    except Exception as e:
        raise SetupError(f"Failed to process root files from {repo_id}: {e}") from e

    # 2. Download FP32 ONNX Model File (to cache) and copy
    onnx_model_rel_path = f"{ONNX_SUBFOLDER}/{MODEL_ONNX_BASENAME}.onnx"
    logger.info(f"Downloading/Verifying ONNX model file: '{onnx_model_rel_path}'...")
    try:
        source_path_str = download_file_with_retry(
            repo_id=repo_id,
            filename=onnx_model_rel_path,
            cache_dir=cache_location,
            force=force,
            repo_type="model",
        )
        target_onnx_path = active_onnx_dir / Path(onnx_model_rel_path).name
        logger.debug(f"Copying {source_path_str} to {target_onnx_path}")
        shutil.copyfile(source_path_str, target_onnx_path)
        logger.debug(f"Copied {onnx_model_rel_path} to {target_onnx_path}")
    except EntryNotFoundError:
        logger.error(
            f"Required ONNX file '{onnx_model_rel_path}' not found in repo {repo_id}."
        )
        raise SetupError(f"Required ONNX file missing: {onnx_model_rel_path}")
    except (SetupError, IOError, OSError) as e:
        raise SetupError(f"Error getting/copying ONNX file {onnx_model_rel_path}: {e}") from e
    except Exception as e:
        raise SetupError(f"Unexpected error processing ONNX file {onnx_model_rel_path}: {e}") from e

    # 3. Download the common ONNX Data File (to cache) and copy
    onnx_data_rel_path = f"{ONNX_SUBFOLDER}/{MODEL_ONNX_BASENAME}.onnx_data"
    logger.info(f"Downloading/Verifying ONNX data file: '{onnx_data_rel_path}'...")
    try:
        cached_data_file_str = download_file_with_retry(
            repo_id=repo_id,
            filename=onnx_data_rel_path,
            cache_dir=cache_location,
            force=force,
            repo_type="model",
        )
        target_data_path = active_onnx_dir / Path(onnx_data_rel_path).name
        logger.debug(f"Copying {cached_data_file_str} to {target_data_path}")
        shutil.copyfile(cached_data_file_str, target_data_path)
        logger.info(f"Processed ONNX data file: copied to {target_data_path.name}")

    except EntryNotFoundError:
        logger.error(
            f"Required ONNX data file '{onnx_data_rel_path}' not found in repo {repo_id}."
        )
        raise SetupError(
            f"Required ONNX data file '{onnx_data_rel_path}' missing from repository."
        )
    except (SetupError, IOError, OSError) as e:
        raise SetupError(
            f"Error getting/copying ONNX data file {onnx_data_rel_path}: {e}"
        ) from e
    except Exception as e:
        raise SetupError(
            f"Unexpected error processing ONNX data file {onnx_data_rel_path}: {e}"
        ) from e

    logger.info(f"Assembly of {active_model_dir} complete for FP32.")


# --- Verification Function ---
def verify_setup():
    """Attempts to load models based on the installed files in active_model."""
    logger.info("--- Verifying Model Setup (CPU-Only, FP32) ---")
    all_verified = True
    embedder = None
    llm = None

    # Verify Embedder
    logger.info(f"Verifying Embedder '{DEFAULT_EMBEDDER_MODEL}' (CPU)...")
    try:
        # Pass default batch size, let embedder handle config
        embedder = EnhancedEmbedder(batch_size=0)
        dim = embedder.get_embedding_dimension()
        if not (dim and isinstance(dim, int) and dim > 0 and embedder.model is not None):
             raise SetupError(f"Embedder invalid state (Model:{embedder.model is not None}, Dim:{dim}). Check logs.")
        logger.info(f"Embedder OK (CPU, Dim: {dim}).")
    except ModelNotFoundError as e:
        logger.error(f"FAIL: Embedder model not found. {e}")
        all_verified = False
        raise SetupError(f"Embedder model files not found: {e}") from e
    except Exception as e:
        logger.error(f"FAIL: Error loading CPU embedder: {e}", exc_info=True)
        all_verified = False
        raise SetupError(f"Failed to initialize embedder: {e}") from e
    finally:
        if embedder:
            try:
                embedder.close()
            except Exception as close_e:
                logger.warning(f"Error closing embedder during verification: {close_e}")
            del embedder
            gc.collect()

    # Verify Generic ONNX LLM (CPU, FP32)
    logger.info("Verifying Generic ONNX LLM (CPU, FP32 in active_model)...")
    try:
        # Load without specifying quantization, it will check for base files
        llm = load_onnx_llm()
        if llm and isinstance(llm, GenericONNXLLM):
            logger.info(
                f"ONNX LLM OK ({llm.model_info.model_id} loaded from active_model on CPU)."
            )
        else:
            raise SetupError("ONNX LLM loader returned None or unexpected type during verification.")
    except ModelNotFoundError as e:
        logger.error(f"FAIL: ONNX LLM files missing or incomplete in active_model. {e}")
        all_verified = False
        raise SetupError(f"ONNX LLM model files not found in active_model: {e}") from e
    except Exception as e:
        logger.error(
            f"FAIL: Error loading ONNX LLM from active_model: {e}", exc_info=True
        )
        all_verified = False
        raise SetupError(f"Failed to initialize ONNX LLM from active_model: {e}") from e
    finally:
        if llm:
            try:
                llm.unload()
            except Exception as unload_e:
                logger.warning(f"Error unloading LLM during verification: {unload_e}")
            del llm
            gc.collect()

    if not all_verified:
        raise SetupError("One or more models failed CPU verification.")
    else:
        logger.info("--- Model Verification Successful (CPU-Only, FP32) ---")


# --- Main Setup Function ---
def main():
    parser = argparse.ArgumentParser(
        description="Download/verify models for LlamaSearch (CPU-Only, FP32)."
    )
    parser.add_argument(
        "--force", action="store_true", help="Force redownload/reassembly"
    )
    # Removed quantization flags

    args = parser.parse_args()

    logger.info("--- Starting LlamaSearch Model Setup (CPU-Only, FP32) ---")
    if args.force:
        logger.info("Force mode enabled: Active directory will be recreated.")

    try:
        logger.info("Setup configured for CPU-only operation (FP32 ONNX).")
        models_dir_str = data_manager.get_data_paths().get("models")
        if not models_dir_str:
            raise SetupError("Models directory path not configured.")
        models_dir = Path(models_dir_str)
        models_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Using models directory: {models_dir}")

        try:
            if HfFolder.get_token():
                logger.info("HF token found.")
            else:
                logger.warning("HF token not found. Downloads might fail for gated models (not applicable to default models).")
        except Exception as token_err:
            logger.warning(f"Could not check HF token: {token_err}")

        # Download/Verify components
        check_or_download_embedder(models_dir, args.force)
        check_or_download_onnx_llm(models_dir, args.force) # No suffix needed

        # Final Verification (CPU)
        verify_setup() # No suffix needed
        logger.info(
            "--- LlamaSearch Model Setup Completed Successfully (FP32) ---"
        )
        sys.exit(0)

    except SetupError as e:
        logger.error(f"Setup failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during setup: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()