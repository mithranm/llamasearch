# FIX: src/llamasearch/setup.py
# Added '_select_onnx_quantization' to imports from core.onnx_model

#!/usr/bin/env python3
"""
setup.py - Downloads and verifies LlamaSearch models (CPU-Only).

Configured for a generic ONNX Causal LM (Qwen3-0.6B) and embedder.
Downloads necessary files and assembles the active model directory,
respecting the 'onnx/' subfolder structure for model files.
"""

import argparse
import shutil
import sys
import time
from pathlib import Path
from typing import List, Optional
import gc

from huggingface_hub import (
    hf_hub_download,
    snapshot_download,
)
from huggingface_hub.errors import (
    EntryNotFoundError,
    LocalEntryNotFoundError,
)
from huggingface_hub.utils._hf_folder import HfFolder

from llamasearch.core.embedder import DEFAULT_MODEL_NAME as DEFAULT_EMBEDDER_MODEL
from llamasearch.core.embedder import EnhancedEmbedder

# --- CHANGE: Import from onnx_model.py ---
from llamasearch.core.onnx_model import (
    MODEL_ONNX_BASENAME,
    ONNX_SUBFOLDER,
    ONNX_MODEL_REPO_ID,
    load_onnx_llm,
    _select_onnx_quantization,  # <<< FIX: Added import >>>
)
from llamasearch.core.onnx_model import LLM  # Import LLM protocol

# --- END CHANGE ---
from llamasearch.data_manager import data_manager
from llamasearch.exceptions import ModelNotFoundError, SetupError
from llamasearch.utils import setup_logging

logger = setup_logging("llamasearch.setup")


# Helper: Download with Retries (No functional change needed)
def download_file_with_retry(
    repo_id: str,
    filename: str,
    cache_dir: Path,
    force: bool,
    max_retries: int = 2,
    delay: int = 5,
    repo_type: Optional[str] = "model",
    **kwargs,
):
    """Attempts to download a file with retries on failure."""
    assert isinstance(cache_dir, Path)
    for attempt in range(max_retries + 1):
        try:
            logger.debug(
                f"Attempt {attempt + 1} DL: {filename} from {repo_id} ({repo_type})"
            )
            file_path = hf_hub_download(
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
            assert file_path is not None
            fpath = Path(file_path)
            if not fpath.exists():
                raise FileNotFoundError(
                    f"File {filename} invalid after DL {attempt + 1}."
                )
            logger.debug(f"OK DL {filename} -> {file_path}")
            return file_path
        except EntryNotFoundError:
            logger.debug(
                f"File {filename} not in repo {repo_id} (attempt {attempt + 1})."
            )
            raise
        except (ConnectionError, TimeoutError, FileNotFoundError) as e:
            logger.warning(f"DL attempt {attempt + 1} for {filename} failed: {e}")
            if attempt < max_retries:
                time.sleep(delay)
            else:
                raise SetupError(f"Failed DL after retries: {filename}") from e
        except Exception as e:
            logger.error(
                f"Unexpected DL error {filename} attempt {attempt + 1}: {e}",
                exc_info=True,
            )
            raise SetupError(f"Failed DL unexpected: {filename}") from e
    raise SetupError(f"DL logic error {filename}.")


# --- Embedder Check/Download (No functional change needed) ---
def check_or_download_embedder(models_dir: Path, force: bool = False) -> None:
    model_name = DEFAULT_EMBEDDER_MODEL
    logger.info(f"Checking/Downloading Embedder: {model_name}")
    ignore_patterns: List[str] = ["*.onnx", "onnx/*", "*.gguf", "gguf/*", "openvino/*"]
    logger.info(f"Ignoring patterns: {ignore_patterns}")
    try:
        if not force:
            try:
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
                    f"Embedder '{model_name}' (PyTorch) not found/incomplete. Downloading..."
                )
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
        raise SetupError(f"Failed get embedder {model_name}") from e


# --- Generic ONNX LLM Check/Download ---
def check_or_download_onnx_llm(
    models_dir: Path, quant_pref: str = "auto", force: bool = False
) -> None:
    """Downloads required generic ONNX LLM files and assembles the 'active_model' dir."""
    logger.info(f"Checking/Downloading Generic ONNX LLM (Quant Pref: {quant_pref})")
    # <<< FIX: Uses imported _select_onnx_quantization >>>
    quant_suffix = _select_onnx_quantization(quant_pref)
    logger.info(f"Targeting ONNX suffix: '{quant_suffix}' for assembly")

    active_model_dir = models_dir / "active_model"
    # --- Define path to ONNX subfolder within active_model ---
    active_onnx_dir = active_model_dir / ONNX_SUBFOLDER
    needs_clean = force

    if not needs_clean and active_model_dir.exists():
        # --- Check ONNX file existence within the active ONNX subfolder ---
        target_onnx_file = active_onnx_dir / f"{MODEL_ONNX_BASENAME}{quant_suffix}.onnx"
        if not target_onnx_file.exists():
            needs_clean = True
            logger.warning(
                f"Target ONNX file '{target_onnx_file.name}' missing in {active_onnx_dir}. Forcing clean."
            )

    if needs_clean and active_model_dir.exists():
        logger.info(f"Cleaning existing active model directory: {active_model_dir}")
        try:
            shutil.rmtree(active_model_dir)
        except OSError as e:
            raise SetupError(f"Failed clear active dir {active_model_dir}: {e}") from e

    # --- Ensure both active_model and active_onnx directories exist ---
    active_model_dir.mkdir(parents=True, exist_ok=True)
    active_onnx_dir.mkdir(parents=True, exist_ok=True)
    cache_location = models_dir
    repo_id = ONNX_MODEL_REPO_ID

    # 1. Download all root-level config/tokenizer files etc.
    logger.info(f"Downloading/Verifying root files from {repo_id}...")
    root_files_ignore = [f"{ONNX_SUBFOLDER}/*", ".gitattributes", "*.md"]
    try:
        snapshot_dir_path = snapshot_download(
            repo_id=repo_id,
            cache_dir=cache_location,
            local_dir=active_model_dir, # Download directly into active dir
            local_dir_use_symlinks=False,
            allow_patterns=None, # Download all non-ignored files
            ignore_patterns=root_files_ignore,
            force_download=force,
            resume_download=not force,
            repo_type="model",
        )
        logger.info(f"Root files downloaded/verified in {snapshot_dir_path}")
    except Exception as e:
        raise SetupError(f"Failed to download root files from {repo_id}: {e}") from e

    # 2. Download Specific ONNX Model File
    # --- Construct relative path including ONNX_SUBFOLDER ---
    onnx_model_rel_path = f"{ONNX_SUBFOLDER}/{MODEL_ONNX_BASENAME}{quant_suffix}.onnx"
    logger.info(f"Downloading/Verifying ONNX model file: '{onnx_model_rel_path}'...")
    try:
        source_path_str = download_file_with_retry(
            repo_id=repo_id,
            filename=onnx_model_rel_path,
            cache_dir=cache_location,
            force=force,
            repo_type="model",
        )
        assert source_path_str is not None
        # Copy the downloaded ONNX file to the active ONNX subdir
        target_onnx_path = active_onnx_dir / Path(onnx_model_rel_path).name
        shutil.copyfile(source_path_str, target_onnx_path)
        logger.debug(f"Copied {onnx_model_rel_path} to {target_onnx_path}")
    except EntryNotFoundError:
        logger.error(
            f"Required ONNX file '{onnx_model_rel_path}' not in repo {repo_id}."
        )
        raise SetupError(f"Required ONNX file missing: {onnx_model_rel_path}")
    except SetupError:
        raise
    except Exception as e:
        raise SetupError(f"Error getting ONNX file {onnx_model_rel_path}: {e}") from e

    # 3. Download Corresponding .onnx_data File (if exists)
    # --- Construct relative path including ONNX_SUBFOLDER ---
    onnx_data_rel_path = (
        f"{ONNX_SUBFOLDER}/{MODEL_ONNX_BASENAME}{quant_suffix}.onnx_data"
    )
    logger.info(f"Checking for ONNX data file: '{onnx_data_rel_path}'...")
    try:
        source_path_str = download_file_with_retry(
            repo_id=repo_id,
            filename=onnx_data_rel_path,
            cache_dir=cache_location,
            force=force,
            repo_type="model",
        )
        assert source_path_str is not None
        # Copy the downloaded ONNX data file to the active ONNX subdir
        target_data_path = active_onnx_dir / Path(onnx_data_rel_path).name
        shutil.copyfile(source_path_str, target_data_path)
        logger.debug(f"Copied {onnx_data_rel_path} to {target_data_path}")
        logger.info(f"Found and downloaded ONNX data file: {onnx_data_rel_path}")
    except EntryNotFoundError:
        logger.info(f"ONNX data file '{onnx_data_rel_path}' not found. Skipping.")
    except SetupError:
        raise
    except Exception as e:
        raise SetupError(
            f"Error getting ONNX data file {onnx_data_rel_path}: {e}"
        ) from e

    # Assembly is now done directly after each download via snapshot_download or copyfile
    logger.info(f"Assembly of {active_model_dir} complete.")


# --- Verification Function (Corrected for GenericONNXLLM) ---
def verify_setup(onnx_quant_pref: str = "auto"):
    """Attempts to load all required models to verify CPU setup."""
    logger.info("--- Verifying Model Setup (CPU-Only) ---")
    all_verified = True

    # Verify Embedder (No change needed)
    logger.info(f"Verifying Embedder '{DEFAULT_EMBEDDER_MODEL}' (CPU)...")
    embedder = None
    try:
        embedder = EnhancedEmbedder()
        dim = embedder.get_embedding_dimension()
        if dim and dim > 0 and embedder.model is not None:
            logger.info(f"Embedder OK (CPU, Dim: {dim}).")
        else:
            logger.error(
                f"FAIL: Embedder invalid state (Model:{embedder.model is not None}, Dim:{dim})."
            )
            all_verified = False
    except ModelNotFoundError as e:
        logger.error(f"FAIL: Embedder model not found. {e}")
        all_verified = False
    except Exception as e:
        logger.error(f"FAIL: Error loading CPU embedder: {e}", exc_info=True)
        all_verified = False
    finally:
        if embedder:
            embedder.close()
            del embedder
        gc.collect()

    # Verify Generic ONNX LLM (CPU)
    logger.info("Verifying Generic ONNX LLM (CPU)...")
    llm = None
    try:
        llm = load_onnx_llm(
            onnx_quantization=onnx_quant_pref, preferred_provider="CPUExecutionProvider"
        )
        if llm and isinstance(llm, LLM):  # Check against the LLM protocol
            logger.info(f"ONNX LLM OK ({llm.model_info.model_id} on CPU).")
        else:
            raise RuntimeError("ONNX LLM loader returned None/wrong type.")
    except ModelNotFoundError as e:
        logger.error(f"FAIL: ONNX LLM files missing. {e}")
        all_verified = False
    except Exception as e:
        logger.error(f"FAIL: Error loading ONNX LLM: {e}", exc_info=True)
        all_verified = False
    finally:
        if llm:
            llm.unload()
        if llm is not None:
            del llm
        gc.collect()

    if not all_verified:
        raise SetupError("One or more models failed CPU verification.")
    else:
        logger.info("--- Model Verification Successful (CPU-Only) ---")


# --- Main Setup Function (Corrected for Qwen3) ---
def main():
    parser = argparse.ArgumentParser(
        description="Download/verify models for LlamaSearch (CPU-Only)."
    )
    parser.add_argument(
        "--force", action="store_true", help="Force redownload/reassembly"
    )
    parser.add_argument(
        "--onnx-quant",
        type=str,
        default="auto",
        choices=[
            "auto",
            "fp32",
            "fp16",
            "int8",
            "quantized",
            "q4",
            "q4f16",
            "uint8",
            "bnb4",
        ],
        help="Specify ONNX quantization for the LLM (default: auto)",
    )
    args = parser.parse_args()
    logger.info("--- Starting LlamaSearch Model Setup (CPU-Only) ---")
    if args.force:
        logger.info("Force mode enabled: Active directory will be recreated.")

    try:
        logger.info("Setup configured for CPU-only operation.")
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
                logger.warning("HF token not found. Downloads might fail.")
        except Exception:
            logger.warning("Could not check HF token.")

        # Download/Verify components
        check_or_download_embedder(models_dir, args.force)
        # --- Call the generic ONNX LLM downloader ---
        check_or_download_onnx_llm(models_dir, args.onnx_quant, args.force)

        # Final Verification (CPU)
        verify_setup(args.onnx_quant)
        logger.info("--- LlamaSearch Model Setup Completed Successfully (CPU-Only) ---")
        sys.exit(0)

    except SetupError as e:
        logger.error(f"Setup failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during setup: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()