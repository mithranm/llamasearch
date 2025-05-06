#!/usr/bin/env python3
"""
setup.py - Downloads and verifies LlamaSearch models (CPU-Only).

Configured for Teapot LLM and mixedbread-ai/mxbai-embed-large-v1 embedder.
Downloads only the necessary PyTorch files for the embedder, ignoring other runtimes.
"""

import argparse
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List  # Added List
import gc

from huggingface_hub import (
    hf_hub_download,
    snapshot_download,
)  # Import snapshot_download directly
from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError
from huggingface_hub.utils._hf_folder import HfFolder

# Use updated default model name
from llamasearch.core.embedder import DEFAULT_MODEL_NAME as DEFAULT_EMBEDDER_MODEL

# Use updated embedder
from llamasearch.core.embedder import EnhancedEmbedder
from llamasearch.core.teapot import TEAPOT_BASE_FILES
from llamasearch.core.teapot import (
    ONNX_SUBFOLDER,
    REQUIRED_ONNX_BASENAMES,
    TEAPOT_REPO_ID,
    _determine_onnx_provider,
    _select_onnx_quantization,
    load_teapot_onnx_llm,
)
from llamasearch.data_manager import data_manager
from llamasearch.exceptions import ModelNotFoundError, SetupError
from llamasearch.utils import setup_logging

logger = setup_logging("llamasearch.setup")


# Helper: Download with Retries (No changes needed here)
def download_file_with_retry(
    repo_id: str,
    filename: str,
    cache_dir: Path,
    force: bool,
    max_retries: int = 2,
    delay: int = 5,
    **kwargs,
):
    """Attempts to download a file with retries on failure."""
    assert isinstance(cache_dir, Path), f"cache_dir must be Path, got {type(cache_dir)}"
    for attempt in range(max_retries + 1):
        try:
            logger.debug(
                f"Attempt {attempt + 1} downloading: {filename} from {repo_id}"
            )
            file_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=cache_dir,
                force_download=force,
                resume_download=True,
                local_files_only=False,
                local_dir_use_symlinks=False,
                **kwargs,
            )
            assert file_path is not None, f"Download returned None for {filename}"
            fpath = Path(file_path)
            if not fpath.exists() or fpath.stat().st_size < 10:
                raise FileNotFoundError(
                    f"File {filename} invalid after DL attempt {attempt + 1}."
                )
            logger.debug(f"Successfully downloaded {filename} to {file_path}")
            return file_path
        except (ConnectionError, TimeoutError, FileNotFoundError) as e:
            logger.warning(f"Download attempt {attempt + 1} for {filename} failed: {e}")
            if attempt < max_retries:
                logger.info(f"Retrying download of {filename} in {delay}s...")
                time.sleep(delay)
            else:
                logger.error(f"Max retries reached for {filename}. Download failed.")
                raise SetupError(f"Failed download after retries: {filename}") from e
        except Exception as e:
            logger.error(
                f"Unexpected error downloading {filename} attempt {attempt + 1}: {e}",
                exc_info=True,
            )
            raise SetupError(
                f"Failed download due to unexpected error: {filename}"
            ) from e
    # Should not be reachable
    raise SetupError(f"Download logic error for {filename}.")


# --- Updated Embedder Check/Download ---
def check_or_download_embedder(models_dir: Path, force: bool = False) -> None:
    """
    Checks or downloads the default embedder model (mxbai), ignoring
    files related to ONNX, GGUF, and OpenVINO runtimes.
    """
    model_name = DEFAULT_EMBEDDER_MODEL  # mxbai-embed-large-v1
    logger.info(f"Checking/Downloading Embedder Model: {model_name}")

    # Define patterns to ignore alternative runtimes
    # Based on common conventions and mxbai repo structure
    ignore_patterns: List[str] = [
        "*.onnx",  # Ignore all ONNX model files
        "onnx/*",  # Ignore files within an 'onnx' subdirectory (if any)
        "*.gguf",  # Ignore GGUF model files
        "gguf/*",  # Ignore files within a 'gguf' subdirectory (if any)
        "openvino/*",  # Ignore files within an 'openvino' subdirectory (if any)
        # Add other specific files/patterns if necessary
        # "config_onnx.json",
    ]
    logger.info(f"Ignoring patterns for embedder download: {ignore_patterns}")

    try:
        # Use snapshot_download directly as before
        # Try local check first, applying ignore patterns even for local check
        # This ensures we don't consider a cache valid if it only has ignored files
        if not force:
            try:
                snapshot_download(
                    repo_id=model_name,
                    cache_dir=models_dir,
                    local_files_only=True,
                    local_dir_use_symlinks=False,
                    ignore_patterns=ignore_patterns,  # Apply ignore patterns to local check
                )
                logger.info(
                    f"Embedder model '{model_name}' (PyTorch files) found locally."
                )
                return  # Found locally, exit
            except (EntryNotFoundError, LocalEntryNotFoundError, FileNotFoundError):
                logger.info(
                    f"Embedder model '{model_name}' (PyTorch files) not found locally or incomplete. Attempting download..."
                )

        # Download if needed or forced, applying ignore patterns
        snapshot_download(
            repo_id=model_name,
            cache_dir=models_dir,
            force_download=force,
            resume_download=True,
            local_files_only=False,
            local_dir_use_symlinks=False,
            ignore_patterns=ignore_patterns,  # Apply ignore patterns during download
        )
        logger.info(
            f"Embedder model '{model_name}' (PyTorch files) cache verified/downloaded in {models_dir}."
        )
    except Exception as download_err:
        logger.error(
            f"Failed to download/verify embedder model {model_name}: {download_err}",
            exc_info=True,
        )
        raise SetupError(f"Failed get embedder model {model_name}") from download_err


# --- Teapot ONNX Check/Download (Remains the same) ---
def check_or_download_teapot_onnx(
    models_dir: Path, quant_pref: str = "auto", force: bool = False
) -> None:
    """Downloads required Teapot files and assembles the 'active_teapot' dir."""
    logger.info(f"Checking/Downloading Teapot ONNX Files (Quantization: {quant_pref})")
    # Always use CPUExecutionProvider for ONNX, no hardware detection needed
    provider_name, _ = _determine_onnx_provider("CPUExecutionProvider")
    quant_suffix = _select_onnx_quantization(quant_pref)
    logger.info(f"Targeting ONNX quantization suffix: '{quant_suffix}' for CPU")

    active_model_dir = models_dir / "active_teapot"
    active_onnx_dir = active_model_dir / ONNX_SUBFOLDER
    needs_clean = force
    if not needs_clean and active_model_dir.exists():
        if not active_onnx_dir.is_dir():
            needs_clean = True
            logger.warning(
                "Active dir exists but ONNX subfolder missing. Forcing clean."
            )
        else:
            for basename in REQUIRED_ONNX_BASENAMES:
                target_onnx = active_onnx_dir / f"{basename}{quant_suffix}.onnx"
                if not target_onnx.exists():
                    needs_clean = True
                    logger.warning(
                        f"Target ONNX file '{target_onnx.name}' missing. Forcing clean."
                    )
                    break
            if not needs_clean:
                for base_file in TEAPOT_BASE_FILES:
                    target_base = active_model_dir / base_file
                    if not target_base.exists():
                        needs_clean = True
                        logger.warning(
                            f"Base file '{base_file}' missing. Forcing clean."
                        )
                        break

    if needs_clean and active_model_dir.exists():
        logger.info(f"Cleaning existing active model directory: {active_model_dir}")
        try:
            shutil.rmtree(active_model_dir)
        except OSError as e:
            raise SetupError(f"Failed clear active dir {active_model_dir}: {e}") from e

    active_onnx_dir.mkdir(parents=True, exist_ok=True)
    cache_location = models_dir
    files_to_copy_or_link: Dict[Path, Path] = {}

    # 1. Download Base Files
    logger.info("Downloading/Verifying base Teapot files...")
    for base_file in TEAPOT_BASE_FILES:
        try:
            source_path_str = download_file_with_retry(
                repo_id=TEAPOT_REPO_ID,
                filename=base_file,
                cache_dir=cache_location,
                force=force,
                repo_type="model",
            )
            assert source_path_str is not None
            files_to_copy_or_link[active_model_dir / base_file] = Path(source_path_str)
        except SetupError:
            raise
        except Exception as e:
            raise SetupError(f"Error getting base file {base_file}: {e}") from e

    # 2. Download Specific ONNX Files
    logger.info(f"Downloading/Verifying ONNX files for suffix '{quant_suffix}'...")
    onnx_files_to_download = [
        f"{ONNX_SUBFOLDER}/{basename}{quant_suffix}.onnx"
        for basename in REQUIRED_ONNX_BASENAMES
    ]
    for onnx_file_rel_path in onnx_files_to_download:
        try:
            source_path_str = download_file_with_retry(
                repo_id=TEAPOT_REPO_ID,
                filename=onnx_file_rel_path,
                cache_dir=cache_location,
                force=force,
                repo_type="model",
            )
            assert source_path_str is not None
            target_path = active_onnx_dir / Path(onnx_file_rel_path).name
            files_to_copy_or_link[target_path] = Path(source_path_str)
        except EntryNotFoundError:
            logger.error(
                f"ONNX file '{onnx_file_rel_path}' not found. Is quant '{quant_suffix}' valid?"
            )
            raise SetupError(f"Required ONNX file missing: {onnx_file_rel_path}")
        except SetupError:
            raise
        except Exception as e:
            raise SetupError(
                f"Error getting ONNX file {onnx_file_rel_path}: {e}"
            ) from e

    # 3. Assemble active_model_dir by copying
    logger.info(f"Assembling active model directory: {active_model_dir}")
    copied_count = 0
    for target, source in files_to_copy_or_link.items():
        if not source.exists():
            raise SetupError(f"Source file missing before copy: {source}")
        try:
            logger.debug(f"Copying {source.name} -> {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied_count += 1
        except Exception as e:
            raise SetupError(f"Failed copy {source} -> {target}: {e}") from e

    logger.info(f"Successfully assembled {copied_count} files into {active_model_dir}.")


# --- Verification Function (CPU-Only) ---
def verify_setup(onnx_quant_pref: str = "auto"):
    """Attempts to load all required models to verify CPU setup."""
    logger.info("--- Verifying Model Setup (CPU-Only) ---")
    all_verified = True

    # Verify Embedder (mxbai on CPU)
    logger.info(f"Verifying Embedder model '{DEFAULT_EMBEDDER_MODEL}' (CPU)...")
    embedder = None
    try:
        embedder = EnhancedEmbedder()  # Uses mxbai, CPU-only settings
        logger.info(f"Embedder initialized for device: {embedder.config.device}")
        dim = embedder.get_embedding_dimension()
        if dim and dim > 0 and embedder.model is not None:
            logger.info(
                f"Embedder model loaded successfully on CPU (Effective Dim: {dim})."
            )
        else:
            if embedder.model is None:
                logger.error(
                    "Verification Failed: Embedder failed to load model on CPU."
                )
            else:
                logger.error(
                    f"Verification Failed: Embedder loaded but invalid dimension ({dim})."
                )
            all_verified = False
    except ModelNotFoundError as e:
        logger.error(f"Verification Failed: Embedder model not found. {e}")
        all_verified = False
    except Exception as e:
        logger.error(
            f"Verification Failed: Error loading CPU embedder model: {e}", exc_info=True
        )
        all_verified = False
    finally:
        if embedder and hasattr(embedder, "close"):
            embedder.close()
            del embedder
        gc.collect()

    # Verify Teapot ONNX LLM (CPU)
    logger.info("Verifying Teapot ONNX LLM (CPU)...")
    llm = None
    try:
        llm = load_teapot_onnx_llm(
            onnx_quantization=onnx_quant_pref,
            preferred_provider="CPUExecutionProvider",  # Force CPU check
        )
        if llm:
            logger.info(
                f"Teapot ONNX LLM ({llm.model_info.model_id}) loaded successfully on CPU."
            )
        else:
            raise RuntimeError("Teapot loader returned None.")
    except ModelNotFoundError as e:
        logger.error(f"Verification Failed: Teapot model/files missing. {e}")
        all_verified = False
    except Exception as e:
        logger.error(
            f"Verification Failed: Error loading Teapot ONNX LLM: {e}", exc_info=True
        )
        all_verified = False
    finally:
        if llm and hasattr(llm, "unload"):
            llm.unload()
        if llm is not None:
            del llm
        gc.collect()

    if not all_verified:
        logger.error("--- Model Verification Failed ---")
        raise SetupError("One or more models failed CPU verification.")
    else:
        logger.info("--- Model Verification Successful (CPU-Only) ---")


# --- Main Setup Function (CPU-Only) ---
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
        choices=["auto", "fp32", "fp16", "int8", "q4", "q4f16", "bnb4", "uint8"],
        help="Specify ONNX quantization for Teapot LLM (default: auto)",
    )
    args = parser.parse_args()
    logger.info("--- Starting LlamaSearch Model Setup (CPU-Only) ---")
    if args.force:
        logger.info("Force mode enabled: Active directory will be recreated.")

    try:
        logger.info("Setup configured for CPU-only operation.")

        # Get models dir
        models_dir_str = data_manager.get_data_paths().get("models")
        if not models_dir_str:
            raise SetupError("Models directory path not configured.")
        models_dir = Path(models_dir_str)
        models_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Using models directory: {models_dir}")

        # Check HF token
        try:
            if HfFolder.get_token():
                logger.info("Hugging Face token found.")
            else:
                logger.warning(
                    "Hugging Face token not found. Downloads might be slower or fail."
                )
        except Exception:
            logger.warning("Could not check for Hugging Face token.")

        # Download/Verify components
        check_or_download_embedder(
            models_dir, args.force
        )  # Downloads mxbai (PyTorch only)
        check_or_download_teapot_onnx(models_dir, args.onnx_quant, args.force)

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
