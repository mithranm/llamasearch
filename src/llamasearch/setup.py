#!/usr/bin/env python3
"""
setup.py - Command-line utility for downloading and verifying LlamaSearch models.

Downloads only the necessary configuration, tokenizer, and selected ONNX
quantization files using hf_hub_download and assembles them into an
'active_teapot' directory for loading. Ensures the active directory is clean.
"""

import argparse
import sys
import spacy
import subprocess
from pathlib import Path
import time
import shutil
from huggingface_hub import hf_hub_download
from huggingface_hub.utils._hf_folder import HfFolder
from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError

from llamasearch.data_manager import data_manager
from llamasearch.utils import setup_logging
from llamasearch.exceptions import SetupError, ModelNotFoundError
from llamasearch.core.embedder import (
    DEFAULT_MODEL_NAME as DEFAULT_EMBEDDER_MODEL,
    EnhancedEmbedder,
)
from llamasearch.core.teapot import (
    TEAPOT_REPO_ID,
    ONNX_SUBFOLDER,
    REQUIRED_ONNX_BASENAMES,
    load_teapot_onnx_llm,
    _select_onnx_quantization,
    _determine_onnx_provider,
    TEAPOT_BASE_FILES,  # Import the constant
)
from llamasearch.core.bm25 import load_nlp_model
from llamasearch.hardware import detect_hardware_info
from typing import Optional

logger = setup_logging("llamasearch.setup")


# --- Helper: Download with Retries ---
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
    assert isinstance(cache_dir, Path), (
        f"cache_dir must be a Path object, got {type(cache_dir)}"
    )
    for attempt in range(max_retries + 1):
        try:
            logger.debug(f"Attempt {attempt + 1} downloading: {filename}")
            file_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=cache_dir,
                force_download=force,
                resume_download=True,
                local_files_only=False,
                **kwargs,
            )
            fpath = Path(file_path)
            if not fpath.exists() or fpath.stat().st_size < 10:
                raise FileNotFoundError(
                    f"File {filename} missing/empty DL attempt {attempt + 1}."
                )
            logger.debug(f"Successfully downloaded {filename} to {file_path}")
            return file_path  # Success
        except (ConnectionError, TimeoutError, FileNotFoundError) as e:
            logger.warning(f"Download attempt {attempt + 1} for {filename} failed: {e}")
            if attempt < max_retries:
                logger.info(f"Retrying download of {filename} in {delay} seconds...")
                time.sleep(delay)
            else:
                logger.error(f"Max retries reached for {filename}. Download failed.")
                raise SetupError(f"Failed download: {filename}") from e
        except Exception as e:
            logger.error(
                f"Unexpected error DL {filename} attempt {attempt + 1}: {e}",
                exc_info=True,
            )
            raise SetupError(f"Failed DL {filename}") from e
    # Added fallback return to satisfy static analysis, though it shouldn't be reached due to exceptions
    raise SetupError(f"Download failed for {filename} after retries.")


# --- Model Check/Download Functions (Embedder and Spacy remain the same) ---
def check_or_download_embedder(models_dir: Path, force: bool = False) -> None:
    model_name = DEFAULT_EMBEDDER_MODEL
    logger.info(f"Checking/Downloading Embedder Model: {model_name}")
    try:
        from huggingface_hub import snapshot_download as embedder_snapshot_download

        embedder_snapshot_download(
            repo_id=model_name,
            cache_dir=models_dir,
            force_download=force,
            resume_download=True,
            local_files_only=not force,
            local_dir_use_symlinks=False,
        )
        logger.info(
            f"Embedder model '{model_name}' cache verified/downloaded in {models_dir}."
        )
    except (EntryNotFoundError, LocalEntryNotFoundError):
        if not force:
            logger.info(
                f"Embedder model '{model_name}' not found locally. Attempting download..."
            )
            try:
                from huggingface_hub import (
                    snapshot_download as embedder_snapshot_download_retry,
                )

                embedder_snapshot_download_retry(
                    repo_id=model_name,
                    cache_dir=models_dir,
                    force_download=False,
                    resume_download=True,
                    local_files_only=False,
                    local_dir_use_symlinks=False,
                )
                logger.info(f"Embedder model '{model_name}' downloaded successfully.")
            except Exception as download_err:
                raise SetupError(
                    f"Failed to download embedder model {model_name}"
                ) from download_err
        else:
            raise SetupError(f"Failed to get embedder model {model_name} with --force")
    except Exception as e:
        raise SetupError(f"Unexpected error getting embedder model {model_name}") from e


def check_or_download_spacy(force: bool = False) -> None:
    models = ["en_core_web_trf", "en_core_web_sm"]
    all_ok = True
    for model_name in models:
        logger.info(f"Checking/Downloading SpaCy Model: {model_name}")
        try:
            is_installed = spacy.util.is_package(model_name)
            if is_installed and not force:
                logger.info(f"SpaCy model '{model_name}' already installed.")
                continue
            if is_installed and force:
                logger.info(f"Force re-downloading SpaCy model '{model_name}'...")
            py_exec = sys.executable
            cmd = [py_exec, "-m", "spacy", "download", model_name]
            logger.debug(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=False, timeout=300
            )
            if result.returncode == 0:
                logger.info(
                    f"Successfully downloaded/verified SpaCy model '{model_name}'."
                )
            else:
                logger.error(
                    f"Failed DL SpaCy model '{model_name}'. RC: {result.returncode}\nStderr:\n{result.stderr}"
                )
                all_ok = False
        except subprocess.TimeoutExpired:
            logger.error(f"Timeout DL SpaCy model '{model_name}'.")
            all_ok = False
        except Exception as e:
            logger.error(f"Error DL SpaCy model '{model_name}': {e}", exc_info=True)
            all_ok = False
    if not all_ok:
        raise SetupError("One or more SpaCy models failed to download.")


# --- Modified Teapot ONNX Download & Assembly ---
def check_or_download_teapot_onnx(
    models_dir: Path, quant_pref: str = "auto", force: bool = False
) -> None:
    """Downloads required Teapot files and assembles the 'active_teapot' dir."""
    logger.info(f"Checking/Downloading Teapot ONNX Files (Quantization: {quant_pref})")
    hw_info = detect_hardware_info()
    provider_name, _ = _determine_onnx_provider()
    quant_suffix = _select_onnx_quantization(hw_info, provider_name, None, quant_pref)
    logger.info(f"Targeting ONNX quantization suffix: '{quant_suffix}'")

    active_model_dir = models_dir / "active_teapot"
    active_onnx_dir = active_model_dir / ONNX_SUBFOLDER

    logger.info(f"Ensuring clean active model directory: {active_model_dir}")
    if active_model_dir.exists():
        logger.debug("Removing existing active directory...")
        try:
            shutil.rmtree(active_model_dir)
        except OSError as e:
            logger.error(
                f"Failed to remove existing active directory: {e}", exc_info=True
            )
            raise SetupError(
                f"Failed to clear active directory {active_model_dir}. Please remove it manually and retry."
            ) from e
    active_onnx_dir.mkdir(parents=True, exist_ok=True)

    cache_location = models_dir
    assert isinstance(cache_location, Path), (
        f"cache_location must be a Path object, got {type(cache_location)}"
    )

    files_to_copy_or_link = {}  # Store {target_path: source_path}

    # 1. Download Base Files into cache_location
    logger.info("Downloading/Verifying base Teapot files...")
    for base_file in TEAPOT_BASE_FILES:
        try:
            source_path_str: Optional[str] = None  # Explicitly type hint
            if not force:
                try:
                    source_path_str = hf_hub_download(
                        repo_id=TEAPOT_REPO_ID,
                        filename=base_file,
                        cache_dir=cache_location,
                        local_files_only=True,
                    )
                    logger.debug(f"Base file '{base_file}' found locally.")
                except (LocalEntryNotFoundError, FileNotFoundError):
                    logger.debug(
                        f"Base file '{base_file}' not found locally or symlink broken, proceeding to download."
                    )
                    source_path_str = None

            if source_path_str is None:
                source_path_str = download_file_with_retry(
                    repo_id=TEAPOT_REPO_ID,
                    filename=base_file,
                    cache_dir=cache_location,
                    force=force,
                    repo_type="model",
                )

            # --- Assertion added ---
            assert source_path_str is not None, (
                f"source_path_str should not be None after download for {base_file}"
            )
            target_path = active_model_dir / base_file
            files_to_copy_or_link[target_path] = Path(source_path_str)
        except Exception:
            raise  # Error handled within download_file_with_retry which raises SetupError

    # 2. Download Specific ONNX Files into cache_location
    logger.info(
        f"Downloading/Verifying specific ONNX files for suffix '{quant_suffix}'..."
    )
    onnx_files_to_download = [
        f"{ONNX_SUBFOLDER}/{basename}{quant_suffix}.onnx"
        for basename in REQUIRED_ONNX_BASENAMES
    ]
    for onnx_file_rel_path in onnx_files_to_download:
        try:
            source_path_str: Optional[str] = None  # Explicitly type hint
            if not force:
                try:
                    source_path_str = hf_hub_download(
                        repo_id=TEAPOT_REPO_ID,
                        filename=onnx_file_rel_path,
                        cache_dir=cache_location,
                        local_files_only=True,
                    )
                    logger.debug(f"ONNX file '{onnx_file_rel_path}' found locally.")
                except (LocalEntryNotFoundError, FileNotFoundError):
                    logger.debug(
                        f"ONNX file '{onnx_file_rel_path}' not found locally or symlink broken, proceeding to download."
                    )
                    source_path_str = None

            if source_path_str is None:
                source_path_str = download_file_with_retry(
                    repo_id=TEAPOT_REPO_ID,
                    filename=onnx_file_rel_path,
                    cache_dir=cache_location,
                    force=force,
                    repo_type="model",
                )

            # --- Assertion added ---
            assert source_path_str is not None, (
                f"source_path_str should not be None after download for {onnx_file_rel_path}"
            )
            target_path = active_onnx_dir / Path(onnx_file_rel_path).name
            files_to_copy_or_link[target_path] = Path(source_path_str)
        except EntryNotFoundError:
            logger.error(
                f"Required ONNX file '{onnx_file_rel_path}' not found in repo {TEAPOT_REPO_ID}. Is quant '{quant_suffix}' valid?"
            )
            raise SetupError(
                f"Required ONNX file missing from repository: {onnx_file_rel_path}"
            )
        except Exception:
            raise  # Error handled within download_file_with_retry

    # 3. Assemble the active_model_dir by copying files
    logger.info(f"Assembling active model directory: {active_model_dir}")
    copied_count = 0
    for target, source in files_to_copy_or_link.items():
        if not source.exists():
            logger.error(f"Source file does not exist, cannot copy: {source}")
            raise SetupError(f"Downloaded file missing before copy: {source}")
        try:
            logger.debug(f"Copying {source.name} -> {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied_count += 1
        except Exception as e:
            logger.error(f"Failed to copy {source} to {target}: {e}", exc_info=True)
            raise SetupError(f"Failed to assemble active model directory at {target}")

    logger.info(
        f"Successfully downloaded and assembled {copied_count} required files into {active_model_dir}."
    )


# --- Verification Function ---
def verify_setup(onnx_quant_pref: str = "auto"):
    """Attempts to load all required models to verify setup."""
    logger.info("--- Verifying Model Setup ---")
    all_verified = True
    # Verify Embedder
    logger.info("Verifying Embedder model...")
    try:
        embedder = EnhancedEmbedder(auto_optimize=False)
        embedder.close()
        logger.info("Embedder model loaded successfully.")
    except ModelNotFoundError as e:
        logger.error(f"Verification Failed: Embedder model not found. {e}")
        all_verified = False
    except Exception as e:
        logger.error(
            f"Verification Failed: Error loading embedder model: {e}", exc_info=True
        )
        all_verified = False
    # Verify SpaCy
    logger.info("Verifying SpaCy models...")
    try:
        nlp = load_nlp_model()
        del nlp
        logger.info("SpaCy models ('trf' or 'sm') loaded successfully.")
    except ModelNotFoundError as e:
        logger.error(f"Verification Failed: SpaCy model not found. {e}")
        all_verified = False
    except Exception as e:
        logger.error(
            f"Verification Failed: Error loading SpaCy model: {e}", exc_info=True
        )
        all_verified = False
    # Verify Teapot ONNX LLM
    logger.info("Verifying Teapot ONNX LLM...")
    try:
        hw_info = detect_hardware_info()
        provider_name, _ = _determine_onnx_provider()
        expected_quant_suffix = _select_onnx_quantization(
            hw_info, provider_name, None, onnx_quant_pref
        )
        logger.info(
            f"(Verification targets quantization suffix: '{expected_quant_suffix}')"
        )
        llm = load_teapot_onnx_llm(
            onnx_quantization=onnx_quant_pref, preferred_provider="CPUExecutionProvider"
        )
        if llm:
            llm.unload()
            logger.info("Teapot ONNX LLM loaded successfully from active directory.")
        else:
            logger.error("Verification Failed: Teapot ONNX LLM loader returned None.")
            all_verified = False
    except ModelNotFoundError as e:
        logger.error(
            f"Verification Failed: Teapot ONNX model/files not found/invalid in active directory. {e}"
        )
        all_verified = False
    except Exception as e:
        logger.error(
            f"Verification Failed: Error loading Teapot ONNX LLM: {e}", exc_info=True
        )
        all_verified = False
    if not all_verified:
        logger.error("--- Model Verification Failed ---")
        raise SetupError("One or more models failed verification.")
    else:
        logger.info("--- Model Verification Successful ---")


# --- Main Setup Function ---
def main():
    parser = argparse.ArgumentParser(
        description="Download/verify models for LlamaSearch."
    )
    parser.add_argument(
        "--force", action="store_true", help="Force redownload/reassembly"
    )
    parser.add_argument(
        "--onnx-quant",
        type=str,
        default="auto",
        choices=["auto", "fp32", "fp16", "int8", "q4", "q4f16", "bnb4", "uint8"],
        help="Specify ONNX quantization (default: auto)",
    )
    args = parser.parse_args()
    logger.info("--- Starting LlamaSearch Model Setup ---")
    if args.force:
        logger.info("Force mode enabled: Active directory will be recreated.")
    try:
        models_dir_str = data_manager.get_data_paths().get("models")
        if not models_dir_str:
            raise SetupError(
                "Models directory path not configured or found in settings."
            )
        models_dir = Path(models_dir_str)
        models_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Using models directory: {models_dir}")
        try:
            if HfFolder.get_token():
                logger.info("HF token found.")
            else:
                logger.warning("HF token not found.")
        except Exception:
            logger.warning("Could not check HF token.")
        # Download components
        check_or_download_embedder(models_dir, args.force)
        check_or_download_spacy(args.force)
        # Download and assemble Teapot files into 'active_teapot'
        check_or_download_teapot_onnx(models_dir, args.onnx_quant, args.force)
        # Verification Step (will check 'active_teapot')
        verify_setup(args.onnx_quant)
        logger.info("--- LlamaSearch Model Setup Completed Successfully ---")
        sys.exit(0)
    except SetupError as e:
        logger.error(f"Setup failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during setup: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
