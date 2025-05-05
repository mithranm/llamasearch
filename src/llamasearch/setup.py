#!/usr/bin/env python3
"""
setup.py - Command-line utility for downloading and verifying LlamaSearch models.

Downloads only the necessary configuration, tokenizer, and selected ONNX
quantization files using hf_hub_download and assembles them into an
'active_teapot' directory for loading. Ensures the active directory is clean.
"""

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

# <<< FIX: Add Dict import >>>
from typing import Dict

import spacy
import torch  # Import torch for device check
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError
from huggingface_hub.utils._hf_folder import HfFolder

from llamasearch.core.bm25 import load_nlp_model
from llamasearch.core.embedder import DEFAULT_MODEL_NAME as DEFAULT_EMBEDDER_MODEL
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
from llamasearch.hardware import detect_hardware_info
from llamasearch.utils import setup_logging

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
    assert isinstance(cache_dir, Path), f"cache_dir must be Path, got {type(cache_dir)}"
    for attempt in range(max_retries + 1):
        try:
            logger.debug(f"Attempt {attempt + 1} downloading: {filename}")
            # Use local_dir_use_symlinks=False for better Windows compatibility
            file_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=cache_dir,
                force_download=force,
                resume_download=True,
                local_files_only=False,
                local_dir_use_symlinks=False,  # Added for Windows
                **kwargs,
            )
            fpath = Path(file_path)
            # Check size > 10 bytes as a basic validity check
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
                logger.error(f"Max retries for {filename}. Download failed.")
                raise SetupError(f"Failed download: {filename}") from e
        except Exception as e:
            logger.error(
                f"Unexpected error DL {filename} attempt {attempt + 1}: {e}",
                exc_info=True,
            )
            raise SetupError(f"Failed DL {filename}") from e
    # Should be unreachable due to exceptions
    raise SetupError(f"Download failed for {filename} after retries.")


# --- Model Check/Download Functions ---
def check_or_download_embedder(models_dir: Path, force: bool = False) -> None:
    model_name = DEFAULT_EMBEDDER_MODEL
    logger.info(f"Checking/Downloading Embedder Model: {model_name}")
    try:
        from huggingface_hub import snapshot_download as embedder_snapshot_download

        # First, try local check only
        if not force:
            try:
                embedder_snapshot_download(
                    repo_id=model_name,
                    cache_dir=models_dir,
                    local_files_only=True,
                    local_dir_use_symlinks=False,  # Avoid symlink issues on Windows
                )
                logger.info(f"Embedder model '{model_name}' found locally.")
                return  # Found locally, no need to download
            except (EntryNotFoundError, LocalEntryNotFoundError, FileNotFoundError):
                logger.info(
                    f"Embedder model '{model_name}' not found locally. Attempting download..."
                )
        # Proceed with download if local check failed or force=True
        embedder_snapshot_download(
            repo_id=model_name,
            cache_dir=models_dir,
            force_download=force,
            resume_download=True,
            local_files_only=False,  # Ensure download happens
            local_dir_use_symlinks=False,
        )
        logger.info(
            f"Embedder model '{model_name}' cache verified/downloaded in {models_dir}."
        )
    except Exception as download_err:
        logger.error(
            f"Failed to download/verify embedder model {model_name}: {download_err}",
            exc_info=True,
        )
        raise SetupError(f"Failed get embedder model {model_name}") from download_err


def check_or_download_spacy(force: bool = False) -> None:
    models = ["en_core_web_trf", "en_core_web_sm"]
    all_ok = True
    for model_name in models:
        logger.info(f"Checking/Downloading SpaCy Model: {model_name}")
        try:
            is_installed = spacy.util.is_package(model_name)
            if is_installed and not force:
                logger.info(f"SpaCy '{model_name}' already installed.")
                continue
            if is_installed and force:
                logger.info(f"Force re-downloading SpaCy '{model_name}'...")
            py_exec = sys.executable or "python"  # Fallback to 'python'
            cmd = [py_exec, "-m", "spacy", "download", model_name]
            logger.debug(f"Running: {' '.join(cmd)}")
            # Add creationflags for Windows
            creationflags = (
                getattr(subprocess, "CREATE_NO_WINDOW", 0)
                if sys.platform == "win32"
                else 0
            )
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=300,
                creationflags=creationflags,
            )
            if result.returncode == 0:
                logger.info(f"Successfully downloaded/verified SpaCy '{model_name}'.")
            else:
                logger.error(
                    f"Failed DL SpaCy '{model_name}'. RC:{result.returncode}\nStderr:\n{result.stderr}"
                )
                all_ok = False
        except subprocess.TimeoutExpired:
            logger.error(f"Timeout DL SpaCy '{model_name}'.")
            all_ok = False
        except Exception as e:
            logger.error(f"Error DL SpaCy '{model_name}': {e}", exc_info=True)
            all_ok = False
    if not all_ok:
        raise SetupError("One or more SpaCy models failed to download.")


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

    # Clean active directory if forcing or if it seems incomplete/invalid
    needs_clean = force
    if not needs_clean and active_model_dir.exists():
        if not active_onnx_dir.is_dir():
            logger.warning(
                "Active directory exists but ONNX subfolder missing. Forcing clean."
            )
            needs_clean = True
        else:
            # Check if required files for the *targeted* quant suffix exist
            for basename in REQUIRED_ONNX_BASENAMES:
                target_onnx = active_onnx_dir / f"{basename}{quant_suffix}.onnx"
                if not target_onnx.exists():
                    logger.warning(
                        f"Target ONNX file '{target_onnx.name}' missing in active dir. Forcing clean."
                    )
                    needs_clean = True
                    break
            if not needs_clean:  # Check base files too
                for base_file in TEAPOT_BASE_FILES:
                    target_base = active_model_dir / base_file
                    if not target_base.exists():
                        logger.warning(
                            f"Base file '{base_file}' missing in active dir. Forcing clean."
                        )
                        needs_clean = True
                        break

    if needs_clean and active_model_dir.exists():
        logger.info(f"Cleaning existing active model directory: {active_model_dir}")
        try:
            shutil.rmtree(active_model_dir)
        except OSError as e:
            raise SetupError(f"Failed clear active dir {active_model_dir}: {e}") from e
    active_onnx_dir.mkdir(parents=True, exist_ok=True)

    cache_location = models_dir  # Download directly into models_dir cache
    # <<< FIX: Use correct Dict type hint >>>
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
            assert source_path_str is not None, (
                f"Download returned None for {base_file}"
            )
            target_path = active_model_dir / base_file
            files_to_copy_or_link[target_path] = Path(source_path_str)
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
            assert source_path_str is not None, (
                f"Download returned None for {onnx_file_rel_path}"
            )
            target_path = active_onnx_dir / Path(onnx_file_rel_path).name
            files_to_copy_or_link[target_path] = Path(source_path_str)
        except EntryNotFoundError:
            logger.error(
                f"ONNX file '{onnx_file_rel_path}' not found in repo {TEAPOT_REPO_ID}. Is quant '{quant_suffix}' valid?"
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
            shutil.copy2(source, target)  # copy2 preserves metadata
            copied_count += 1
        except Exception as e:
            raise SetupError(f"Failed copy {source} -> {target}: {e}") from e

    logger.info(
        f"Successfully downloaded/assembled {copied_count} files into {active_model_dir}."
    )


# --- Verification Function ---
def verify_setup(onnx_quant_pref: str = "auto"):
    """Attempts to load all required models to verify setup."""
    logger.info("--- Verifying Model Setup ---")
    all_verified = True

    # Verify Embedder
    logger.info("Verifying Embedder model...")
    embedder = None
    try:
        embedder = EnhancedEmbedder()
        dim = embedder.get_embedding_dimension()
        if dim and dim > 0:
            logger.info(f"Embedder model loaded successfully (Dim: {dim}).")
        else:
            raise ValueError("Embedder loaded but returned invalid dimension.")
    except ModelNotFoundError as e:
        logger.error(f"Verification Failed: Embedder model not found. {e}")
        all_verified = False
    except Exception as e:
        logger.error(
            f"Verification Failed: Error loading embedder model: {e}", exc_info=True
        )
        all_verified = False
    finally:
        if embedder and hasattr(embedder, "close"):
            embedder.close()

    # Verify SpaCy
    logger.info("Verifying SpaCy models...")
    nlp = None  # Define outside try
    try:
        nlp = load_nlp_model()
        logger.info("SpaCy models ('trf' or 'sm') loaded successfully.")
    except ModelNotFoundError as e:
        logger.error(f"Verification Failed: SpaCy model not found. {e}")
        all_verified = False
    except Exception as e:
        logger.error(
            f"Verification Failed: Error loading SpaCy model: {e}", exc_info=True
        )
        all_verified = False
    finally:
        if nlp is not None:
            del nlp  # Release nlp object

    # Verify Teapot ONNX LLM
    logger.info("Verifying Teapot ONNX LLM...")
    llm = None
    try:
        hw_info = detect_hardware_info()
        provider_name, provider_opts = _determine_onnx_provider()
        # Force CPU provider for verification to avoid GPU memory issues during setup check
        verification_provider = "CPUExecutionProvider"
        verification_opts = None
        logger.info(f"(Verification uses {verification_provider})")

        expected_quant_suffix = _select_onnx_quantization(
            hw_info, provider_name, provider_opts, onnx_quant_pref
        )
        logger.info(
            f"(Target suffix based on hardware/prefs: '{expected_quant_suffix}')"
        )

        # Load using the user's preference, but force CPU provider for the check
        llm = load_teapot_onnx_llm(
            onnx_quantization=onnx_quant_pref,
            preferred_provider=verification_provider,
            preferred_options=verification_opts,
        )
        if llm:
            logger.info(
                f"Teapot ONNX LLM ({llm.model_info.model_id}) loaded successfully from active directory."
            )
        else:
            raise RuntimeError("Teapot loader returned None.")
    except ModelNotFoundError as e:
        logger.error(
            f"Verification Failed: Teapot model/files missing/invalid in active dir. {e}"
        )
        all_verified = False
    except Exception as e:
        logger.error(
            f"Verification Failed: Error loading Teapot ONNX LLM: {e}", exc_info=True
        )
        all_verified = False
    finally:
        if llm and hasattr(llm, "unload"):
            llm.unload()
        # Explicit cleanup
        if llm is not None:
            del llm
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
            raise SetupError("Models directory path not configured.")
        models_dir = Path(models_dir_str)
        models_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Using models directory: {models_dir}")

        # Check Hugging Face token
        try:
            hf_token = HfFolder.get_token()
            if hf_token:
                logger.info("Hugging Face token found.")
            else:
                logger.warning(
                    "Hugging Face token not found. Downloads might be slower or fail for gated models."
                )
        except Exception:
            logger.warning("Could not check for Hugging Face token.")

        # Download/Verify components
        check_or_download_embedder(models_dir, args.force)
        check_or_download_spacy(args.force)
        check_or_download_teapot_onnx(models_dir, args.onnx_quant, args.force)

        # Final Verification
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
