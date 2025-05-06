"""
setup.py - Downloads and verifies LlamaSearch models (CPU-Only).

Configured for a generic ONNX Causal LM (Llama 3.2 1B) and embedder.
Downloads necessary files based on user-specified quantization flag
and assembles the active model directory.
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
    HfHubHTTPError, # <<< Added specific HTTP error >>>
)
from huggingface_hub.utils._hf_folder import HfFolder

from llamasearch.core.embedder import DEFAULT_MODEL_NAME as DEFAULT_EMBEDDER_MODEL
from llamasearch.core.embedder import EnhancedEmbedder
from llamasearch.core.onnx_model import (
    MODEL_ONNX_BASENAME,
    ONNX_SUBFOLDER,
    ONNX_MODEL_REPO_ID,
    load_onnx_llm,
)
from llamasearch.core.onnx_model import LLM  # Import LLM protocol
from llamasearch.data_manager import data_manager
from llamasearch.exceptions import ModelNotFoundError, SetupError
from llamasearch.utils import setup_logging

logger = setup_logging("llamasearch.setup")


# Helper: Download with Retries (No change needed)
def download_file_with_retry(
    repo_id: str,
    filename: str,
    cache_dir: Path,
    force: bool,
    max_retries: int = 2,
    delay: int = 5,
    repo_type: Optional[str] = "model",
    **kwargs,
) -> str: # <<< Return type hint added >>>
    """Attempts to download a file with retries on failure."""
    assert isinstance(cache_dir, Path)
    last_error : Optional[Exception] = None # Track last error
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
            # Check if path is None or empty string, which can happen
            if not file_path_str:
                 raise FileNotFoundError(f"hf_hub_download returned invalid path: {file_path_str}")

            fpath = Path(file_path_str)
            # Check if file actually exists after download call returns
            if not fpath.is_file(): # Check is_file() for robustness
                raise FileNotFoundError(
                    f"File {filename} invalid or DNE after DL attempt {attempt + 1} at path {fpath}."
                )
            logger.debug(f"OK DL {filename} -> {file_path_str}")
            return file_path_str # Return the string path
        except EntryNotFoundError as e:
            logger.debug(
                f"File {filename} not in repo {repo_id} (attempt {attempt + 1})."
            )
            last_error = e # Store last error
            raise # Reraise immediately, no retry for 404
        except (ConnectionError, TimeoutError, HfHubHTTPError, FileNotFoundError) as e: # Catch more specific errors
            logger.warning(f"DL attempt {attempt + 1} for {filename} failed: {e}")
            last_error = e # Store last error
            if attempt < max_retries:
                logger.info(f"Retrying download of {filename} in {delay} seconds...")
                time.sleep(delay)
            else:
                 # Raise SetupError wrapping the last encountered error
                raise SetupError(f"Failed DL after retries: {filename}") from last_error
        except Exception as e:
            logger.error(
                f"Unexpected DL error {filename} attempt {attempt + 1}: {e}",
                exc_info=True,
            )
            last_error = e # Store last error
            raise SetupError(f"Failed DL unexpected: {filename}") from last_error
    # This should not be reachable if max_retries >= 0
    raise SetupError(f"Download logic error for {filename}.")


# --- Embedder Check/Download (No functional change needed) ---
def check_or_download_embedder(models_dir: Path, force: bool = False) -> None:
    model_name = DEFAULT_EMBEDDER_MODEL
    logger.info(f"Checking/Downloading Embedder: {model_name}")
    ignore_patterns: List[str] = ["*.onnx", "onnx/*", "*.gguf", "gguf/*", "openvino/*"]
    logger.info(f"Ignoring patterns: {ignore_patterns}")
    try:
        # Attempt local check first only if not forcing
        if not force:
            try:
                logger.debug(f"Checking locally for {model_name} in {models_dir}")
                snapshot_download(
                    repo_id=model_name,
                    cache_dir=models_dir,
                    local_files_only=True, # <<< Check locally >>>
                    local_dir_use_symlinks=False,
                    ignore_patterns=ignore_patterns,
                )
                logger.info(f"Embedder '{model_name}' (PyTorch) found locally.")
                return # Found locally, nothing more to do
            except (EntryNotFoundError, LocalEntryNotFoundError, FileNotFoundError):
                logger.info(
                    f"Embedder '{model_name}' (PyTorch) not found/incomplete locally. Proceeding to download/verify..."
                )
            except Exception as local_check_err:
                 logger.warning(f"Local check for embedder failed ({local_check_err}), proceeding to download/verify...")

        # Download or verify remote if local check failed or force=True
        logger.info(f"Downloading/Verifying embedder {model_name} from Hub...")
        snapshot_download(
            repo_id=model_name,
            cache_dir=models_dir,
            force_download=force,
            resume_download=not force,
            local_files_only=False, # <<< Ensure fetching from remote >>>
            local_dir_use_symlinks=False,
            ignore_patterns=ignore_patterns,
        )
        logger.info(f"Embedder '{model_name}' (PyTorch) cache verified/downloaded.")
    except Exception as e:
        # Catch any exception during download/verification
        raise SetupError(f"Failed get/verify embedder {model_name}") from e


# --- Generic ONNX LLM Check/Download (Modified for explicit suffix) ---
def check_or_download_onnx_llm(
    models_dir: Path, quant_suffix: str, force: bool = False
) -> None:
    """Downloads required generic ONNX LLM files based on the chosen suffix."""
    quant_name = quant_suffix.lstrip("_") if quant_suffix else "fp32"
    logger.info(f"Checking/Downloading Generic ONNX LLM (Quantization: {quant_name})")
    logger.info(f"Targeting ONNX suffix: '{quant_suffix}' for assembly")

    active_model_dir = models_dir / "active_model"
    active_onnx_dir = active_model_dir / ONNX_SUBFOLDER
    needs_clean = force

    # Determine if cleaning is needed even if force=False
    if not needs_clean and active_model_dir.exists():
        # Check if the specific ONNX file for the target suffix exists
        target_onnx_file = active_onnx_dir / f"{MODEL_ONNX_BASENAME}{quant_suffix}.onnx"
        if not target_onnx_file.is_file(): # Use is_file() for robustness
            needs_clean = True
            logger.warning(
                f"Target ONNX file '{target_onnx_file.name}' missing in {active_onnx_dir}. Forcing clean assembly."
            )
        # Also check for data file if target is fp32
        elif quant_suffix == "":
            target_data_file = active_onnx_dir / f"{MODEL_ONNX_BASENAME}.onnx_data"
            if not target_data_file.is_file(): # Use is_file()
                needs_clean = True
                logger.warning(
                    f"Target ONNX data file '{target_data_file.name}' missing for fp32 in {active_onnx_dir}. Forcing clean assembly."
                )

    # Clean if forced or if target files are missing
    if needs_clean and active_model_dir.exists():
        logger.info(f"Cleaning existing active model directory: {active_model_dir}")
        try:
            shutil.rmtree(active_model_dir)
        except OSError as e:
            raise SetupError(f"Failed clear active dir {active_model_dir}: {e}") from e

    # Ensure directories exist before proceeding
    active_model_dir.mkdir(parents=True, exist_ok=True)
    active_onnx_dir.mkdir(parents=True, exist_ok=True)

    cache_location = models_dir
    repo_id = ONNX_MODEL_REPO_ID

    # 1. Download all root-level config/tokenizer files etc. into active_model_dir
    logger.info(f"Downloading/Verifying root files from {repo_id} into {active_model_dir}...")
    # Define files expected at the root level (adjust if repo structure changes)
    root_files_allow = [
        "config.json", "generation_config.json", "special_tokens_map.json",
        "tokenizer.json", "tokenizer_config.json"
    ]
    # Download only the allowed root files directly into the target active_model_dir
    try:
        snapshot_download(
            repo_id=repo_id,
            cache_dir=cache_location, # Still use main cache for underlying storage
            local_dir=active_model_dir,  # <<< Download directly into active dir >>>
            local_dir_use_symlinks=False,
            allow_patterns=root_files_allow, # <<< Only download these specific files >>>
            ignore_patterns=["*", "*/*"], # Ignore everything else initially
            force_download=force,
            resume_download=not force,
            repo_type="model",
        )
        logger.info(f"Root config/tokenizer files downloaded/verified in {active_model_dir}")
    except Exception as e:
        raise SetupError(f"Failed to download/verify root files from {repo_id}: {e}") from e

    # 2. Download Specific ONNX Model File based on suffix (to cache) and copy
    onnx_model_rel_path = f"{ONNX_SUBFOLDER}/{MODEL_ONNX_BASENAME}{quant_suffix}.onnx"
    logger.info(f"Downloading/Verifying ONNX model file: '{onnx_model_rel_path}'...")
    try:
        source_path_str = download_file_with_retry(
            repo_id=repo_id,
            filename=onnx_model_rel_path,
            cache_dir=cache_location,  # Download to main cache first
            force=force,
            repo_type="model",
        )
        # source_path_str is guaranteed to be non-None and exist if no exception
        target_onnx_path = active_onnx_dir / Path(onnx_model_rel_path).name
        logger.debug(f"Copying {source_path_str} to {target_onnx_path}")
        shutil.copyfile(source_path_str, target_onnx_path) # Copy from cache to active
        logger.debug(f"Copied {onnx_model_rel_path} to {target_onnx_path}")
    except EntryNotFoundError:
        logger.error(
            f"Required ONNX file '{onnx_model_rel_path}' not found in repo {repo_id}."
        )
        raise SetupError(f"Required ONNX file missing: {onnx_model_rel_path}")
    except (SetupError, IOError, OSError) as e: # Catch copy errors too
        raise SetupError(f"Error getting/copying ONNX file {onnx_model_rel_path}: {e}") from e
    except Exception as e:
        raise SetupError(f"Unexpected error processing ONNX file {onnx_model_rel_path}: {e}") from e


    # 3. Download Corresponding .onnx_data File (ONLY if suffix is "" for fp32)
    if quant_suffix == "":
        onnx_data_rel_path = f"{ONNX_SUBFOLDER}/{MODEL_ONNX_BASENAME}.onnx_data"
        logger.info(f"Checking/Downloading ONNX data file: '{onnx_data_rel_path}' (fp32)...")
        try:
            source_path_str = download_file_with_retry(
                repo_id=repo_id,
                filename=onnx_data_rel_path,
                cache_dir=cache_location,  # Download to main cache
                force=force,
                repo_type="model",
            )
            # source_path_str is guaranteed to be non-None and exist if no exception
            target_data_path = active_onnx_dir / Path(onnx_data_rel_path).name
            logger.debug(f"Copying {source_path_str} to {target_data_path}")
            shutil.copyfile(source_path_str, target_data_path) # Copy from cache to active
            logger.debug(f"Copied {onnx_data_rel_path} to {target_data_path}")
            logger.info(f"Found and processed ONNX data file: {onnx_data_rel_path}")
        except EntryNotFoundError:
            # This IS an error for fp32
            logger.error(
                f"Required ONNX data file '{onnx_data_rel_path}' not found for fp32 model in repo {repo_id}."
            )
            raise SetupError(
                f"Required ONNX data file missing for fp32: {onnx_data_rel_path}"
            )
        except (SetupError, IOError, OSError) as e: # Catch copy errors too
            raise SetupError(
                f"Error getting/copying ONNX data file {onnx_data_rel_path}: {e}"
            ) from e
        except Exception as e:
            raise SetupError(
                f"Unexpected error processing ONNX data file {onnx_data_rel_path}: {e}"
            ) from e
    else:
        logger.info(f"Skipping .onnx_data check (quantization: {quant_name})")

    logger.info(f"Assembly of {active_model_dir} complete for {quant_name}.")


# --- Verification Function (Modified for explicit suffix) ---
def verify_setup(quant_suffix_to_verify: str):
    """Attempts to load models based on the *installed* suffix in active_model."""
    logger.info("--- Verifying Model Setup (CPU-Only) ---")
    all_verified = True
    embedder = None
    llm = None # Initialize llm to None

    # Verify Embedder (No change needed)
    logger.info(f"Verifying Embedder '{DEFAULT_EMBEDDER_MODEL}' (CPU)...")
    try:
        embedder = EnhancedEmbedder()
        dim = embedder.get_embedding_dimension()
        if dim and dim > 0 and embedder.model is not None:
            logger.info(f"Embedder OK (CPU, Dim: {dim}).")
        else:
            # Raise error instead of just logging
            raise SetupError(f"Embedder invalid state (Model:{embedder.model is not None}, Dim:{dim}). Check logs.")
    except ModelNotFoundError as e:
        logger.error(f"FAIL: Embedder model not found. {e}")
        all_verified = False
        # Reraise as SetupError for main loop to catch
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
            del embedder # Aid garbage collection
            gc.collect()

    # Verify Generic ONNX LLM (CPU) - Loads whatever is in active_model
    logger.info("Verifying Generic ONNX LLM (CPU in active_model)...")
    try:
        # load_onnx_llm now detects the suffix inside active_model
        # Pass "auto" so it performs detection based on the assembled active_model dir
        llm = load_onnx_llm(onnx_quantization="auto")
        if llm and isinstance(llm, LLM):
            # Retrieve suffix actually loaded by load_onnx_llm
            # Accessing protected member _quant_suffix is necessary here for verification logic
            loaded_suffix = getattr(llm, "_quant_suffix", "unknown")

            logger.info(
                f"ONNX LLM OK ({llm.model_info.model_id} - suffix '{loaded_suffix}' loaded from active_model on CPU)."
            )
            # Check if loaded suffix matches the one setup *intended* to install
            # This ensures the correct files were assembled
            if loaded_suffix != quant_suffix_to_verify:
                logger.error(
                    f"FAIL: Verification loaded suffix '{loaded_suffix}' but setup intended to install '{quant_suffix_to_verify}'."
                )
                all_verified = False
                raise SetupError("Loaded ONNX model quantization does not match setup intent.")
        else:
             # Should not happen if load_onnx_llm doesn't raise error, but handle defensively
            raise SetupError("ONNX LLM loader returned None or unexpected type.")
    except ModelNotFoundError as e:
        logger.error(f"FAIL: ONNX LLM files missing or incomplete in active_model. {e}")
        all_verified = False
        # Reraise as SetupError
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
            del llm # Aid garbage collection
            gc.collect()

    if not all_verified:
        # This path might not be reached if exceptions are raised earlier,
        # but kept for logical completeness.
        raise SetupError("One or more models failed CPU verification.")
    else:
        logger.info("--- Model Verification Successful (CPU-Only) ---")


# --- Main Setup Function (Modified argparse) ---
def main():
    parser = argparse.ArgumentParser(
        description="Download/verify models for LlamaSearch (CPU-Only)."
    )
    parser.add_argument(
        "--force", action="store_true", help="Force redownload/reassembly"
    )
    # Use mutually exclusive group for quantization flags
    quant_group = parser.add_mutually_exclusive_group()
    quant_group.add_argument(
        "--fp32",
        action="store_const",
        dest="quant_suffix",
        const="",
        help="Download FP32 ONNX model (default).",
    )
    quant_group.add_argument(
        "--fp16",
        action="store_const",
        dest="quant_suffix",
        const="_fp16",
        help="Download FP16 ONNX model.",
    )
    quant_group.add_argument(
        "--int8",
        action="store_const",
        dest="quant_suffix",
        const="_int8",
        help="Download INT8 ONNX model.",
    )
    quant_group.add_argument(
        "--quantized",
        action="store_const",
        dest="quant_suffix",
        const="_quantized",
        help="Download generic 'quantized' ONNX model.",
    )
    quant_group.add_argument(
        "--q4",
        action="store_const",
        dest="quant_suffix",
        const="_q4",
        help="Download Q4 ONNX model.",
    )
    quant_group.add_argument(
        "--q4f16",
        action="store_const",
        dest="quant_suffix",
        const="_q4f16",
        help="Download Q4_f16 ONNX model.",
    )
    quant_group.add_argument(
        "--uint8",
        action="store_const",
        dest="quant_suffix",
        const="_uint8",
        help="Download UINT8 ONNX model.",
    )
    quant_group.add_argument(
        "--bnb4",
        action="store_const",
        dest="quant_suffix",
        const="_bnb4",
        help="Download BNB 4-bit ONNX model.",
    )

    parser.set_defaults(quant_suffix="")  # Default to FP32 (empty suffix)

    args = parser.parse_args()
    logger.info("--- Starting LlamaSearch Model Setup (CPU-Only) ---")
    if args.force:
        logger.info("Force mode enabled: Active directory will be recreated.")

    # Determine the chosen quantization suffix
    chosen_suffix = args.quant_suffix
    chosen_quant_name = chosen_suffix.lstrip("_") if chosen_suffix else "fp32"
    logger.info(
        f"User selected quantization: {chosen_quant_name} (suffix: '{chosen_suffix}')"
    )

    try:
        logger.info("Setup configured for CPU-only operation.")
        models_dir_str = data_manager.get_data_paths().get("models")
        if not models_dir_str:
            raise SetupError("Models directory path not configured.")
        models_dir = Path(models_dir_str)
        models_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Using models directory: {models_dir}")

        # Check HF Token (optional, provide warning)
        try:
            if HfFolder.get_token():
                logger.info("HF token found.")
            else:
                logger.warning("HF token not found. Downloads might fail for gated models (not applicable to default models).")
        except Exception as token_err:
            logger.warning(f"Could not check HF token: {token_err}")

        # Download/Verify components
        check_or_download_embedder(models_dir, args.force)
        # Pass the chosen suffix directly for download/assembly
        check_or_download_onnx_llm(models_dir, chosen_suffix, args.force)

        # Final Verification (CPU) - verify the suffix that was intended to be installed
        verify_setup(chosen_suffix)
        logger.info(
            f"--- LlamaSearch Model Setup Completed Successfully (Quant: {chosen_quant_name}) ---"
        )
        sys.exit(0) # Success exit code

    except SetupError as e:
        logger.error(f"Setup failed: {e}")
        sys.exit(1) # Failure exit code
    except Exception as e:
        logger.error(f"Unexpected error during setup: {e}", exc_info=True)
        sys.exit(1) # Failure exit code


if __name__ == "__main__":
    main()