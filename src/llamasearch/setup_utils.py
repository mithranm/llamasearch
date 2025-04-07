#!/usr/bin/env python
"""
setup_utils.py - Installation utilities for llama-cpp-python with acceleration support

This module installs llama-cpp-python with support for CPU, CUDA, and Metal (Apple Silicon).
It auto-detects your systems configuration, sets up the build environment,
builds from source if GPU support is desired, and downloads required models.
"""

import os
import platform
import subprocess
import sys
import logging
import datetime
import json
import traceback
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

DEBUG_LOG_PATH = ""

# -----------------------
# Environment and Path Utilities

def is_dev_mode() -> bool:
    """Determine if we're running in development mode."""
    if os.environ.get("LLAMASEARCH_DEV_MODE", "").lower() in ("1", "true", "yes"):
        return True
    try:
        import llamasearch  # type: ignore
        pkg_dir = Path(llamasearch.__file__).parent
        for parent in [pkg_dir] + list(pkg_dir.parents):
            if (parent / ".git").exists():
                return True
        if ".egg-link" in str(pkg_dir) or "site-packages" not in str(pkg_dir):
            return True
    except Exception:
        pass
    return False

def get_llamasearch_dir() -> Path:
    """Return the base directory for LlamaSearch data."""
    if is_dev_mode():
        current = Path(__file__).resolve().parent
        for parent in [current] + list(current.parents):
            if (parent / "pyproject.toml").exists():
                return parent
        return Path.cwd()
    else:
        if platform.system() == "Windows":
            base = os.path.join(os.environ.get("APPDATA", ""), "LlamaSearch")
        elif platform.system() == "Darwin":
            base = os.path.join(os.path.expanduser("~"), "Library", "Application Support", "LlamaSearch")
        else:
            base = os.path.join(os.path.expanduser("~"), ".llamasearch")
        return Path(os.environ.get("LLAMASEARCH_DATA_DIR", base))

def get_data_paths() -> Dict[str, Path]:
    """Return dictionary with required data directories."""
    base = get_llamasearch_dir()
    paths = {
        "base": base,
        "models": base / "models",
        "index": base / "index",
        "logs": base / "logs",
        "crawl_data": base / "crawl_data",
        "temp": base / "temp",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths

# -----------------------
# Setup Configuration Tracking

def get_setup_config_path() -> Path:
    """Return the path to the setup configuration file."""
    return get_llamasearch_dir() / "setup_config.json"

def load_setup_config() -> Dict[str, Any]:
    """Load the setup configuration from JSON file."""
    config_path = get_setup_config_path()
    if config_path.exists():
        try:
            with config_path.open("r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            pass
    return {
        "setup_complete": False,
        "steps_completed": [],
        "llama_cpp_installed": False,
        "spacy_models_installed": False,
        "torch_installed": False,
        "models_downloaded": False,
        "last_setup_attempt": None
    }

def save_setup_config(config: Dict[str, Any]) -> None:
    """Save the setup configuration to JSON file."""
    config["last_setup_attempt"] = datetime.datetime.now().isoformat()
    config_path = get_setup_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w") as f:
        json.dump(config, f, indent=2)

def mark_step_complete(step: str) -> None:
    """Mark a specific step as complete and update configuration."""
    config = load_setup_config()
    if step not in config["steps_completed"]:
        config["steps_completed"].append(step)
    if step == "llama_cpp_install":
        config["llama_cpp_installed"] = True
    elif step == "spacy_models_install":
        config["spacy_models_installed"] = True
    elif step == "torch_install":
        config["torch_installed"] = True
    elif step == "model_download":
        config["models_downloaded"] = True
    all_steps = ["llama_cpp_install", "spacy_models_install", "torch_install", "model_download"]
    if all(s in config["steps_completed"] for s in all_steps):
        config["setup_complete"] = True
    save_setup_config(config)

# -----------------------
# Logging and Subprocess Helpers

def setup_logger() -> None:
    """Configure logging to both console and file."""
    paths = get_data_paths()
    log_dir = paths["logs"]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"setup_debug_{timestamp}.log"
    global DEBUG_LOG_PATH
    DEBUG_LOG_PATH = str(log_file)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logging.info(f"Debug logs will be saved to: {DEBUG_LOG_PATH}")
    logging.info(f"Running in {'development' if is_dev_mode() else 'production'} mode")

def run_subprocess(cmd: List[str], env: Optional[Dict[str, str]] = None,
                   cwd: Optional[str] = None, check: bool = True) -> Dict[str, Any]:
    """Run a subprocess command and return stdout, stderr, and return code."""
    logging.info(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env, cwd=cwd, capture_output=True, text=True)
    if check and result.returncode != 0:
        logging.error(f"Command failed with return code {result.returncode}")
        logging.error(result.stderr)
        raise subprocess.CalledProcessError(result.returncode, cmd, output=result.stdout, stderr=result.stderr)
    return {"stdout": result.stdout, "stderr": result.stderr, "returncode": result.returncode}

def log_system_info() -> None:
    """Log system information for debugging."""
    info = [
        "=" * 80,
        "SYSTEM INFORMATION".center(80),
        "=" * 80,
        f"OS: {platform.system()} {platform.release()} {platform.version()}",
        f"Architecture: {platform.machine()}",
        f"Python version: {platform.python_version()}",
        f"Python executable: {sys.executable}",
        f"Development mode: {is_dev_mode()}",
        f"Data directory: {get_llamasearch_dir()}",
        "=" * 80,
    ]
    for line in info:
        logging.info(line)

# -----------------------
# Acceleration Detection and Build Environment

def detect_optimal_acceleration() -> Dict[str, Any]:
    """Auto-detect the optimal acceleration configuration."""
    result = {"type": "cpu", "cuda_version": None, "cuda_path": None,
              "nvcc_path": None, "cuda_wheel_version": None}
    sys_platform = platform.system()
    if sys_platform == "Windows":
        cuda_path = os.environ.get("CUDA_PATH")
        if cuda_path and os.path.exists(cuda_path):
            result.update({"cuda_path": cuda_path, "type": "cuda"})
            nvcc = os.path.join(cuda_path, "bin", "nvcc.exe")
            if os.path.exists(nvcc):
                result["nvcc_path"] = nvcc
            if "v12" in cuda_path:
                result["cuda_version"] = "12.4"
                result["cuda_wheel_version"] = "cu121"
            elif "v11.8" in cuda_path:
                result["cuda_version"] = "11.8"
                result["cuda_wheel_version"] = "cu118"
            elif "v11.7" in cuda_path:
                result["cuda_version"] = "11.7"
                result["cuda_wheel_version"] = "cu117"
            else:
                result["cuda_version"] = "11.6"
                result["cuda_wheel_version"] = "cu116"
    elif sys_platform == "Linux":
        try:
            proc = run_subprocess(["nvcc", "--version"], check=False)
            if proc["returncode"] == 0:
                result["type"] = "cuda"
                import re
                match = re.search(r"release (\d+\.\d+)", proc["stdout"])
                if match:
                    cuda_ver = match.group(1)
                    result["cuda_version"] = cuda_ver
                    if cuda_ver.startswith("12"):
                        result["cuda_wheel_version"] = "cu121"
                    elif cuda_ver.startswith("11.8"):
                        result["cuda_wheel_version"] = "cu118"
                    elif cuda_ver.startswith("11.7"):
                        result["cuda_wheel_version"] = "cu117"
                    elif cuda_ver.startswith("11"):
                        result["cuda_wheel_version"] = "cu116"
        except Exception:
            pass
    elif sys_platform == "Darwin" and platform.machine() == "arm64":
        result["type"] = "metal"
    return result

def map_cuda_version_to_wheel(version: str) -> Optional[str]:
    version = version.strip()
    if version.startswith("12"):
        return "cu121"
    elif version.startswith("11.8"):
        return "cu118"
    elif version.startswith("11.7"):
        return "cu117"
    elif version.startswith("11"):
        return "cu116"
    return None

def get_acceleration_config() -> Dict[str, Any]:
    """Return acceleration configuration from environment or auto-detection."""
    accel = os.environ.get("LLAMASEARCH_ACCELERATION", "auto").lower()
    if accel == "auto":
        return detect_optimal_acceleration()
    config = {"type": accel}
    if accel == "cuda":
        cuda_path = os.environ.get("CUDA_PATH")
        if cuda_path:
            config["cuda_path"] = cuda_path
        cuda_version = os.environ.get("LLAMASEARCH_CUDA_VERSION")
        if cuda_version:
            config["cuda_version"] = cuda_version
            wheel_ver = map_cuda_version_to_wheel(cuda_version)
            if wheel_ver:
                config["cuda_wheel_version"] = wheel_ver
    return config

def check_llama_cpp_cuda_support() -> bool:
    """Check if the installed llama-cpp-python package has CUDA support."""
    try:
        result = run_subprocess([sys.executable, "-m", "pip", "show", "llama-cpp-python"], check=False)
        if result["returncode"] != 0:
            logging.info("llama-cpp-python not installed.")
            return False
        
        # Method 1: Check version string
        import llama_cpp  # type: ignore
        if hasattr(llama_cpp, "__version__") and "cuda" in llama_cpp.__version__.lower():
            logging.info("CUDA support detected via version string.")
            return True
        
        # Method 2: Try to use library functions or GPU initialization
        from llama_cpp import Llama  # type: ignore
        
        # Method 3: Check if shared library supports GPU offload
        try:
            from llama_cpp.llama_cpp import load_shared_library
            import pathlib
            lib_path = pathlib.Path(llama_cpp.__file__).parent / "lib"
            lib = load_shared_library('llama', lib_path)
            if hasattr(lib, "llama_supports_gpu_offload") and callable(getattr(lib, "llama_supports_gpu_offload")):
                if bool(lib.llama_supports_gpu_offload()):
                    logging.info("CUDA support detected via llama_supports_gpu_offload().")
                    return True
        except Exception as lib_err:
            logging.debug(f"Error checking shared library GPU support: {lib_err}")
        
        # Method 4: Try to initialize model with GPU layers
        try:
            # Just initialize with a minimal configuration
            _ = Llama(model_path="", n_gpu_layers=1)
            logging.info("CUDA support detected via Llama instantiation.")
            return True
        except Exception as model_err:
            # Check if the error message mentions CUDA
            if "cuda" in str(model_err).lower():
                logging.info("CUDA support detected via CUDA-related error message.")
                return True
        
        logging.info("No CUDA support detected in llama-cpp-python.")
        return False
    except Exception as e:
        logging.error(f"Error during CUDA support check: {e}")
        return False

def configure_build_environment(config: Dict[str, Any]) -> Tuple[str, List[str], Dict[str, str]]:
    """Configure build environment for building llama-cpp-python from source."""
    cmake_args = []
    env_vars: Dict[str, str] = {}
    acc_type = config.get("type", "cpu")
    if acc_type == "metal":
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            cmake_args.append("-DLLAMA_METAL=on")
            logging.info("Configuring for Metal acceleration.")
        else:
            logging.warning("Metal acceleration requested but not supported; defaulting to CPU.")
            acc_type = "cpu"
    elif acc_type == "cuda":
        env_vars["FORCE_CMAKE"] = "1"
        cmake_args.extend(["-DGGML_CUDA=on", "-DCMAKE_CUDA_ARCHITECTURES=all"])
        cuda_path = config.get("cuda_path")
        if cuda_path and os.path.exists(cuda_path):
            env_vars["CUDA_PATH"] = cuda_path
            nvcc = config.get("nvcc_path") or os.path.join(cuda_path, "bin", "nvcc.exe")
            if os.path.exists(nvcc):
                env_vars["CUDACXX"] = nvcc
            logging.info(f"Using NVCC at: {env_vars.get('CUDACXX')}")
    if cmake_args:
        env_vars["CMAKE_ARGS"] = " ".join(cmake_args)
    return acc_type, cmake_args, env_vars

# -----------------------
# Main Installation Functions

def install_llama_cpp_python() -> bool:
    """Install llama-cpp-python from source with proper acceleration support."""
    LATEST_VERSION = "0.3.8"
    logging.info("=== Installing llama-cpp-python with acceleration support ===")
    acc_config = get_acceleration_config()
    acc_type = acc_config.get("type", "cpu")
    if acc_type == "metal" and not (platform.system() == "Darwin" and platform.machine() == "arm64"):
        logging.warning("Metal acceleration not supported on this platform; falling back to CPU.")
        acc_type = "cpu"
        acc_config["type"] = "cpu"
    elif acc_type == "cuda" and platform.system() == "Darwin":
        logging.warning("CUDA not available on macOS; switching to Metal for Apple Silicon or CPU otherwise.")
        acc_type = "metal" if platform.machine() == "arm64" else "cpu"
        acc_config["type"] = acc_type
    logging.info(f"Detected acceleration type: {acc_type}")
    _, cmake_args, env_vars = configure_build_environment(acc_config)
    try:
        run_subprocess([sys.executable, "-m", "pip", "uninstall", "-y", "llama-cpp-python"], check=False)
        logging.info("Uninstalled existing llama-cpp-python (if any).")
    except Exception as e:
        logging.warning(f"Error during uninstall: {e}")
    logging.info("Building llama-cpp-python from source (skipping pre-built wheels)...")
    build_env = os.environ.copy()
    build_env.update(env_vars)
    if platform.system() == "Windows" and acc_type == "cuda":
        cuda_path = acc_config.get("cuda_path")
        if cuda_path:
            build_env["FORCE_CMAKE"] = "1"
            build_env["CUDA_PATH"] = cuda_path
            nvcc = os.path.join(cuda_path, "bin", "nvcc.exe")
            if os.path.exists(nvcc):
                build_env["CUDACXX"] = nvcc
            cmake_args_str = f'-DGGML_CUDA=ON -DCMAKE_GENERATOR_TOOLSET="cuda={cuda_path}" -DCMAKE_CXX_STANDARD=17'
            build_env["CMAKE_ARGS"] = cmake_args_str
            logging.info(f"Windows CUDA configuration: {cmake_args_str}")
    else:
        if cmake_args and "CMAKE_ARGS" not in build_env:
            build_env["CMAKE_ARGS"] = " ".join(cmake_args)
        build_env["FORCE_CMAKE"] = "1"
    build_env["CMAKE_VERBOSE_MAKEFILE"] = "1"
    cmd = [
        sys.executable, "-m", "pip", "install",
        "--no-cache-dir", "--verbose", "--force-reinstall",
        f"llama-cpp-python=={LATEST_VERSION}"
    ]
    try:
        result = run_subprocess(cmd, env=build_env, check=True)
        paths = get_data_paths()
        build_log = paths["logs"] / "build_output.log"
        with build_log.open("w", encoding="utf-8") as f:
            f.write("STDOUT:\n" + result["stdout"] + "\n\nSTDERR:\n" + result["stderr"])
        logging.info(f"Successfully built llama-cpp-python v{LATEST_VERSION}.")
        if acc_type in ["cuda", "metal"] and check_llama_cpp_cuda_support():
            logging.info("GPU acceleration support verified in the built package.")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Error building llama-cpp-python: {e}")
        logging.error(traceback.format_exc())
        return False

def install_torch_with_cuda() -> bool:
    """Install PyTorch with appropriate support for the current platform."""
    if platform.system() == "Darwin":
        logging.info("=== Installing PyTorch for macOS ===")
        cmd = [sys.executable, "-m", "pip", "install", "-U", "--no-cache-dir", "torch", "torchvision", "torchaudio"]
        try:
            run_subprocess(cmd, check=True)
            verify = [
                sys.executable, "-c",
                "import torch; print('MPS available:', getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available()); "
                "print('PyTorch version:', torch.__version__)"
            ]
            logging.info(run_subprocess(verify, check=False)["stdout"].strip())
            return True
        except Exception as e:
            logging.error(f"Error installing PyTorch for macOS: {e}")
            return False
    else:
        logging.info("=== Installing PyTorch with CUDA support (if available) ===")
        cuda_version = "cu124"
        has_cuda = False
        try:
            proc = run_subprocess(["nvcc", "--version"], check=False)
            if proc["returncode"] == 0:
                has_cuda = True
                import re
                match = re.search(r"release (\d+\.\d+)", proc["stdout"])
                if match:
                    ver = float(match.group(1).split(".")[0])
                    cuda_version = "cu113" if ver < 11.0 else ("cu118" if ver < 12.0 else "cu124")
        except Exception as e:
            logging.warning(f"CUDA not detected: {e}")
        if has_cuda:
            cmd = [sys.executable, "-m", "pip", "install", "-U", "--no-cache-dir",
                   f"--index-url=https://download.pytorch.org/whl/{cuda_version}",
                   "torch", "torchvision", "torchaudio"]
            logging.info(f"Installing PyTorch with {cuda_version} support...")
        else:
            cmd = [sys.executable, "-m", "pip", "install", "-U", "--no-cache-dir",
                   "torch", "torchvision", "torchaudio"]
            logging.info("Installing CPU-only PyTorch...")
        try:
            run_subprocess(cmd, check=True)
            verify = [
                sys.executable, "-c",
                "import torch; print('CUDA available:', torch.cuda.is_available()); "
                "print('PyTorch version:', torch.__version__)"
            ]
            logging.info(run_subprocess(verify, check=False)["stdout"].strip())
            return True
        except Exception as e:
            logging.error(f"Error installing PyTorch: {e}")
            return False

def download_qwen_model() -> None:
    """Download the Qwen model if not already present."""
    paths = get_data_paths()
    models_dir = paths["models"]
    model_file = models_dir / "qwen2.5-1.5b-instruct-q4_k_m.gguf"
    if model_file.exists():
        logging.info(f"Qwen model already exists at {model_file}; skipping download.")
        return
    try:
        logging.info("Downloading Qwen2.5-1.5B-Instruct-GGUF model...")
        run_subprocess(["huggingface-cli", "download", "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
                        "qwen2.5-1.5b-instruct-q4_k_m.gguf", "--local-dir", str(models_dir)], check=True)
        logging.info(f"Model downloaded successfully to {models_dir}.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error downloading Qwen model: {e}")
        logging.error(f"Please download manually and place it into {models_dir}")

def install_spacy_models() -> None:
    """Install required spaCy models."""
    models = {"English": "en_core_web_trf", "Chinese": "zh_core_web_trf", "Russian": "ru_core_news_lg"}
    for lang, model in models.items():
        logging.info(f"Installing spaCy {lang} model: {model}...")
        try:
            subprocess.run([sys.executable, "-m", "spacy", "download", model], check=True)
            logging.info(f"spaCy {lang} model installed successfully.")
        except subprocess.CalledProcessError as e:
            logging.error(f"Error installing spaCy {lang} model: {e}")

def copy_cuda_visual_studio_files() -> None:
    """Optional Windows helper for CUDA integration (not implemented)."""
    if platform.system() != "Windows":
        return
    logging.info("Setting up CUDA integration for Visual Studio... [Not Implemented]")

def create_windows_cuda_installer() -> None:
    """Optional: Create a Windows batch script for forcing CUDA installation."""
    if platform.system() != "Windows":
        return
    paths = get_data_paths()
    script_path = paths["base"] / "windows-cuda-installer.bat"
    cuda_path = os.environ.get("CUDA_PATH", r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4")
    with script_path.open("w") as f:
        f.write("@echo off\n")
        f.write("echo Windows llama-cpp-python CUDA Installer\n")
        f.write("echo =====================================\n\n")
        f.write("echo Setting environment variables...\n")
        f.write(f'set "CUDA_PATH={cuda_path}"\n')
        f.write(f'set "CUDACXX={cuda_path}\\bin\\nvcc.exe"\n')
        f.write('set "FORCE_CMAKE=1"\n')
        f.write(f'set "CMAKE_ARGS=-DGGML_CUDA=ON -DCMAKE_GENERATOR_TOOLSET=\\"cuda={cuda_path}\\" -DCMAKE_CXX_STANDARD=17"\n\n')
        f.write("echo Uninstalling previous llama-cpp-python...\n")
        f.write("pip uninstall -y llama-cpp-python\n\n")
        f.write("echo Installing llama-cpp-python with CUDA support...\n")
        f.write("pip install --no-cache-dir --force-reinstall llama-cpp-python==0.3.8\n")
        f.write("\necho Done!\npause\n")
    logging.info(f"Created Windows CUDA installer script at: {script_path}")

def setup_dependencies() -> None:
    """Main function to set up all dependencies."""
    setup_logger()
    logging.info("=" * 80)
    logging.info("LlamaSearch Setup".center(80))
    logging.info("=" * 80)
    log_system_info()
    config = load_setup_config()
    force_setup = os.environ.get("LLAMASEARCH_FORCE_SETUP", "0").lower() in ("1", "true", "yes")
    if config.get("setup_complete") and not force_setup:
        logging.info("Setup already complete. To force reinstall, set LLAMASEARCH_FORCE_SETUP=1")
        return
    elif force_setup:
        logging.info("Forcing setup to run again...")
    # Ensure required directories exist
    for path in get_data_paths().values():
        path.mkdir(parents=True, exist_ok=True)
    if platform.system() == "Windows":
        copy_cuda_visual_studio_files()
        create_windows_cuda_installer()
    if not config.get("torch_installed"):
        if install_torch_with_cuda():
            mark_step_complete("torch_install")
        else:
            logging.warning("Failed to install PyTorch. Some features may not work properly.")
    else:
        logging.info("PyTorch already installed. Skipping...")
    if not config.get("llama_cpp_installed"):
        if install_llama_cpp_python():
            mark_step_complete("llama_cpp_install")
        else:
            logging.error("Failed to install llama-cpp-python. Setup cannot continue.")
            return
    else:
        logging.info("llama-cpp-python already installed. Skipping...")
    if not config.get("spacy_models_installed"):
        try:
            install_spacy_models()
            mark_step_complete("spacy_models_install")
        except Exception as e:
            logging.warning(f"Failed to install spaCy models: {e}")
    else:
        logging.info("spaCy models already installed. Skipping...")
    if not config.get("models_downloaded"):
        try:
            download_qwen_model()
            mark_step_complete("model_download")
        except Exception as e:
            logging.warning(f"Failed to download models: {e}")
    else:
        logging.info("Models already downloaded. Skipping...")
    config = load_setup_config()
    if config.get("setup_complete"):
        logging.info("=" * 80)
        logging.info("Setup Complete!".center(80))
        accel = get_acceleration_config().get("type")
        if accel == "cuda" and check_llama_cpp_cuda_support():
            logging.info("GPU (CUDA) acceleration is enabled for llama-cpp-python.")
        elif accel == "metal":
            logging.info("Metal acceleration is enabled for llama-cpp-python.")
        else:
            logging.warning("GPU acceleration is NOT enabled; running in CPU-only mode.")
    else:
        logging.error("Setup Incomplete. Some steps failed.")
        all_steps = ["llama_cpp_install", "spacy_models_install", "torch_install", "model_download"]
        incomplete = [step for step in all_steps if step not in config["steps_completed"]]
        logging.error(f"Incomplete steps: {', '.join(incomplete)}")
    logging.info(f"Full debug logs are available at: {DEBUG_LOG_PATH}")

def run_setup_command() -> None:
    """Entry point for command-line execution."""
    import argparse
    parser = argparse.ArgumentParser(description="LlamaSearch Setup Utility")
    parser.add_argument("--force", action="store_true", help="Force setup to run again")
    parser.add_argument("--dev-mode", action="store_true", help="Run in development mode")
    args = parser.parse_args()
    if args.force:
        os.environ["LLAMASEARCH_FORCE_SETUP"] = "1"
    if args.dev_mode:
        os.environ["LLAMASEARCH_DEV_MODE"] = "1"
    setup_dependencies()

if __name__ == "__main__":
    run_setup_command()