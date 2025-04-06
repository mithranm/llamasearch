"""
setup_utils.py – Installation utilities for llama-cpp-python with acceleration support

This module installs llama-cpp-python with support for CPU, CUDA, and Metal (Apple Silicon)
acceleration. It auto-detects the system configuration, sets up the build environment, and
builds from source if GPU support is desired (skipping pre-built wheels). It also downloads
the spaCy English model and a Qwen model.
"""

import os
import platform
import subprocess
import sys
import logging
import datetime
import traceback
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

# Global variable for debug log path
DEBUG_LOG_PATH = ""

# -----------------------------------------------------------------------------
# Utility functions

def find_project_root() -> str:
    """
    Find the root of the project by looking for a 'pyproject.toml' file.
    If not found, return the directory of this file.
    """
    current_dir = os.path.abspath(os.path.dirname(__file__))
    while True:
        if os.path.exists(os.path.join(current_dir, "pyproject.toml")):
            return current_dir
        parent = os.path.dirname(current_dir)
        if parent == current_dir:
            break
        current_dir = parent
    return os.path.abspath(os.path.dirname(__file__))

def setup_logger() -> None:
    """Set up logging to both console and a file in the project root."""
    log_dir = Path(find_project_root()) / "logs"
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"setup_debug_{timestamp}.log"
    global DEBUG_LOG_PATH
    DEBUG_LOG_PATH = str(log_file)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # File handler logs DEBUG and above.
    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Console handler logs INFO and above.
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.handlers = []  # Clear existing handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logging.info(f"Debug logs will be saved to: {DEBUG_LOG_PATH}")
    print(f"[INFO] Debug logs will be saved to: {DEBUG_LOG_PATH}")

def run_subprocess(cmd: List[str],
                   env: Optional[Dict[str, str]] = None,
                   cwd: Optional[str] = None,
                   check: bool = True) -> Dict[str, Any]:
    """Run a subprocess command and return its stdout, stderr, and return code."""
    logging.info(f"Running command: {' '.join(cmd)}")
    print(f"[INFO] Running command: {' '.join(cmd)}")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               env=env, cwd=cwd, universal_newlines=True)
    stdout, stderr = process.communicate()
    if check and process.returncode != 0:
        logging.error(f"Command failed with return code {process.returncode}")
        logging.error(stderr)
        print(f"[ERROR] Command failed with return code {process.returncode}")
        raise subprocess.CalledProcessError(process.returncode, cmd)
    return {"stdout": stdout, "stderr": stderr, "returncode": process.returncode}

def log_system_info() -> None:
    """Log basic system information for debugging purposes."""
    info = (
        "=" * 80,
        "SYSTEM INFORMATION".center(80),
        "=" * 80,
        f"OS: {platform.system()} {platform.release()} {platform.version()}",
        f"Architecture: {platform.machine()}",
        f"Python version: {platform.python_version()}",
        f"Python executable: {sys.executable}",
        "=" * 80
    )
    for line in info:
        logging.info(line)
        print(line)

# -----------------------------------------------------------------------------
# Functions for GPU (CUDA) detection and configuration

def detect_optimal_acceleration() -> Dict[str, Any]:
    """
    Detect the optimal acceleration configuration.
    For Windows/Linux with NVIDIA GPUs, check for CUDA availability.
    For Darwin on arm64, enable Metal acceleration.
    Otherwise, default to CPU.
    """
    result = {
        "type": "cpu",
        "cuda_version": None,
        "cuda_path": None,
        "nvcc_path": None,
        "cuda_wheel_version": None
    }
    
    # Check for Windows CUDA
    if platform.system() == "Windows":
        cuda_path = os.environ.get("CUDA_PATH")
        if cuda_path and os.path.exists(cuda_path):
            result["cuda_path"] = cuda_path
            result["type"] = "cuda"
            nvcc_path = os.path.join(cuda_path, "bin", "nvcc.exe")
            if os.path.exists(nvcc_path):
                result["nvcc_path"] = nvcc_path
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
    # Check for Linux CUDA
    elif platform.system() == "Linux":
        try:
            # Check if nvcc is available on Linux
            process = subprocess.Popen(["nvcc", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, _ = process.communicate()
            if process.returncode == 0:
                result["type"] = "cuda"
                # Try to parse the CUDA version
                output = stdout.decode("utf-8")
                import re
                match = re.search(r"release (\d+\.\d+)", output)
                if match:
                    cuda_version = match.group(1)
                    result["cuda_version"] = cuda_version
                    # Map to wheel version
                    if cuda_version.startswith("12"):
                        result["cuda_wheel_version"] = "cu121"
                    elif cuda_version.startswith("11.8"):
                        result["cuda_wheel_version"] = "cu118"
                    elif cuda_version.startswith("11.7"):
                        result["cuda_wheel_version"] = "cu117"
                    elif cuda_version.startswith("11"):
                        result["cuda_wheel_version"] = "cu116"
        except (subprocess.SubprocessError, FileNotFoundError):
            # CUDA not available, stay with CPU
            pass
    # Check for Apple Silicon
    elif platform.system() == "Darwin" and platform.machine() == "arm64":
        result["type"] = "metal"
    
    return result

def map_cuda_version_to_wheel(version: str) -> Optional[str]:
    """
    Map a detected CUDA version string to a wheel version tag.
    """
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
    """
    Get the acceleration configuration from the environment or auto-detection.
    Environment variable LLAMASEARCH_ACCELERATION may override auto-detection.
    """
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
            wheel_version = map_cuda_version_to_wheel(cuda_version)
            if wheel_version:
                config["cuda_wheel_version"] = wheel_version
    return config

def check_llama_cpp_cuda_support() -> bool:
    """
    Check if the installed llama-cpp-python package has CUDA support.
    For version 0.3.8, we first try to check for a module-level attribute __cuda_supported__,
    and if that doesn't exist, we check for the attribute llama_backend and see if it contains 'cuda'.
    """
    try:
        import llama_cpp
        # Method 1: Try using the llama_cpp library's load_shared_library function (0.3.x+)
        try:
            from llama_cpp._ctypes_extensions import load_shared_library
            import pathlib
            lib_path = pathlib.Path(llama_cpp.__file__).parent / "lib"
            lib = load_shared_library('llama', lib_path)
            return bool(lib.llama_supports_gpu_offload())
        except (ImportError, AttributeError):
            # Method 2: Check if CUDA is in the version string (fallback)
            if hasattr(llama_cpp, "__version__") and "cuda" in llama_cpp.__version__.lower():
                return True
            # If all else fails, assume no CUDA support
            return False
    except Exception as e:
        logging.debug(f"Error checking CUDA support: {e}")
        return False

def configure_build_environment(config: Dict[str, Any]) -> Tuple[str, List[str], Dict[str, str]]:
    """
    Configure the build environment for building llama-cpp-python from source.
    Returns a tuple: (acceleration_type, cmake_args, env_vars)
    """
    cmake_args = []
    env_vars: Dict[str, str] = {}
    acceleration_type = config.get("type", "cpu")
    if acceleration_type == "metal":
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            cmake_args.append("-DLLAMA_METAL=on")
            logging.info("Configuring for Metal acceleration")
        else:
            logging.warning("Metal acceleration requested but not supported; falling back to CPU")
            acceleration_type = "cpu"
    elif acceleration_type == "cuda":
        env_vars["FORCE_CMAKE"] = "1"
        cmake_args.append("-DGGML_CUDA=on")
        cmake_args.append("-DCMAKE_CUDA_ARCHITECTURES=all")
        cuda_path = config.get("cuda_path")
        if cuda_path and os.path.exists(cuda_path):
            env_vars["CUDA_PATH"] = cuda_path
            nvcc_path = config.get("nvcc_path")
            if nvcc_path and os.path.exists(nvcc_path):
                env_vars["CUDACXX"] = nvcc_path
            else:
                nvcc_path = os.path.join(cuda_path, "bin", "nvcc.exe")
                if os.path.exists(nvcc_path):
                    env_vars["CUDACXX"] = nvcc_path
            logging.info(f"Using NVCC at: {env_vars.get('CUDACXX')}")
    if cmake_args:
        env_vars["CMAKE_ARGS"] = " ".join(cmake_args)
    return acceleration_type, cmake_args, env_vars

# -----------------------------------------------------------------------------
# Main installation functions

def install_llama_cpp_python() -> bool:
    """
    Install llama-cpp-python with the detected acceleration support.
    Uninstalls any previous installation and always builds from source.
    """
    LATEST_VERSION = "0.3.8"
    logging.info("\n=== Installing llama-cpp-python with acceleration support ===")
    print("\n[INFO] Installing llama-cpp-python with acceleration support...")
    acceleration_config = get_acceleration_config()
    acceleration_type = acceleration_config.get("type", "cpu")
    
    # Additional checks for platform compatibility
    if acceleration_type == "metal" and (platform.system() != "Darwin" or platform.machine() != "arm64"):
        logging.warning("Metal acceleration requested but not supported on this platform. Falling back to CPU.")
        print("[WARNING] Metal acceleration requested but not supported on this platform. Falling back to CPU.")
        acceleration_type = "cpu"
        acceleration_config["type"] = "cpu"
    elif acceleration_type == "cuda" and platform.system() == "Darwin":
        logging.warning("CUDA acceleration is not available on macOS. Falling back to CPU or Metal depending on hardware.")
        print("[WARNING] CUDA acceleration is not available on macOS. Falling back to CPU or Metal depending on hardware.")
        if platform.machine() == "arm64":
            acceleration_type = "metal"
            acceleration_config["type"] = "metal"
            logging.info("Detected Apple Silicon Mac. Using Metal acceleration instead.")
            print("[INFO] Detected Apple Silicon Mac. Using Metal acceleration instead.")
        else:
            acceleration_type = "cpu"
            acceleration_config["type"] = "cpu"
    
    logging.info(f"Acceleration type: {acceleration_type}")
    print(f"[INFO] Detected acceleration type: {acceleration_type}")
    _, cmake_args, env_vars = configure_build_environment(acceleration_config)

    # Uninstall previous installation
    try:
        run_subprocess([sys.executable, "-m", "pip", "uninstall", "-y", "llama-cpp-python"], check=False)
        print("[INFO] Uninstalled any existing llama-cpp-python installation.")
    except Exception as e:
        logging.warning(f"Error uninstalling previous version: {e}")

    # Always build from source (skipping pre-built wheels)
    logging.info("Building llama-cpp-python from source (skipping pre-built wheels)...")
    print("[INFO] Building llama-cpp-python from source (skipping pre-built wheels)...")
    build_env = os.environ.copy()
    build_env.update(env_vars)

    # Windows-specific configuration for CUDA: ensure proper quoting of the CUDA path.
    if platform.system() == "Windows" and acceleration_type == "cuda":
        cuda_path = acceleration_config.get("cuda_path")
        if cuda_path:
            build_env["FORCE_CMAKE"] = "1"
            build_env["CUDA_PATH"] = cuda_path
            build_env["CUDACXX"] = os.path.join(cuda_path, "bin", "nvcc.exe")
            cmake_args_str = f'-DGGML_CUDA=ON -DCMAKE_GENERATOR_TOOLSET="cuda={cuda_path}"'
            build_env["CMAKE_ARGS"] = cmake_args_str
            logging.info(f"Using Windows CUDA configuration: {cmake_args_str}")
            print(f"[INFO] Configuring Windows CUDA build with CMAKE_ARGS: {cmake_args_str}")
            logging.info(f"CUDA_PATH: {cuda_path}")
            logging.info(f"CUDACXX: {build_env['CUDACXX']}")
            build_env["CMAKE_VERBOSE_MAKEFILE"] = "1"
            build_env["VERBOSE"] = "1"
            build_env["CMAKE_ARGS"] += " -DCMAKE_CXX_STANDARD=17"
        else:
            logging.warning("CUDA path not found; cannot configure CUDA build on Windows")
    else:
        if "CMAKE_ARGS" not in build_env and cmake_args:
            build_env["CMAKE_ARGS"] = " ".join(cmake_args)
        build_env["FORCE_CMAKE"] = "1"

    logging.info("Build configuration:")
    for key in ["CMAKE_ARGS", "FORCE_CMAKE", "CUDA_PATH", "CUDACXX"]:
        if key in build_env:
            logging.info(f"- {key}: {build_env[key]}")
    build_env["CMAKE_VERBOSE"] = "1"

    cmd = [
        sys.executable, "-m", "pip", "install",
        "--no-cache-dir",
        "--verbose",
        "--force-reinstall",
        f"llama-cpp-python=={LATEST_VERSION}"
    ]
    try:
        result = run_subprocess(cmd, env=build_env, check=True)
        build_log_path = os.path.join(os.path.dirname(DEBUG_LOG_PATH), "build_output.log")
        with open(build_log_path, "w", encoding="utf-8") as f:
            f.write("STDOUT:\n" + result["stdout"] + "\n\nSTDERR:\n" + result["stderr"])
        logging.info(f"Successfully built llama-cpp-python v{LATEST_VERSION} from source")
        print(f"[INFO] Successfully built llama-cpp-python v{LATEST_VERSION} from source")
        logging.info(f"Full build logs saved to: {build_log_path}")
        if acceleration_type in ["cuda", "metal"]:
            if check_llama_cpp_cuda_support():
                logging.info("GPU acceleration support verified in the built package")
                print("[INFO] GPU acceleration support verified in the built package")
                return True
            else:
                logging.warning("Built from source but GPU support not verified")
                print("[WARNING] Built from source but GPU support not verified")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Error building llama-cpp-python: {e}")
        print(f"[ERROR] Error building llama-cpp-python: {e}")
        logging.error(traceback.format_exc())
        return False

# -----------------------------------------------------------------------------
# Additional dependency installations

def install_torch_with_cuda() -> bool:
    """
    Install PyTorch with appropriate acceleration support based on platform:
    - Windows/Linux with NVIDIA GPUs: CUDA support
    - macOS with Apple Silicon: MPS (Metal Performance Shaders) support
    - Other platforms: CPU-only
    """
    system = platform.system()
    
    # For macOS (especially Apple Silicon), use the default PyTorch installation
    if system == "Darwin":
        logging.info("\n=== Installing PyTorch for macOS ===")
        print("\n[INFO] Installing PyTorch for macOS...")
        
        cmd = [
            sys.executable, "-m", "pip", "install", "-U", 
            "--no-cache-dir",
            "torch", "torchvision", "torchaudio"
        ]
        
        try:
            logging.info("Installing PyTorch with native macOS support...")
            print("[INFO] Installing PyTorch with native macOS support...")
            result = run_subprocess(cmd, check=True)
            
            # Verify the installation
            verify_cmd = [
                sys.executable, "-c", 
                "import torch; print('MPS available:', torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False); "
                "print('PyTorch version:', torch.__version__); "
                "print('Device:', 'MPS' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'CPU')"
            ]
            verify_result = run_subprocess(verify_cmd, check=False)
            output = verify_result.get("stdout", "")
            
            logging.info("Successfully installed PyTorch for macOS")
            print("[INFO] Successfully installed PyTorch for macOS")
            logging.info(output.strip())
            print(output.strip())
            return True
            
        except Exception as e:
            logging.error(f"Error installing PyTorch for macOS: {e}")
            print(f"[ERROR] Error installing PyTorch for macOS: {e}")
            return False
    
    # For Windows/Linux, check for CUDA support
    logging.info("\n=== Installing PyTorch with CUDA support ===")
    print("\n[INFO] Installing PyTorch with CUDA support...")
    
    # Default to using CUDA 12.4 which is compatible with CUDA 12.8 drivers
    cuda_version = "cu124"
    has_cuda = False
    
    # Try to determine installed CUDA version using nvcc (CUDA compiler)
    try:
        result = run_subprocess(["nvcc", "--version"], check=False)
        if result["returncode"] == 0:
            has_cuda = True
            output = result.get("stdout", "")
            if "release" in output and "V" in output:
                import re
                # Look for patterns like "release 12.4, V12.4.99"
                match = re.search(r"release (\d+\.\d+)", output)
                if match:
                    detected_version = match.group(1)
                    logging.info(f"Detected CUDA version: {detected_version}")
                    print(f"[INFO] Detected CUDA version: {detected_version}")
                    
                    # Map detected version to closest supported PyTorch CUDA version
                    major_version = float(detected_version.split(".")[0])
                    if major_version < 11.0:
                        cuda_version = "cu113"
                    elif 11.0 <= major_version < 12.0:
                        cuda_version = "cu118"  # CUDA 11.x → use 11.8
                    else:
                        cuda_version = "cu124"  # CUDA 12.x → use 12.4
    except Exception as e:
        has_cuda = False
        logging.warning(f"CUDA not detected: {e}")
        logging.warning("Installing CPU-only version of PyTorch")
        print("[WARNING] CUDA not detected. Installing CPU-only version of PyTorch")
    
    # Install PyTorch with appropriate support
    cmd = [
        sys.executable, "-m", "pip", "install", "-U", 
        "--no-cache-dir",
        "torch", "torchvision", "torchaudio"
    ]
    
    # Add CUDA-specific index URL only if CUDA is detected
    if has_cuda:
        cmd.append(f"--index-url=https://download.pytorch.org/whl/{cuda_version}")
        logging.info(f"Installing PyTorch with {cuda_version} support...")
        print(f"[INFO] Installing PyTorch with {cuda_version} support...")
    else:
        logging.info("Installing CPU-only PyTorch...")
        print("[INFO] Installing CPU-only PyTorch...")
    
    try:
        result = run_subprocess(cmd, check=True)
        
        # Verify the installation
        verify_cmd = [
            sys.executable, "-c", 
            "import torch; print('CUDA available:', torch.cuda.is_available()); "
            "print('PyTorch version:', torch.__version__); "
            "print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'Not available')"
        ]
        verify_result = run_subprocess(verify_cmd, check=False)
        output = verify_result.get("stdout", "")
        
        if has_cuda and "CUDA available: True" in output:
            logging.info("Successfully installed PyTorch with CUDA support")
            print("[INFO] Successfully installed PyTorch with CUDA support")
            logging.info(output.strip())
            print(output.strip())
            return True
        else:
            if has_cuda:
                logging.warning("PyTorch installed but CUDA support is not available")
                print("[WARNING] PyTorch installed but CUDA support is not available")
            else:
                logging.info("Successfully installed CPU-only PyTorch")
                print("[INFO] Successfully installed CPU-only PyTorch")
            logging.info(output.strip())
            print(output.strip())
            return True  # Return True for CPU-only as it's a successful install
    except Exception as e:
        logging.error(f"Error installing PyTorch: {e}")
        print(f"[ERROR] Error installing PyTorch: {e}")
        return False

def download_qwen_model() -> None:
    """
    Download the Qwen2.5-1.5B-Instruct-GGUF model to the 'models' directory.
    """
    project_root = Path(find_project_root())
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / "qwen2.5-1.5b-instruct-q4_k_m.gguf"
    if model_path.exists():
        logging.info(f"Model already exists at {model_path}; skipping download.")
        print(f"[INFO] Qwen model already exists at {model_path}; skipping download.")
        return
    try:
        logging.info("Downloading Qwen2.5-1.5B-Instruct-GGUF model...")
        print("[INFO] Downloading Qwen model...")
        run_subprocess([
            "huggingface-cli", "download",
            "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
            "qwen2.5-1.5b-instruct-q4_k_m.gguf",
            "--local-dir", str(models_dir)
        ], check=True)
        logging.info(f"Model downloaded successfully to {models_dir}")
        print(f"[INFO] Model downloaded successfully to {models_dir}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error downloading Qwen model: {e}")
        print(f"[ERROR] Error downloading Qwen model: {e}")
        logging.error(f"Please download manually and place it into {models_dir}")

def install_spacy_model() -> None:
    """
    Install spaCy's English model.
    """
    logging.info("Installing spaCy English model...")
    print("[INFO] Installing spaCy English model...")
    try:
        run_subprocess([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
        logging.info("spaCy English model installed successfully")
        print("[INFO] spaCy English model installed successfully")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error installing spaCy model: {e}")
        print(f"[ERROR] Error installing spaCy model: {e}")
        logging.error("Please install manually using: python -m spacy download en_core_web_sm")

# -----------------------------------------------------------------------------
# Optional Windows CUDA integration helpers

def copy_cuda_visual_studio_files() -> None:
    """
    For Windows systems, attempt to copy CUDA integration files into Visual Studio directories.
    (This is optional and can be customized as needed.)
    """
    if platform.system() != "Windows":
        return
    logging.info("Setting up CUDA integration for Visual Studio...")
    print("[INFO] (Optional) Setting up CUDA integration for Visual Studio... [Not Implemented]")
    logging.info("CUDA Visual Studio file copying not implemented in this minimal module.")

def create_windows_cuda_installer() -> None:
    """
    Create a Windows batch script to force CUDA installation.
    (Optional helper.)
    """
    if platform.system() != "Windows":
        return
    project_root = Path(find_project_root())
    script_path = project_root / "windows-cuda-installer.bat"
    cuda_path = os.environ.get("CUDA_PATH", r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4")
    with open(script_path, "w") as f:
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
    print(f"[INFO] Created Windows CUDA installer script at: {script_path}")

# -----------------------------------------------------------------------------
# Main setup function

def setup_dependencies() -> None:
    """
    Main function to set up dependencies:
      - Log system information
      - Create project directories (models, logs, data)
      - Optionally set up Windows CUDA integration
      - Install llama-cpp-python (compiling from source if CUDA is detected)
      - Install spaCy English model
      - Download Qwen model
    """
    logging.info("=" * 80)
    logging.info("LlamaSearch Setup".center(80))
    logging.info("=" * 80)
    print("=" * 80)
    print("LlamaSearch Setup".center(80))
    print("=" * 80)
    log_system_info()

    project_root = Path(find_project_root())
    for d in ["models", "logs", "data"]:
        (project_root / d).mkdir(exist_ok=True)
        logging.debug(f"Created directory: {(project_root / d)}")
        print(f"[INFO] Ensured directory exists: {(project_root / d)}")

    if platform.system() == "Windows":
        copy_cuda_visual_studio_files()
        create_windows_cuda_installer()
    
    # Install PyTorch with appropriate acceleration support first
    installed_torch = install_torch_with_cuda()
    if not installed_torch:
        logging.warning("Failed to install PyTorch. Some features may not work properly.")
        print("[WARNING] Failed to install PyTorch. Some features may not work properly.")
    
    if install_llama_cpp_python():
        install_spacy_model()
        download_qwen_model()
        logging.info("=" * 80)
        logging.info("Setup Complete!".center(80))
        print("=" * 80)
        print("Setup Complete!".center(80))
        print("=" * 80)
        config = get_acceleration_config()
        if config.get("type") == "cuda" and check_llama_cpp_cuda_support():
            logging.info("GPU acceleration is enabled for llama-cpp-python")
            print("[INFO] GPU acceleration is enabled for llama-cpp-python")
        elif config.get("type") == "metal":
            logging.info("Metal acceleration is enabled for llama-cpp-python")
            print("[INFO] Metal acceleration is enabled for llama-cpp-python")
        else:
            logging.warning("GPU acceleration is NOT enabled; running in CPU-only mode (significantly slower).")
            print("[WARNING] GPU acceleration is NOT enabled; running in CPU-only mode (significantly slower).")
    else:
        logging.error("=" * 80)
        logging.error("Setup Failed".center(80))
        logging.error("Please check the error messages above.")
        print("=" * 80)
        print("Setup Failed".center(80))
        print("Please check the error messages above.")
        print("=" * 80)
    logging.info(f"\nFull debug logs are available at: {DEBUG_LOG_PATH}")
    print(f"[INFO] Full debug logs are available at: {DEBUG_LOG_PATH}")

if __name__ == "__main__":
    setup_logger()
    setup_dependencies()