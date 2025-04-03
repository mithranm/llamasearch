"""Setup utilities for LlamaSearch installation."""

import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Dict, Tuple

def find_project_root():
    """Finds the root of the project by looking for pyproject.toml."""
    current_dir = os.path.abspath(os.path.dirname(__file__))
    while current_dir != os.path.dirname(current_dir):
        if os.path.exists(os.path.join(current_dir, "pyproject.toml")):
            return current_dir
        current_dir = os.path.dirname(current_dir)
    return os.path.dirname(os.path.abspath(__file__))

def detect_optimal_acceleration() -> str:
    """Detect the optimal acceleration type for the current system."""
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return "metal"
    elif os.environ.get("CUDA_VISIBLE_DEVICES") is not None or os.path.exists("/usr/local/cuda"):
        return "cuda"
    return "cpu"

def get_acceleration_config() -> str:
    """Get the acceleration configuration from environment or auto-detect.
    
    Priority:
    1. LLAMASEARCH_ACCELERATION environment variable
    2. Auto-detection based on system capabilities
    """
    acc_type = os.environ.get("LLAMASEARCH_ACCELERATION", "auto").lower()
    if acc_type == "auto":
        return detect_optimal_acceleration()

    if acc_type not in ["cpu", "cuda", "metal"]:
        print(f"Warning: Unknown acceleration type '{acc_type}', falling back to auto-detection")
        return detect_optimal_acceleration()
    return acc_type

def configure_build_environment() -> Tuple[str, list, Dict[str, str]]:
    """Configure the build environment for llama-cpp-python."""
    acc_type = get_acceleration_config()
    cmake_args = []
    env_vars = {}

    if acc_type == "metal":
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            cmake_args.append("-DLLAMA_METAL=on")
            print("Configuring for Metal acceleration")
        else:
            print("Warning: Metal acceleration requested but not supported, falling back to CPU")
            acc_type = "cpu"

    elif acc_type == "cuda":
        cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
        # If on Linux and no env var, assume /usr/local/cuda if it exists
        if not cuda_home and platform.system() == "Linux":
            if os.path.exists("/usr/local/cuda"):
                cuda_home = "/usr/local/cuda"
        if not cuda_home and platform.system() == "Windows":
            program_files = os.environ.get("ProgramFiles", "C:\\Program Files")
            cuda_paths = [os.path.join(program_files, "NVIDIA GPU Computing Toolkit", "CUDA", ver)
                          for ver in ["v12.3", "v12.2", "v12.1", "v12.0", "v11.8"]]
            for path in cuda_paths:
                if os.path.exists(path):
                    cuda_home = path
                    break

        if cuda_home and os.path.exists(cuda_home):
            # Instead of 'all-major', we rely on the user’s environment or default to sm80
            # If user didn't define CMAKE_CUDA_ARCHITECTURES in env, we set a fallback.
            if "CMAKE_ARGS" not in os.environ:
                # fallback if user hasn't set environment
                cmake_args.append("-DGGML_CUDA=on")
                cmake_args.append("-DCMAKE_CUDA_ARCHITECTURES=80;86")
            # If user already has CMAKE_ARGS with architecture, we won't override it
            nvcc_path = os.path.join(cuda_home, "bin", "nvcc")
            if platform.system() == "Windows":
                nvcc_path += ".exe"

            if os.path.exists(nvcc_path):
                env_vars["CUDACXX"] = nvcc_path
                env_vars["CUDA_PATH"] = cuda_home
                print(f"Configuring for CUDA acceleration using {cuda_home}")
            else:
                print("Warning: CUDA installation found but nvcc not found, falling back to CPU")
                acc_type = "cpu"
        else:
            print("Warning: CUDA acceleration requested but CUDA not found, falling back to CPU")
            acc_type = "cpu"

    # If we appended any cmake_args, set them in env
    if cmake_args:
        # If user already set CMAKE_ARGS, append
        old_cmake_args = os.environ.get("CMAKE_ARGS", "")
        if old_cmake_args:
            merged = old_cmake_args + " " + " ".join(cmake_args)
        else:
            merged = " ".join(cmake_args)
        env_vars["CMAKE_ARGS"] = merged

    return acc_type, cmake_args, env_vars

def setup_project_directories():
    project_root = Path(__file__).parent.parent
    dirs = ["models", "logs", "data"]
    for d in dirs:
        (project_root / d).mkdir(exist_ok=True)

def install_llama_cpp_python():
    """Install llama-cpp-python with configured acceleration support."""
    acc_type, cmake_args, env_vars = configure_build_environment()
    try:
        # Uninstall old version if present
        subprocess.run(
            [sys.executable, "-m", "pip", "uninstall", "-y", "llama-cpp-python"],
            check=False
        )
        # Install with environment
        build_env = os.environ.copy()
        build_env.update(env_vars)

        print(f"Installing llama-cpp-python with {acc_type} acceleration")
        if cmake_args:
            print(f"CMAKE_ARGS: {' '.join(cmake_args)}")
        if env_vars:
            print(f"Environment: {env_vars}")

        cmd = [
            sys.executable, "-m", "pip", "install",
            "--no-cache-dir",
            "--verbose",
            "llama-cpp-python"
        ]
        subprocess.run(cmd, env=build_env, check=True)
        print(f"Successfully installed llama-cpp-python with {acc_type} acceleration")

    except subprocess.CalledProcessError as e:
        print(f"Error installing llama-cpp-python: {e}")
        print("\nTroubleshooting tips:")
        if acc_type == "cuda":
            print("- Ensure NVIDIA CUDA Toolkit is installed")
            print("- Check if nvcc is in your PATH")
            print("- Try setting CUDA_HOME environment variable")
            print("- Or set LLAMASEARCH_ACCELERATION=cpu to force CPU mode")
        elif acc_type == "metal":
            print("- Ensure you're on Apple Silicon Mac")
            print("- Ensure Xcode Command Line Tools are installed")
            print("- Or set LLAMASEARCH_ACCELERATION=cpu to force CPU mode")
        print("\nFalling back to CPU version...")

        try:
            os.environ["LLAMASEARCH_ACCELERATION"] = "cpu"
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--no-cache-dir", "llama-cpp-python"],
                check=True
            )
            print("Successfully installed CPU version of llama-cpp-python")
        except subprocess.CalledProcessError as e2:
            print(f"Error installing CPU version: {e2}")
            raise

def download_qwen_model():
    """Download Qwen2.5-1.5B-Instruct-GGUF model to 'models' dir."""
    print("Downloading Qwen2.5-1.5B-Instruct-GGUF model...")
    project_root = Path(__file__).parent.parent
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)
    try:
        subprocess.run(
            [
                "huggingface-cli",
                "download",
                "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
                "qwen2.5-1.5b-instruct-q4_k_m.gguf",
                "--local-dir",
                str(models_dir),
            ],
            check=True,
        )
        print(f"Model downloaded successfully to {models_dir}!")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading model: {e}")
        print("Please download manually from https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF")
        print(f"And place it into {models_dir}")

def setup_dependencies():
    print("LlamaSearch Setup")
    print("=================")
    setup_project_directories()
    install_llama_cpp_python()
    try:
        download_qwen_model()
    except Exception as e:
        print(f"Error during model download: {e}")
        print("Please check your internet connection or do a manual download.")
