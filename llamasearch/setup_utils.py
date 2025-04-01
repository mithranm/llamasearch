"""Setup utilities for LlamaSearch installation."""

import os
import platform
import subprocess
import sys
from pathlib import Path


def find_project_root():
    """Finds the root of the project by looking for `setup.py`."""
    current_dir = os.path.abspath(os.path.dirname(__file__))
    while current_dir != os.path.dirname(current_dir):
        if os.path.exists(os.path.join(current_dir, "pyproject.toml")):
            return current_dir
        current_dir = os.path.dirname(current_dir)
    raise RuntimeError("Could not find project root. Check your structure.")


def setup_project_directories():
    """Set up required project directories."""
    project_root = Path(__file__).parent.parent
    dirs = ["models", "logs", "data"]
    for dir_name in dirs:
        (project_root / dir_name).mkdir(exist_ok=True)


def install_llama_cpp_python():
    """Install llama-cpp-python with platform-specific optimizations."""
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        os.environ["CMAKE_ARGS"] = "-DLLAMA_METAL=on"
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--upgrade", "llama-cpp-python"]
        )
    except subprocess.CalledProcessError as e:
        print(f"Error installing llama-cpp-python: {e}")
        print("Please try installing it manually: pip install llama-cpp-python")


def download_qwen_model():
    """Download Qwen2.5-1.5B-Instruct-GGUF model to 'models' dir."""
    print("Downloading Qwen2.5-1.5B-Instruct-GGUF model...")

    # Use consistent project root determination
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
        print(
            "Please download manually from https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF"
        )
        print(f"And place it into {models_dir}")


def setup_dependencies():
    """Main setup function called by the CLI."""
    print("LlamaSearch Setup")
    print("=================")

    setup_project_directories()
    install_llama_cpp_python()

    try:
        download_qwen_model()
    except Exception as e:
        print(f"Error during model download: {e}")
        print(
            "Please check your internet connection and try again or download it yourself."
        )
