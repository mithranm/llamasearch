#!/usr/bin/env python3
"""
Setup script for LlamaSearch with llama-cpp-python using Qwen 2.5 1.5B model.
"""

import os
import sys
import subprocess
import platform
from setuptools import setup, find_packages


def install_llama_cpp_python():
    """Install llama-cpp-python with appropriate settings for the current platform."""
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        # For Apple Silicon (M1/M2/M3)
        print("Installing llama-cpp-python with Metal support for Apple Silicon...")
        try:
            # First try to install prebuilt wheel
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "llama-cpp-python",
                    "--upgrade",
                    "--force-reinstall",
                    "--no-cache-dir",
                    "--extra-index-url",
                    "https://abetlen.github.io/llama-cpp-python/whl/cpu",
                ]
            )

            # Then reinstall with Metal support
            subprocess.run(
                'CMAKE_ARGS="-DLLAMA_METAL=on" pip install --force-reinstall --no-cache-dir llama-cpp-python',
                shell=True,
                check=True,
            )
            print("Successfully installed llama-cpp-python with Metal support")
        except subprocess.CalledProcessError as e:
            print(f"Warning: Could not install with Metal support: {e}")
            print("Falling back to CPU-only installation...")
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "llama-cpp-python",
                    "--no-cache-dir",
                ],
                check=True,
            )

    elif platform.system() == "Windows":
        # For Windows
        print("Installing llama-cpp-python for Windows...")
        # Windows often works better with pre-built binaries
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "llama-cpp-python",
                "--no-cache-dir",
                "--extra-index-url",
                "https://abetlen.github.io/llama-cpp-python/whl/cpu",
            ],
            check=True,
        )

    elif platform.system() == "Linux":
        # For Linux, try to use OpenBLAS
        print("Installing llama-cpp-python with OpenBLAS for Linux...")
        try:
            subprocess.run(
                'CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS" pip install --no-cache-dir llama-cpp-python',
                shell=True,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"Warning: Could not install with OpenBLAS: {e}")
            print("Falling back to default installation...")
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "llama-cpp-python",
                    "--no-cache-dir",
                ],
                check=True,
            )

    else:
        # Default fallback
        print("Installing llama-cpp-python (default)...")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "llama-cpp-python",
                "--no-cache-dir",
            ],
            check=True,
        )


def download_qwen_model(models_dir):
    """Download the Qwen 2.5 1.5B model with Q4 quantization."""
    print("Downloading Qwen2.5-1.5B-Instruct-GGUF model...")
    try:
        # Make sure the models directory exists
        os.makedirs(models_dir, exist_ok=True)

        # Using huggingface-cli to download the model
        subprocess.run(
            [
                "huggingface-cli",
                "download",
                "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
                "qwen2.5-1.5b-instruct-q4_k_m.gguf",
                "--local-dir",
                models_dir,
            ],
            check=True,
        )
        print("Model downloaded successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading model: {e}")
        print(
            "Please download the model manually from: https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF"
        )
        print(f"And place it in the models directory: {models_dir}")


def setup_project_directories():
    """Set up project directories."""
    # Get project root directory (where setup.py is located)
    project_root = os.path.dirname(os.path.abspath(__file__))

    # Define directories to create
    directories = [
        os.path.join(project_root, "models"),
        os.path.join(project_root, "vector_db"),
        os.path.join(project_root, "logs"),
        os.path.join(project_root, "temp"),
    ]

    # Create directories if they don't exist
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")


# Setup configuration for setuptools
setup(
    name="llamasearch",
    version="0.1.0",
    description="RAG-based search application with llama-cpp-python",
    author="Mithran Mohanraj, Leo Angulo, Georgi Zahariev, Robin Hwang, Dhanush Reddy Kandukuri, Shaheda Tawakalyar, Malek Bashagha",
    author_email="mithran.mohanraj@gmail.com",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch",
        "transformers",
        "sentence-transformers",
        "markdown",
        "beautifulsoup4",
        "colorama",
        "scikit-learn",
        "numpy",
        "huggingface_hub",
        "gradio",
        "requests",
        "pydantic",
    ],
    entry_points={
        "console_scripts": [
            "llamasearch=llamasearch.__main__:main",
        ],
    },
    classifiers=[
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)

# Only run setup steps when this script is executed directly
if __name__ == "__main__":
    print("LlamaSearch Setup")
    print("=================")

    # Set up project directories
    setup_project_directories()

    # Install or update llama-cpp-python
    install_llama_cpp_python()

    project_root = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(project_root, "models")
    try:
        download_qwen_model(models_dir)
    except Exception as e:
        print(f"Error during model download: {e}")
        print(
            "Please check your internet connection and try again or download it yourself."
        )

    print("\nSetup completed successfully!")
    print("To use LlamaSearch:")
    print("  1. Run with: python -m llamasearch.core.llm --interactive")
    print("  2. Or import in your code: from llamasearch.core.llm import OptimizedLLM")
