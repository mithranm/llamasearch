#!/usr/bin/env python3
"""
Setup script for LlamaSearch with llama-cpp-python using Qwen 2.5 1.5B model.
"""

import os
import subprocess
import platform
from setuptools import setup, Distribution
from setuptools.command.develop import develop
from setuptools.command.install import install

class BinaryDistribution(Distribution):
    """Distribution which always forces a binary package with proper platform tags."""
    def has_ext_modules(self):
        return True

def setup_llamasearch():
    """Set up LlamaSearch post-installation."""
    print("LlamaSearch Setup")
    print("=================")
    
    # Set up project directories
    setup_project_directories()
    
    # Install llama-cpp-python
    install_llama_cpp_python()
    
    # Download model
    project_root = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(project_root, "models")
    try:
        download_qwen_model(models_dir)
    except Exception as e:
        print(f"Error during model download: {e}")
        print("Please check your internet connection and try again or download it yourself.")

class PostDevelopCommand(develop):
    """Post-installation for development mode."""
    def run(self):
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            os.environ["CMAKE_ARGS"] = "-DLLAMA_METAL=on"
        develop.run(self)
        setup_llamasearch()

class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            os.environ["CMAKE_ARGS"] = "-DLLAMA_METAL=on"
        install.run(self)
        setup_llamasearch()

def setup_project_directories():
    """Set up project directories."""
    project_root = os.path.dirname(os.path.abspath(__file__))
    directories = [
        os.path.join(project_root, "models"),
        os.path.join(project_root, "vector_db"),
        os.path.join(project_root, "logs"),
        os.path.join(project_root, "temp"),
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def install_llama_cpp_python():
    """Install llama-cpp-python with appropriate settings for the current platform."""
    print("Installing llama-cpp-python...")
    try:
        subprocess.run(
            ["pip", "install", "--no-cache-dir", "llama-cpp-python"],
            check=True,
            env=os.environ.copy(),
        )
        print("Successfully installed llama-cpp-python")
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            print("Metal support enabled for Apple Silicon")
    except subprocess.CalledProcessError as e:
        print(f"Error installing llama-cpp-python: {e}")
        print("Please try installing it manually:")
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            print("CMAKE_ARGS='-DLLAMA_METAL=on' pip install llama-cpp-python")
        else:
            print("pip install llama-cpp-python")
        raise

def download_qwen_model(models_dir):
    """Download the Qwen 2.5 1.5B model with Q4 quantization."""
    print("Downloading Qwen2.5-1.5B-Instruct-GGUF model...")
    try:
        os.makedirs(models_dir, exist_ok=True)
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

setup(
    distclass=BinaryDistribution,
    cmdclass={
        'develop': PostDevelopCommand,
        'install': PostInstallCommand,
    }
)
