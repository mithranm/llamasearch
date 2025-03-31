# llamasearch/utils.py

import os
import json
import logging
import platform
import subprocess
import sys
from datetime import datetime
import numpy as np  # add this import to help detect np.float32 for custom encoder

def find_project_root():
    """Finds the root of the project by looking for `setup.py`."""
    current_dir = os.path.abspath(os.path.dirname(__file__))
    while current_dir != os.path.dirname(current_dir):
        if os.path.exists(os.path.join(current_dir, "setup.py")):
            return current_dir
        current_dir = os.path.dirname(current_dir)
    raise RuntimeError("Could not find project root. Check your structure.")


def setup_logging(name, level=logging.INFO):
    """
    Set up logging to both console and file.
    """
    project_root = find_project_root()
    logs_dir = os.path.join(project_root, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()

    # File handler with daily filename
    today = datetime.now().strftime("%Y-%m-%d")
    log_file = os.path.join(logs_dir, f"{name.split('.')[-1]}_{today}.log")
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(level)

    # Console handler only for WARNING+
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)

    file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_formatter = logging.Formatter("%(levelname)s - %(message)s")

    file_handler.setFormatter(file_formatter)
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


class NumpyEncoder(json.JSONEncoder):
    """
    Custom JSON encoder that gracefully handles numpy types (e.g. float32, int64).
    """
    def default(self, obj):
        # Convert float32 -> float
        if isinstance(obj, np.float32) or isinstance(obj, np.float64):
            return float(obj)
        # Convert int64 -> int
        if isinstance(obj, np.integer):
            return int(obj)
        # Convert array -> list (optional)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def log_query(query, context_chunks, response, debug_info=None):
    """
    Log query, retrieved chunks, and generated response to a JSON file.
    Converting all np.float32 to python float so it won't crash.
    """
    project_root = find_project_root()
    logs_dir = os.path.join(project_root, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(logs_dir, f"query_{timestamp}.json")

    log_data = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "context_chunks": context_chunks,
        "response": response,
    }
    if debug_info is not None:
        log_data["debug_info"] = debug_info

    with open(log_file, "w", encoding="utf-8") as f:
        # Use our custom NumpyEncoder
        json.dump(log_data, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)

    return log_file


def setup_dependencies():
    """Set up LlamaSearch dependencies that require special handling."""
    print("Setting up LlamaSearch dependencies...")
    setup_project_directories()
    install_llama_cpp_python()
    try:
        print("Checking and installing SpaCy and rank-bm25...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "spacy>=3.0.0", "rank-bm25"],
            check=True,
        )
        print("Successfully installed SpaCy and rank-bm25")
        download_spacy_model()
    except Exception as e:
        print(f"Error during SpaCy or rank-bm25 install: {e}")
        print("Please install manually: pip install spacy rank-bm25")
        print("python -m spacy download en_core_web_sm")
    try:
        download_qwen_model()
    except Exception as e:
        print(f"Error downloading model: {e}")
        print("Check your internet or download manually.")
    print("LlamaSearch setup complete!")


def setup_project_directories():
    project_root = find_project_root()
    directories = [
        os.path.join(project_root, "models"),
        os.path.join(project_root, "vector_db"),
        os.path.join(project_root, "logs"),
        os.path.join(project_root, "temp"),
    ]
    for d in directories:
        os.makedirs(d, exist_ok=True)
        print(f"Created directory: {d}")


def install_llama_cpp_python():
    """Install llama-cpp-python with platform checks."""
    print("Installing llama-cpp-python...")
    try:
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            print("Detected Apple Silicon, installing with Metal support.")
            env = os.environ.copy()
            env["CMAKE_ARGS"] = "-DLLAMA_METAL=on"
            subprocess.run(["pip", "install", "--no-cache-dir", "llama-cpp-python"], check=True, env=env)
        else:
            subprocess.run(["pip", "install", "llama-cpp-python"], check=True)
        print("llama-cpp-python installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error installing llama-cpp-python: {e}")
        print("Try installing it manually.")
        raise


def download_qwen_model():
    """Download Qwen2.5-1.5B-Instruct-GGUF model to 'models' dir."""
    print("Downloading Qwen2.5-1.5B-Instruct-GGUF model...")
    project_root = find_project_root()
    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)
    try:
        subprocess.run([
            "huggingface-cli", "download",
            "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
            "qwen2.5-1.5b-instruct-q4_k_m.gguf",
            "--local-dir", models_dir
        ], check=True)
        print("Model downloaded successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading model: {e}")
        print("Please download manually from https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF")
        print(f"And place it into {models_dir}")


def download_spacy_model():
    print("Downloading SpaCy en_core_web_sm...")
    try:
        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
        print("SpaCy model downloaded successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading SpaCy model: {e}")
        print("Install manually: python -m spacy download en_core_web_sm")
        raise
