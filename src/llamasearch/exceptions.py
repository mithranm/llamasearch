# src/llamasearch/exceptions.py

class ModelNotFoundError(Exception):
    """Custom exception raised when a required model is not found locally."""
    pass

class SetupError(Exception):
    """Custom exception for errors during the setup process."""
    pass