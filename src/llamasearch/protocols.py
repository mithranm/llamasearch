# src/llamasearch/protocols.py
from typing import Any, Protocol, Tuple, runtime_checkable


@runtime_checkable
class ModelInfo(Protocol):
    """Protocol defining the expected attributes for model metadata."""

    @property
    def model_id(self) -> str:
        """Unique identifier for the model."""
        ...

    @property
    def model_engine(self) -> str:
        """Identifier for the backend engine (e.g., 'onnx_teapot', 'transformers')."""
        ...

    @property
    def description(self) -> str:
        """A brief description of the model."""
        ...

    @property
    def context_length(self) -> int:
        """The maximum context length the model supports."""
        ...


@runtime_checkable
class LLM(Protocol):
    """Protocol defining the interface for a Language Model."""

    @property
    def model_info(self) -> ModelInfo:
        """Provides metadata about the model conforming to the ModelInfo protocol."""
        ...

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repeat_penalty: float = 1.0,
        **kwargs: Any,  # Allow flexible keyword arguments
    ) -> Tuple[str, Any]:
        """
        Generates text based on a prompt.

        Args:
            prompt: The input text prompt.
            max_tokens: Maximum number of new tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling probability.
            repeat_penalty: Penalty for repeating tokens.
            **kwargs: Additional engine-specific generation parameters.

        Returns:
            A tuple containing:
            - The generated text (str).
            - Engine-specific metadata or raw output (Any).
        """
        ...

    def load(self) -> bool:
        """
        Loads the model resources (if not already loaded).
        Returns True if loading was successful or model is already loaded, False otherwise.
        """
        ...

    def unload(self) -> None:
        """
        Unloads the model resources and performs cleanup.
        Should release memory (including GPU memory if applicable).
        """
        ...
