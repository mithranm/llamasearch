"""
LlamaSearch LLM implementation using llama-cpp-python with Qwen 2.5 1.5B model
Enhanced with intent analysis and one-shot learning for improved responses
"""

import os
import argparse
import time
import gc
import colorama
from colorama import Fore, Style
import platform
import re
from typing import List, Dict, Union, Tuple, Optional

from llama_cpp import Llama

from ..utils import find_project_root, setup_logging, log_query

# Initialize colorama for cross-platform color support
colorama.init()

# Configure logging
logger = setup_logging(__name__)


class OptimizedLLM:
    """
    Optimized LLM implementation using llama-cpp-python.
    Uses Qwen 2.5 1.5B with Q4 quantization.
    Enhanced with query intent analysis and one-shot learning.
    """

    def __init__(
        self,
        model_name: str = "qwen2.5-1.5b-instruct-q4_k_m",
        persist: bool = False,
        verbose: bool = True,
        context_length: int = 2048,
        n_results: int = 5,
        custom_model_path: Optional[str] = None,
    ):
        """
        Initialize the OptimizedLLM with efficient settings for limited hardware.

        Args:
            model_name: Name of the model to use (default: "qwen2.5-1.5b-instruct-q4_k_m")
            persist: Whether to persist the vector database
            verbose: Whether to print verbose output
            context_length: Maximum context length for the model
            n_results: Number of results to return from vector search
            custom_model_path: Path to a custom model file
        """
        self.verbose = verbose
        self.persist = persist
        self.context_length = context_length
        self.n_results = n_results
        self.model_name = model_name
        self.custom_model_path = custom_model_path

        # Create model directory if needed
        project_root = find_project_root()
        self.models_dir = os.path.join(project_root, "models")
        os.makedirs(self.models_dir, exist_ok=True)

        # Determine model path
        if custom_model_path:
            self.model_path = custom_model_path
            logger.info(f"Using custom model at: {self.model_path}")
        else:
            self.model_path = os.path.join(self.models_dir, f"{model_name}.gguf")
            logger.info(f"Using model: {model_name} ({self.model_path})")

        # Check if model exists
        if not os.path.exists(self.model_path):
            logger.warning(f"Model not found at {self.model_path}")
            logger.info("Please download the model or provide a valid model path.")
            logger.info(
                "You can download the model using: python setup.py --download-model"
            )

        # Set up vector database with optimized settings
        from .vectordb import VectorDB
        from .embedding import Embedder

        # Create the embedder with minimum settings
        self.embedder = Embedder(
            device="cpu",  # Force CPU for compatibility
            max_length=512,  # Embedding size
            batch_size=1,  # Small batch size for memory efficiency
        )

        # Set up storage directory
        self.storage_dir = os.path.join(project_root, "vector_db")

        # Initialize the VectorDB with memory-efficient settings
        self.vectordb = VectorDB(
            embedder=self.embedder,
            persist=persist,
            chunk_size=250,  # Smaller chunks for memory efficiency
            text_embedding_size=512,
            chunk_overlap=50,  # Smaller overlap for memory efficiency
            min_chunk_size=50,
            batch_size=1,  # Process one at a time to save memory
            similarity_threshold=0.25,  # Lower threshold to capture more potentially relevant chunks
            storage_dir=self.storage_dir,
            use_pca=False,
        )

        # Process any documents in the temp directory
        self._process_temp_documents()

        # Initialize model instance to None, will load on demand
        self.llm_instance = None

        logger.info(f"Initialized OptimizedLLM with model: {self.model_name}")
        logger.info(f"Model path: {self.model_path}")

    def _process_temp_documents(self):
        """Process any documents in the project root's temp directory."""
        project_root = find_project_root()
        temp_dir = os.path.join(project_root, "temp")

        if not os.path.exists(temp_dir):
            logger.info(f"Temp directory not found at {temp_dir}")
            return

        logger.info(f"Processing documents from temp directory: {temp_dir}")
        for filename in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, filename)
            if os.path.isfile(file_path):
                try:
                    logger.info(f"Processing file: {filename}")
                    self.vectordb.add_document(file_path)
                except Exception as e:
                    logger.error(f"Error processing {filename}: {str(e)}")

    def _get_llm(self):
        """
        Get or create the LLM instance with appropriate settings.
        Lazily loads the model when needed.
        """
        if self.llm_instance is None:
            logger.info(f"Loading LLM model from {self.model_path}")

            # Determine optimal number of threads based on CPU
            n_threads = min(os.cpu_count() or 4, 8)  # Use at most 8 threads

            # Determine if we can use GPU acceleration
            n_gpu_layers = 0  # Default to CPU only
            if platform.system() == "Darwin" and platform.processor() == "arm":
                # For Apple Silicon, use Metal
                n_gpu_layers = -1  # Use all layers
                logger.info("Apple Silicon detected, using Metal for acceleration")

            # Load the model
            try:
                self.llm_instance = Llama(
                    model_path=self.model_path,
                    n_ctx=self.context_length,
                    n_threads=n_threads,
                    n_gpu_layers=n_gpu_layers,
                    verbose=self.verbose,
                )
                logger.info(
                    f"Model loaded successfully with context length {self.context_length}"
                )
            except Exception as e:
                logger.error(f"Error loading model: {str(e)}")
                raise RuntimeError(f"Failed to load model: {str(e)}")

        return self.llm_instance

    def _format_retrieved_chunks(self, chunks) -> str:
        """
        Format retrieved chunks for display in console.

        Args:
            chunks: List of chunk dictionaries

        Returns:
            Formatted string for display
        """
        if not chunks:
            return f"{Fore.YELLOW}No relevant chunks found in the knowledge base{Style.RESET_ALL}"

        result = [
            f"{Fore.CYAN}Retrieved {len(chunks)} relevant chunks:{Style.RESET_ALL}\n"
        ]

        for i, chunk in enumerate(chunks):
            similarity = chunk.get("similarity", 0) * 100
            chunk_type = chunk.get("type", "unknown")

            # Add chunk header with similarity score
            result.append(
                f"{Fore.GREEN}Chunk {i+1} ({chunk_type}) - {similarity:.2f}% match:{Style.RESET_ALL}"
            )

            # Add metadata if available
            source = chunk.get("metadata", {}).get("source", "unknown")
            if source:
                result.append(f"{Fore.YELLOW}Source: {source}{Style.RESET_ALL}")

            # Add chunk text with indentation
            text = chunk.get("text", "").strip()
            text_lines = text.split("\n")
            indented_text = "\n".join(f"  {line}" for line in text_lines)
            result.append(f"{indented_text}\n")

        return "\n".join(result)

    def _analyze_query_intent(self, query_text):
        """
        Analyze query to identify multiple intents and information needs.

        Args:
            query_text: The user's query

        Returns:
            dict: Contains intent analysis with keys like:
                - has_greeting: Whether query contains a greeting
                - information_request: Extracted information request if any
                - requires_rag: Whether RAG retrieval is needed
        """
        # Initialize intent analysis
        intent = {
            "has_greeting": False,
            "information_request": None,
            "requires_rag": False,
        }

        # Detect greeting component (looser matching with word boundaries)
        greeting_patterns = [
            r"\bhello\b",
            r"\bhi\b",
            r"\bhey\b",
            r"\bgreetings\b",
            r"\bgood\s(morning|afternoon|evening|day)\b",
            r"\bwhat\'s\sup\b",
            r"\bhowdy\b",
        ]

        intent["has_greeting"] = any(
            re.search(pattern, query_text.lower()) for pattern in greeting_patterns
        )

        # Extract information request using heuristics
        info_request_patterns = [
            r"tell\s+me\s+about\s+(.+)",
            r"what\s+is\s+(.+)",
            r"how\s+(?:do|does|to|can|would|could)\s+(.+)",
            r"explain\s+(.+)",
            r"describe\s+(.+)",
            r"show\s+me\s+(.+)",
            r"find\s+(.+)",
            r"search\s+for\s+(.+)",
        ]

        for pattern in info_request_patterns:
            match = re.search(pattern, query_text.lower())
            if match:
                intent["information_request"] = match.group(1).strip()
                intent["requires_rag"] = True
                break

        # If no pattern matched but query is longer than typical greeting,
        # assume it might contain an information need
        if not intent["information_request"] and len(query_text.split()) > 3:
            # If it's just a greeting with pleasantries, don't use RAG
            if intent["has_greeting"] and all(
                word.lower()
                in [
                    "how",
                    "are",
                    "you",
                    "doing",
                    "today",
                    "hope",
                    "well",
                    "fine",
                    "nice",
                    "meet",
                    "pleasure",
                    "glad",
                    "happy",
                    "see",
                    "chat",
                    "talk",
                ]
                for word in query_text.split()
                if word.lower()
                not in [
                    "hello",
                    "hi",
                    "hey",
                    "greetings",
                    "howdy",
                    "good",
                    "morning",
                    "afternoon",
                    "evening",
                    "day",
                ]
            ):
                intent["requires_rag"] = False
            else:
                intent["information_request"] = query_text
                intent["requires_rag"] = True

        logger.info(f"Query intent analysis: {intent}")
        return intent

    def _create_one_shot_rag_prompt(self, context=""):
        """
        Create a one-shot RAG example prompt to teach the model how to use context.

        Args:
            context: Optional real context to reference in size calculation

        Returns:
            str: One-shot example prompt
        """
        # Calculate available space: limit example to ~15% of context
        context_tokens_estimate = len(context.split()) if context else 0
        # max_example_tokens = max(100, min(300, int(self.context_length * 0.15)))

        # Scale example size based on available space
        if context_tokens_estimate > self.context_length * 0.7:
            # Very large context, use minimal example
            example_context = "The Python programming language was created by Guido van Rossum and released in 1991."
            example_query = "When was Python created?"
            example_response = "According to the provided context, Python was created by Guido van Rossum and released in 1991."
        else:
            # Normal case, use more detailed example
            example_context = (
                "Document 1:\n"
                "The Python programming language was created by Guido van Rossum and released in 1991. "
                "Python features a dynamic type system, automatic memory management, and supports multiple programming paradigms.\n\n"
                "Document 2:\n"
                "JavaScript was created by Brendan Eich in 1995 while working at Netscape Communications Corporation."
            )
            example_query = "When was Python created and what are its features?"
            example_response = (
                "According to the provided context, Python was created by Guido van Rossum and released in 1991. "
                "Python features a dynamic type system, automatic memory management, and supports multiple programming paradigms. "
                "The context doesn't provide more specific information about Python's features."
            )

        # Format the one-shot example with ChatML format
        one_shot = (
            f"<|im_start|>user\nContext:\n{example_context}\n\nQuestion: {example_query}<|im_end|>\n"
            f"<|im_start|>assistant\n{example_response}<|im_end|>\n"
        )

        return one_shot

    def _create_one_shot_greeting_prompt(self):
        """
        Create a one-shot greeting example prompt.

        Returns:
            str: One-shot example prompt for greetings
        """
        example_query = "Hello there! How are you today?"
        example_response = "Hello! I'm doing well, thank you for asking. I'm Qwen, an AI assistant here to help you. What can I do for you today?"

        # Format the one-shot example with ChatML format
        one_shot = (
            f"<|im_start|>user\n{example_query}<|im_end|>\n"
            f"<|im_start|>assistant\n{example_response}<|im_end|>\n"
        )

        return one_shot

    def _create_one_shot_mixed_prompt(self):
        """
        Create a one-shot mixed greeting+question example prompt.

        Returns:
            str: One-shot example prompt for mixed greeting and question
        """
        example_query = "Hi there! Could you tell me about large language models?"
        example_response = (
            "Hello! I'd be happy to tell you about large language models. "
            "Large language models (LLMs) are advanced AI systems trained on vast amounts of text data. "
            "They can generate human-like text, translate languages, write different kinds of creative content, and answer questions in an informative way."
        )

        # Format the one-shot example with ChatML format
        one_shot = (
            f"<|im_start|>user\n{example_query}<|im_end|>\n"
            f"<|im_start|>assistant\n{example_response}<|im_end|>\n"
        )

        return one_shot

    def _build_prompt(self, query_text, query_intent, context=""):
        """
        Build a full prompt with appropriate one-shot examples based on query intent.

        Args:
            query_text: The user's query
            query_intent: The analyzed query intent
            context: Optional context from vector search

        Returns:
            str: Formatted prompt for the model
        """
        # Create appropriate system prompt based on query intent
        if query_intent["has_greeting"] and not query_intent["requires_rag"]:
            # Pure greeting
            system_content = (
                "You are Qwen, a friendly and helpful assistant created by Alibaba Cloud. "
                "You are having a casual conversation and should respond in a warm, helpful manner. "
                "Keep your responses natural and friendly."
            )
            one_shot = self._create_one_shot_greeting_prompt()
        elif query_intent["has_greeting"] and query_intent["requires_rag"]:
            # Mixed greeting and information request
            system_content = (
                "You are Qwen, a helpful assistant created by Alibaba Cloud. "
                "You respond to greetings in a friendly way and also provide helpful information. "
                "When answering questions, refer to the provided context if relevant. "
                "If the context doesn't contain the answer, briefly say so. "
                "Be concise and helpful."
            )
            one_shot = self._create_one_shot_mixed_prompt()
        else:
            # Pure information request
            system_content = (
                "You are Qwen, a helpful assistant created by Alibaba Cloud. "
                "Answer questions based on the provided context. "
                "Be direct, clear, and concise in your responses. "
                "If the context doesn't contain relevant information, briefly say so. "
                "Avoid making up answers when you don't know. "
                "Respond in a helpful, informative manner."
            )
            one_shot = self._create_one_shot_rag_prompt(context)

        # Format the full prompt with ChatML template
        prompt = f"<|im_start|>system\n{system_content}<|im_end|>\n"

        # Add one-shot example
        prompt += one_shot

        # Start the actual user query
        prompt += "<|im_start|>user\n"

        # Add context if needed
        if query_intent["requires_rag"] and context:
            prompt += f"Context:\n{context}\n\n"

        # Add the actual query
        prompt += f"Question: {query_text}<|im_end|>\n<|im_start|>assistant\n"

        return prompt

    def query(
        self,
        query_text: str,
        context: str = None,
        show_retrieved_chunks: bool = True,
        debug_mode: bool = False,
    ) -> Union[str, Tuple[str, Dict]]:
        """
        Run a query through the system, retrieving context and then generating a response.

        Args:
            query_text: The user's query
            context: Optional context to use instead of retrieving from vectordb
            show_retrieved_chunks: Whether to display retrieved chunks
            debug_mode: Whether to return debug information

        Returns:
            Generated response or tuple of (response, debug_info) if debug_mode is True
        """
        # Log the query
        logger.info(f"Query: {query_text}")
        start_time = time.time()

        # Initialize variables
        debug_info = {}
        retrieved_chunks_display = ""
        context_chunks = []

        # Analyze query intent
        query_intent = self._analyze_query_intent(query_text)
        debug_info["query_intent"] = query_intent

        # Get context from vector database if needed and not provided
        if not context and query_intent["requires_rag"]:
            logger.info("Retrieving context from vector database...")
            try:
                if debug_mode:
                    debug_info["original_query"] = query_text
                    # Use the information request part if extracted
                    search_query = query_intent["information_request"] or query_text
                    context, context_debug_info = self.vectordb.get_context_for_query(
                        search_query, n_results=self.n_results, debug_mode=True
                    )
                    debug_info.update(context_debug_info)
                    context_chunks = context_debug_info.get("chunks", [])
                else:
                    # Use the information request part if extracted
                    search_query = query_intent["information_request"] or query_text
                    context, context_chunks = self.vectordb.get_context_for_query(
                        search_query, n_results=self.n_results, return_chunks=True
                    )

                retrieval_time = time.time() - start_time
                logger.info(f"Retrieved context in {retrieval_time:.2f}s")
                debug_info["retrieval_time"] = retrieval_time

                # Create a display of the retrieved chunks for the console
                if show_retrieved_chunks and context_chunks:
                    retrieved_chunks_display = self._format_retrieved_chunks(
                        context_chunks
                    )

            except Exception as e:
                logger.error(f"Error retrieving context: {str(e)}")
                context = ""
                debug_info["retrieval_error"] = str(e)

        # Build the prompt with one-shot examples
        prompt = self._build_prompt(query_text, query_intent, context)
        debug_info["prompt_length"] = len(prompt.split())

        # Log prompt for debugging (if verbose)
        if self.verbose:
            logger.debug(f"Prompt: {prompt}")

        # Get the LLM instance
        llm = self._get_llm()

        logger.info("Generating response...")
        generation_start = time.time()

        try:
            # Adjust parameters based on the query intent
            if query_intent["has_greeting"] and not query_intent["requires_rag"]:
                # Pure greeting - Use high temperature for natural conversation
                output = llm(
                    prompt,
                    max_tokens=256,  # Shorter responses for greetings
                    temperature=0.7,  # More creative
                    top_p=0.9,  # Allow more variety
                    repeat_penalty=1.1,  # Light penalty
                    frequency_penalty=0.1,  # Light penalties for conversational flow
                    presence_penalty=0.1,
                    stop=[
                        "<|im_end|>",
                        "<|endoftext|>",
                        "<|im_start|>",
                    ],  # Precise stop tokens
                    echo=False,
                )
            elif query_intent["has_greeting"] and query_intent["requires_rag"]:
                # Mixed greeting and information - Balanced parameters
                output = llm(
                    prompt,
                    max_tokens=768,  # Medium length responses
                    temperature=0.6,  # Balanced creativity and precision
                    top_p=0.9,
                    repeat_penalty=1.15,  # Moderate penalty
                    frequency_penalty=0.15,
                    presence_penalty=0.15,
                    stop=["<|im_end|>", "<|endoftext|>", "<|im_start|>"],
                    echo=False,
                )
            else:
                # Pure information - Focus on accuracy
                output = llm(
                    prompt,
                    max_tokens=1024,  # Longer responses for information
                    temperature=0.5,  # More deterministic
                    top_p=0.85,
                    repeat_penalty=1.2,  # Higher penalty to prevent repetition
                    frequency_penalty=0.25,  # Moderate penalties to prevent repetition
                    presence_penalty=0.25,
                    stop=[
                        "<|im_end|>",
                        "<|endoftext|>",
                        "<|im_start|>",
                        "\n\n\n",
                    ],  # Clear stop conditions
                    echo=False,
                )

            # Extract the response text
            response = output["choices"][0]["text"].strip()

        except Exception as e:
            logger.error(f"Error during generation: {str(e)}")
            response = f"Error generating response: {str(e)}"
            debug_info["generation_error"] = str(e)

        generation_time = time.time() - generation_start
        total_time = time.time() - start_time

        logger.info(f"Generated response in {generation_time:.2f}s")
        logger.info(f"Total query time: {total_time:.2f}s")

        debug_info["generation_time"] = generation_time
        debug_info["total_time"] = total_time

        # Log the complete interaction
        log_file = log_query(
            query=query_text,
            context_chunks=context_chunks if "context_chunks" in locals() else [],
            response=response,
            debug_info=debug_info,
        )

        logger.info(f"Query log saved to {log_file}")

        if debug_mode:
            return response, debug_info, retrieved_chunks_display
        else:
            return response, retrieved_chunks_display

    def chat(
        self,
        conversation_history: List[Dict[str, str]],
        debug_mode: bool = False,
        show_retrieved_chunks: bool = True,
    ) -> Union[str, Tuple[str, Dict]]:
        """
        Chat with the model using conversation history.

        Args:
            conversation_history: List of conversation turns with 'role' and 'content'
            debug_mode: Whether to return debug information
            show_retrieved_chunks: Whether to show retrieved chunks

        Returns:
            Generated response or tuple of (response, debug_info) if debug_mode is True
        """
        # Format the last user message as the query for retrieval
        last_user_message = ""
        for message in reversed(conversation_history):
            if message.get("role") == "user" or message.get("role") == "human":
                last_user_message = message["content"]
                break

        if not last_user_message:
            return "No user message found in conversation history."

        # Get context and query the model
        return self.query(
            last_user_message,
            debug_mode=debug_mode,
            show_retrieved_chunks=show_retrieved_chunks,
        )

    def add_document(self, file_path: str) -> int:
        """
        Add a document to the vector database.

        Args:
            file_path: Path to the document file

        Returns:
            Number of chunks added
        """
        logger.info(f"Adding document: {file_path}")
        start_time = time.time()

        try:
            chunk_count = self.vectordb.add_document(file_path)
            logger.info(
                f"Added {chunk_count} chunks in {time.time() - start_time:.2f}s"
            )
            return chunk_count
        except Exception as e:
            logger.error(f"Error adding document: {str(e)}")
            raise

    def close(self):
        """Clean up resources."""
        try:
            # Close vectordb
            self.vectordb.close()

            # Close LLM instance
            self.llm_instance = None

            # Force garbage collection
            gc.collect()

            logger.info("Resources cleaned up")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")


def main():
    """Command-line interface for OptimizedLLM."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="LlamaSearch - RAG with llama-cpp-python and Qwen 2.5 1.5B",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--document",
        type=str,
        default=None,
        help="Path to markdown document files to process",
    )
    parser.add_argument(
        "--query", type=str, default=None, help="Query to run against the model"
    )
    parser.add_argument(
        "--interactive", action="store_true", help="Run in interactive mode"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode to show chunks used for response",
    )
    parser.add_argument(
        "--persist", action="store_true", help="Don't clear the vector database"
    )
    parser.add_argument(
        "--custom-model", type=str, default=None, help="Path to a custom model file"
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=2048,
        help="Context window size for the model",
    )
    parser.add_argument(
        "--hide-chunks", action="store_true", help="Hide retrieved chunks in the output"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()

    # Initialize the LLM
    llm = OptimizedLLM(
        persist=args.persist,
        verbose=args.verbose,
        context_length=args.context_length,
        custom_model_path=args.custom_model,
    )

    # Process document if specified
    if args.document:
        if os.path.isfile(args.document):
            try:
                chunk_count = llm.add_document(args.document)
                print(
                    f"{Fore.GREEN}Added {chunk_count} chunks from {args.document}{Style.RESET_ALL}"
                )
            except Exception as e:
                print(f"{Fore.RED}Error adding document: {str(e)}{Style.RESET_ALL}")
        elif os.path.isdir(args.document):
            print(
                f"{Fore.CYAN}Processing documents from directory: {args.document}{Style.RESET_ALL}"
            )

            markdown_files = [
                os.path.join(args.document, f)
                for f in os.listdir(args.document)
                if f.endswith((".md", ".txt"))
            ]

            if not markdown_files:
                print(
                    f"{Fore.YELLOW}No markdown or text files found in directory: {args.document}{Style.RESET_ALL}"
                )
            else:
                total_chunks = 0
                start_time = time.time()

                for file_path in markdown_files:
                    print(f"{Fore.CYAN}Processing file: {file_path}{Style.RESET_ALL}")
                    try:
                        chunk_count = llm.add_document(file_path)
                        total_chunks += chunk_count
                        print(
                            f"{Fore.GREEN}Added {chunk_count} chunks from {file_path}{Style.RESET_ALL}"
                        )
                    except Exception as e:
                        print(
                            f"{Fore.RED}Error processing file {file_path}: {e}{Style.RESET_ALL}"
                        )

                print(
                    f"\n{Fore.GREEN}Added {total_chunks} total chunks in {time.time() - start_time:.2f}s{Style.RESET_ALL}"
                )
        else:
            print(
                f"{Fore.RED}Document path not found: {args.document}{Style.RESET_ALL}"
            )

    # Run query if specified
    if args.query:
        show_chunks = not args.hide_chunks
        print(f"{Fore.CYAN}Query: {args.query}{Style.RESET_ALL}")

        start_time = time.time()

        if args.debug:
            response, debug_info, retrieved_chunks = llm.query(
                args.query, debug_mode=True, show_retrieved_chunks=show_chunks
            )

            # Display retrieved chunks first
            if show_chunks and retrieved_chunks:
                print(f"\n{retrieved_chunks}")

            # Display the response
            print(f"\n{Fore.CYAN}Response:{Style.RESET_ALL}")
            print(response)

            print(
                f"\n{Fore.YELLOW}Query completed in {time.time() - start_time:.2f}s{Style.RESET_ALL}"
            )
            print(
                f"{Fore.YELLOW}Detailed logs saved to the logs directory{Style.RESET_ALL}"
            )
        else:
            response, retrieved_chunks = llm.query(
                args.query, show_retrieved_chunks=show_chunks
            )

            # Display retrieved chunks first
            if show_chunks and retrieved_chunks:
                print(f"\n{retrieved_chunks}")

            # Display the response
            print(f"\n{Fore.CYAN}Response:{Style.RESET_ALL}")
            print(response)

            print(
                f"\n{Fore.YELLOW}Query completed in {time.time() - start_time:.2f}s{Style.RESET_ALL}"
            )
            print(
                f"{Fore.YELLOW}Detailed logs saved to the logs directory{Style.RESET_ALL}"
            )

    # Interactive mode
    if args.interactive:
        show_chunks = not args.hide_chunks

        print(
            f"\n{Fore.GREEN}Entering interactive mode. Type 'exit' to quit.{Style.RESET_ALL}"
        )
        print("Type 'debug on' to enable debug mode, 'debug off' to disable.")
        print("Type 'chunks on' to show retrieved chunks, 'chunks off' to hide them.")

        debug_enabled = args.debug
        print(
            f"{Fore.CYAN}Debug mode: {'ON' if debug_enabled else 'OFF'}{Style.RESET_ALL}"
        )
        print(
            f"{Fore.CYAN}Show chunks: {'ON' if show_chunks else 'OFF'}{Style.RESET_ALL}"
        )
        print(
            f"{Fore.YELLOW}Detailed logs are being saved to the logs directory.{Style.RESET_ALL}"
        )

        while True:
            query = input(f"\n{Fore.GREEN}You: {Style.RESET_ALL}")

            if query.lower() in ["exit", "quit"]:
                break
            elif query.lower() == "debug on":
                debug_enabled = True
                print(f"{Fore.CYAN}Debug mode enabled{Style.RESET_ALL}")
                continue
            elif query.lower() == "debug off":
                debug_enabled = False
                print(f"{Fore.CYAN}Debug mode disabled{Style.RESET_ALL}")
                continue
            elif query.lower() == "chunks on":
                show_chunks = True
                print(f"{Fore.CYAN}Retrieved chunks display enabled{Style.RESET_ALL}")
                continue
            elif query.lower() == "chunks off":
                show_chunks = False
                print(f"{Fore.CYAN}Retrieved chunks display disabled{Style.RESET_ALL}")
                continue

            # Force garbage collection before each query
            gc.collect()

            start_time = time.time()

            if debug_enabled:
                response, debug_info, retrieved_chunks = llm.query(
                    query, debug_mode=True, show_retrieved_chunks=show_chunks
                )

                elapsed = time.time() - start_time

                # Display retrieved chunks first
                if show_chunks and retrieved_chunks:
                    print(f"\n{retrieved_chunks}")

                # Display the response
                print(f"\n{Fore.CYAN}LLM ({elapsed:.2f}s):{Style.RESET_ALL}")
                print(response)

                print(
                    f"\n{Fore.YELLOW}Detailed logs saved to logs directory.{Style.RESET_ALL}"
                )
            else:
                response, retrieved_chunks = llm.query(
                    query, show_retrieved_chunks=show_chunks
                )

                elapsed = time.time() - start_time

                # Display retrieved chunks first
                if show_chunks and retrieved_chunks:
                    print(f"\n{retrieved_chunks}")

                # Display the response
                print(f"\n{Fore.CYAN}LLM ({elapsed:.2f}s):{Style.RESET_ALL}")
                print(response)

    # Clean up
    llm.close()


if __name__ == "__main__":
    main()
