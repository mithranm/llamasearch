# llamasearch/core/llm.py

import os
import argparse
import time
import gc
import re
import platform
from typing import Dict, Any, Tuple, Optional, Union, cast

from llama_cpp import Llama

from ..setup_utils import find_project_root
from ..utils import setup_logging, log_query
from .vectordb import VectorDB

logger = setup_logging(__name__)


class OptimizedLLM:
    """
    LLM orchestrator that uses advanced chunker + vectordb with knowledge graph and name expansions.
    Does not do batch generation (no n_batch param).
    """

    def __init__(
        self,
        model_name: str = "qwen2.5-1.5b-instruct-q4_k_m",
        persist: bool = False,
        verbose: bool = True,
        context_length: int = 2048,
        n_results: int = 5,
        custom_model_path: Optional[str] = None,
        auto_optimize: bool = True,
        embedder_batch_size: Optional[int] = None,
        chunker_batch_size: Optional[int] = None,
    ):
        self.verbose = verbose
        self.persist = persist
        self.context_length = context_length
        self.n_results = n_results
        self.model_name = model_name
        self.custom_model_path = custom_model_path
        self.auto_optimize = auto_optimize
        self.embedder_batch_size = embedder_batch_size

        project_root = find_project_root()
        self.models_dir = os.path.join(project_root, "models")
        os.makedirs(self.models_dir, exist_ok=True)

        if custom_model_path:
            self.model_path = custom_model_path
            logger.info(f"Using custom model at: {self.model_path}")
        else:
            self.model_path = os.path.join(self.models_dir, f"{model_name}.gguf")
            logger.info(f"Using model: {model_name} ({self.model_path})")

        if not os.path.exists(self.model_path):
            logger.warning(f"Model not found at {self.model_path}")
            logger.info("Please download or provide a valid model path.")

        self.storage_dir = os.path.join(project_root, "vector_db")
        
        # Initialize VectorDB with auto_optimize flag and optional batch_size
        self.vectordb = VectorDB(
            persist=persist,
            chunk_size=250,
            text_embedding_size=512,
            chunk_overlap=50,
            min_chunk_size=50,
            chunker_batch_size=chunker_batch_size,
            embedder_batch_size=embedder_batch_size,  # Use provided batch size or auto
            similarity_threshold=0.25,
            storage_dir=self.storage_dir,
            use_pca=False
        )

        self._process_temp_docs()

        self.llm_instance = None
        logger.info(f"Initialized OptimizedLLM with model: {self.model_name}")
        logger.info(f"Model path: {self.model_path}")
        logger.info(f"Auto-optimize: {self.auto_optimize}, Batch size: {self.embedder_batch_size or 'auto'}")

    def _process_temp_docs(self):
        proj_root = find_project_root()
        temp_dir = os.path.join(proj_root, "temp")
        if not os.path.exists(temp_dir):
            logger.info(f"Temp dir not found: {temp_dir}")
            return
        logger.info(f"Processing docs from {temp_dir}")
        for fn in os.listdir(temp_dir):
            fp = os.path.join(temp_dir, fn)
            if os.path.isfile(fp) and fn.lower().endswith(".md"):
                logger.info(f"Processing file: {fn}")
                try:
                    self.vectordb.add_document(fp)
                except Exception as e:
                    logger.error(f"Error processing {fn}: {e}")
            elif os.path.isfile(fp):
                logger.warning(f"Skipping non-markdown file: {fn}")

    def _get_llm(self):
        if self.llm_instance is None:
            logger.info(f"Loading LLM from {self.model_path}")
            n_threads = min(os.cpu_count() or 4, 8)
            n_gpu_layers = 0
            if platform.system() == "Darwin" and platform.processor() == "arm":
                n_gpu_layers = -1
                logger.info("Detected Apple Silicon, using Metal acceleration")

            try:
                # Define a simple chat template without toolcalling
                chat_template = """
                {%- if messages[0]['role'] == 'system' -%}
                {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' -}}
                {%- endif -%}
                {%- for message in messages[1:] -%}
                {{- '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' -}}
                {%- endfor -%}
                {{- '<|im_start|>assistant\\n' -}}
                """

                # Initialize with simpler chat template
                self.llm_instance = Llama(
                    model_path=self.model_path,
                    n_ctx=self.context_length,
                    n_threads=n_threads,
                    n_gpu_layers=n_gpu_layers,
                    verbose=self.verbose,
                    chat_format="custom",
                    chat_template=chat_template,
                )
                logger.info(f"Model loaded with context length {self.context_length}")
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                raise RuntimeError(str(e))
        return self.llm_instance

    def _analyze_query_intent(self, query: str) -> Dict[str, Union[bool, str]]:
        intent = {
            "has_greeting": False,
            "information_request": None,
            "requires_rag": True,  # Always use RAG by default
        }
        greet_pats = [r"\bhello\b", r"\bhi\b", r"\bhey\b", r"\bgreetings\b"]
        if any(re.search(p, query.lower()) for p in greet_pats):
            intent["has_greeting"] = True
        if len(query.split()) > 2:  # Reduced threshold
            intent["information_request"] = query
        return intent

    def _build_prompt(self, query: str, context: str, intent: dict) -> str:
        sys_msg = "You are Qwen, a helpful AI assistant. Answer based on the provided context."

        # Debug logging for context
        logger.info(f"Context length: {len(context) if context else 0}")
        if context:
            logger.info(f"Context preview: {context[:100]}...")

        prompt = f"<|im_start|>system\n{sys_msg}<|im_end|>\n<|im_start|>user\n"

        # Always include context if available, regardless of intent
        if context and context.strip():
            prompt += f"Context information:\n{context}\n\n"
        else:
            logger.warning("No context available to include in prompt")

        prompt += f"Question: {query}\n<|im_end|>\n<|im_start|>assistant\n"

        # Debug log the final prompt
        logger.info(f"Final prompt (truncated): {prompt[:200]}...")

        return prompt

    def query(
        self, query_text: str, show_retrieved_chunks=True, debug_mode=False
    ) -> Union[Tuple[str, str], Tuple[str, Dict[str, Any], str]]:
        """
        Query the LLM with the given text.

        Args:
            query_text: The query text
            show_retrieved_chunks: Whether to show retrieved chunks
            debug_mode: Whether to include debug info in the response

        Returns:
            If debug_mode is True, returns (response, debug_info, retrieved_display)
            If debug_mode is False, returns (response, retrieved_display)
        """
        logger.info(f"Query: {query_text}")
        start_t = time.time()
        debug_info: Dict[str, Any] = {}
        retrieved_display = ""

        intent = self._analyze_query_intent(query_text)
        debug_info["intent"] = intent

        final_context = ""
        logger.info("Retrieving context from vectordb...")
        st = time.time()
        try:
            results = self.vectordb.query(query_text, n_results=self.n_results)

            # Log the number of results found
            logger.info(f"Retrieved {len(results['documents'])} documents")

            # Build final context from top results
            final_context = ""
            for i, doc_text in enumerate(
                results["documents"][:5]
            ):  # Using top 5 results
                score = results["scores"][i] if i < len(results["scores"]) else 0
                final_context += (
                    f"[Document {i+1} (relevance: {score:.2f})]\n{doc_text}\n\n"
                )

            # Log context size
            logger.info(f"Built context with {len(final_context)} characters")

            # Store chunk data for debugging
            chunk_data = []
            for i, doc_text in enumerate(results["documents"]):
                chunk_data.append(
                    {
                        "id": results["ids"][i],
                        "score": results["scores"][i],
                        "metadata": results["metadatas"][i],
                        "text": (
                            doc_text[:200] + "..." if len(doc_text) > 200 else doc_text
                        ),
                    }
                )
            debug_info["chunks"] = chunk_data

            if show_retrieved_chunks:
                lines = [f"Retrieved {len(results['documents'])} relevant chunks:\n"]
                for i, doc_text in enumerate(results["documents"]):
                    sc = results["scores"][i]
                    lines.append(f"Chunk {i+1} - score {sc:.2f}")
                    lines.append(
                        "  "
                        + (doc_text[:200] + "..." if len(doc_text) > 200 else doc_text)
                    )
                    lines.append("")
                retrieved_display = "\n".join(lines)

            debug_info["retrieval_time"] = time.time() - st
            logger.info(f"Context retrieved in {debug_info['retrieval_time']:.2f}s")
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            debug_info["retrieval_error"] = str(e)

        prompt = self._build_prompt(query_text, final_context, intent)
        token_est = len(prompt.split()) * 2
        logger.info(f"Estimated prompt tokens: {token_est}")

        llm = self._get_llm()
        gen_start = time.time()
        response = ""
        try:
            # Explicitly set stream=False to get a dictionary response
            completion = llm(
                prompt,
                max_tokens=128,
                temperature=0.7,
                top_p=0.9,
                repeat_penalty=1.1,
                stop=["<|im_end|>", "<|endoftext|>", "<|im_start|>"],
                echo=False,
                stream=False,  # Set stream to False to get a dictionary
            )
            # Access the text from the completion dictionary
            if (
                isinstance(completion, dict)
                and "choices" in completion
                and len(completion["choices"]) > 0
                and "text" in completion["choices"][0]
            ):
                response = completion["choices"][0]["text"].strip()
            else:
                response = "Unexpected response format from LLM."
                logger.error(f"Unexpected response format: {completion}")

        except Exception as e:
            logger.error(f"Error generating: {e}")
            response = f"Error generating: {str(e)}"
            debug_info["generation_error"] = str(e)

        generation_time = time.time() - gen_start
        total_time = time.time() - start_t
        logger.info(
            f"Generated response in {generation_time:.2f}s, total {total_time:.2f}s"
        )
        debug_info["generation_time"] = generation_time
        debug_info["total_time"] = total_time

        log_file = log_query(
            query_text, debug_info.get("chunks", []), response, debug_info
        )
        logger.info(f"Query log saved to {log_file}")

        if debug_mode:
            return (response, debug_info, retrieved_display)
        else:
            return (response, retrieved_display)

    def close(self):
        self.vectordb.close()
        self.llm_instance = None
        gc.collect()
        logger.info("Resources cleaned up")


def main():
    parser = argparse.ArgumentParser("LlamaSearch advanced setup")
    parser.add_argument("--document", type=str, default=None, 
                      help="Path to document or directory of documents to add")
    parser.add_argument("--query", type=str, default=None,
                      help="Query to run against the documents")
    parser.add_argument("--persist", action="store_true",
                      help="Persist the vector database between runs")
    parser.add_argument("--debug", action="store_true",
                      help="Enable debug mode with additional info")
    parser.add_argument("--custom-model", type=str, default=None,
                      help="Path to a custom model file")
    parser.add_argument("--embedder-batch-size", type=int, default=None,
                      help="Fixed batch size for embeddings (disables auto-optimization)")
    parser.add_argument("--chunker-batch-size", type=int, default=None,
                      help="Fixed batch size for chunker (disables auto-optimization)")
    args = parser.parse_args()
    
    # Initialize with dynamic settings based on args
    llm = OptimizedLLM(
        persist=args.persist, 
        custom_model_path=args.custom_model,
        chunker_batch_size=args.chunker_batch_size
    )

    if args.query:
        st = time.time()
        if args.debug:
            # Cast to the specific tuple type to help type checker
            result = cast(
                Tuple[str, Dict[str, Any], str], llm.query(args.query, debug_mode=True)
            )
            # Explicitly handle this case as a 3-element tuple
            response = result[0]
            debug_info = result[1]
            display = result[2]
            print(display)
            print("\nDebug info:\n", debug_info)
            print("\nResponse:\n", response)
        else:
            # Cast to the specific tuple type to help type checker
            result = llm.query(args.query, debug_mode=False)
            # Explicitly handle this case as a 2-element tuple
            response = result[0]
            display = result[1]
            print(display)
            print("\nResponse:\n", response)
        print(f"\nQuery took {time.time()-st:.2f}s")

    llm.close()


if __name__ == "__main__":
    main()