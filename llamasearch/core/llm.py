# llamasearch/core/llm.py

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
from ..core.vectordb import VectorDB

colorama.init()
logger = setup_logging(__name__)


class OptimizedLLM:
    """
    LLM implementation using llama-cpp-python with Qwen 2.5 1.5B model
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
        self.verbose = verbose
        self.persist = persist
        self.context_length = context_length
        self.n_results = n_results
        self.model_name = model_name
        self.custom_model_path = custom_model_path

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

        # Create a VectorDB that handles embeddings in vectordb.py
        self.storage_dir = os.path.join(project_root, "vector_db")
        self.vectordb = VectorDB(
            persist=persist,
            chunk_size=250,
            text_embedding_size=512,
            chunk_overlap=50,
            min_chunk_size=50,
            batch_size=2,
            similarity_threshold=0.25,
            storage_dir=self.storage_dir,
            use_pca=False,
        )

        self._process_temp_documents()

        self.llm_instance = None
        logger.info(f"Initialized OptimizedLLM with model: {self.model_name}")
        logger.info(f"Model path: {self.model_path}")

    def _process_temp_documents(self):
        project_root = find_project_root()
        temp_dir = os.path.join(project_root, "temp")

        if not os.path.exists(temp_dir):
            logger.info(f"Temp directory not found: {temp_dir}")
            return

        logger.info(f"Processing documents from temp directory: {temp_dir}")
        for filename in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, filename)
            if os.path.isfile(file_path):
                logger.info(f"Processing file: {filename}")
                try:
                    self.vectordb.add_document(file_path)
                except Exception as e:
                    logger.error(f"Error processing {filename}: {e}")

    def _get_llm(self):
        if self.llm_instance is None:
            logger.info(f"Loading LLM model from {self.model_path}")
            n_threads = min(os.cpu_count() or 4, 8)
            n_gpu_layers = 0
            if platform.system() == "Darwin" and platform.processor() == "arm":
                n_gpu_layers = -1
                logger.info("Apple Silicon detected, using Metal for acceleration")

            try:
                # Increase generation batch size by specifying n_batch=4
                self.llm_instance = Llama(
                    model_path=self.model_path,
                    n_ctx=self.context_length,
                    n_threads=n_threads,
                    n_gpu_layers=n_gpu_layers,
                    verbose=self.verbose,
                    n_batch=4,  # <--- increase Llama generation batch size
                )
                logger.info(f"Model loaded successfully with context length {self.context_length}")
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                raise RuntimeError(f"Failed to load model: {e}")

        return self.llm_instance

    def _analyze_query_intent(self, query_text: str) -> Dict[str, Union[bool, str]]:
        intent = {
            "has_greeting": False,
            "information_request": None,
            "requires_rag": False,
        }

        greeting_patterns = [
            r"\bhello\b", r"\bhi\b", r"\bhey\b", r"\bgreetings\b",
            r"\bgood\s(morning|afternoon|evening|day)\b", r"\bwhat\'s\sup\b", r"\bhowdy\b",
        ]
        intent["has_greeting"] = any(
            re.search(p, query_text.lower()) for p in greeting_patterns
        )

        info_req_patterns = [
            r"tell\s+me\s+about\s+(.+)",
            r"what\s+is\s+(.+)",
            r"how\s+(?:do|does|to|can|would|could)\s+(.+)",
            r"explain\s+(.+)",
            r"describe\s+(.+)",
            r"show\s+me\s+(.+)",
            r"find\s+(.+)",
            r"search\s+for\s+(.+)",
        ]
        for pat in info_req_patterns:
            m = re.search(pat, query_text.lower())
            if m:
                intent["information_request"] = m.group(1).strip()
                intent["requires_rag"] = True
                break

        # If still no pattern but more than 3 words => assume info request
        if not intent["information_request"] and len(query_text.split()) > 3:
            if not intent["has_greeting"]:
                intent["information_request"] = query_text
                intent["requires_rag"] = True

        return intent

    def _build_prompt(self, query_text: str, query_intent: Dict[str, Union[bool, str]], context: str = "") -> str:
        if self.context_length <= 2048:
            if query_intent["has_greeting"] and not query_intent["requires_rag"]:
                system_content = "You are Qwen, a helpful AI assistant."
            elif query_intent["has_greeting"] and query_intent["requires_rag"]:
                system_content = "You are Qwen. Answer based on the context if relevant."
            else:
                system_content = "You are Qwen. Provide a direct answer using the context."

            prompt = (
                f"<|im_start|>system\n{system_content}<|im_end|>\n<|im_start|>user\n"
            )
            if query_intent["requires_rag"] and context:
                prompt += f"Context:\n{context}\n\n"
            prompt += f"Q: {query_text}<|im_end|>\n<|im_start|>assistant\n"

        else:
            system_content = "You are Qwen. Provide helpful answers using the context if available."
            prompt = (
                f"<|im_start|>system\n{system_content}<|im_end|>\n"
                f"<|im_start|>user\n"
            )
            if query_intent["requires_rag"] and context:
                prompt += f"Context:\n{context}\n\n"
            prompt += f"Question: {query_text}<|im_end|>\n<|im_start|>assistant\n"

        return prompt

    def _truncate_context_to_fit(
        self, context: str, prompt_template: str, query_text: str, max_tokens: int = 2048
    ) -> str:
        if len(context) > 3000:
            return context[:2000]
        return context

    def query(
        self,
        query_text: str,
        context: str = None,
        show_retrieved_chunks: bool = True,
        debug_mode: bool = False,
    ) -> Union[str, Tuple[str, Dict]]:
        logger.info(f"Query: {query_text}")
        start_time = time.time()

        debug_info = {}
        retrieved_chunks_display = ""

        query_intent = self._analyze_query_intent(query_text)
        debug_info["query_intent"] = query_intent

        # If user wants RAG
        if not context and query_intent["requires_rag"]:
            logger.info("Retrieving context from vector database...")
            retrieval_start = time.time()
            try:
                if debug_mode:
                    ctx, ctx_debug_info, chunk_data = self.vectordb.get_context_for_query(
                        query_text, n_results=self.n_results, debug_mode=True, return_chunks=True
                    )
                    debug_info.update(ctx_debug_info)
                    context = ctx
                    if show_retrieved_chunks and chunk_data:
                        retrieved_chunks_display = self._format_retrieved_chunks(chunk_data)
                else:
                    ctx, chunk_data = self.vectordb.get_context_for_query(
                        query_text, n_results=self.n_results, return_chunks=True
                    )
                    context = ctx
                    if show_retrieved_chunks and chunk_data:
                        retrieved_chunks_display = self._format_retrieved_chunks(chunk_data)

                retrieval_time = time.time() - retrieval_start
                logger.info(f"Retrieved context in {retrieval_time:.2f}s")
                debug_info["retrieval_time"] = retrieval_time
            except Exception as e:
                logger.error(f"Error retrieving context: {e}")
                debug_info["retrieval_error"] = str(e)
                context = ""

        # Build prompt
        prompt_template = self._build_prompt(query_text, query_intent, "{context}")
        if context:
            truncated_context = self._truncate_context_to_fit(
                context, prompt_template, query_text, self.context_length
            )
            prompt = prompt_template.format(context=truncated_context)
        else:
            prompt = prompt_template.format(context="")

        prompt_token_est = len(prompt.split()) * 2
        logger.info(f"Estimated prompt tokens: {prompt_token_est}")
        base_prompt_tokens = len(prompt_template.split()) * 2
        max_tokens = self.context_length
        available_tokens = int(max_tokens * 0.4)
        logger.info(
            f"Token budget - Max: {max_tokens}, Base: {base_prompt_tokens}, Available: {available_tokens}"
        )

        llm = self._get_llm()

        logger.info("Generating response...")
        generation_start = time.time()
        try:
            # We'll keep small max_new_tokens
            max_new_tokens = 128 if self.context_length <= 2048 else 256
            output = llm(
                prompt,
                max_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                repeat_penalty=1.1,
                stop=["<|im_end|>", "<|endoftext|>", "<|im_start|>"],
                echo=False,
            )
            response = output["choices"][0]["text"].strip()
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            response = f"Error generating response: {str(e)}"
            debug_info["generation_error"] = str(e)

        generation_time = time.time() - generation_start
        total_time = time.time() - start_time
        logger.info(f"Generated response in {generation_time:.2f}s")
        logger.info(f"Total query time: {total_time:.2f}s")
        debug_info["generation_time"] = generation_time
        debug_info["total_time"] = total_time

        # Log the entire conversation
        log_file = log_query(
            query=query_text,
            context_chunks=debug_info.get("chunks", []),
            response=response,
            debug_info=debug_info,
        )
        logger.info(f"Query log saved to {log_file}")

        if debug_mode:
            return response, debug_info, retrieved_chunks_display
        else:
            return response, retrieved_chunks_display

    def _format_retrieved_chunks(self, chunk_data: List[Dict]) -> str:
        if not chunk_data:
            return f"{Fore.YELLOW}No relevant chunks found{Style.RESET_ALL}"

        lines = [f"{Fore.CYAN}Retrieved {len(chunk_data)} relevant chunks:{Style.RESET_ALL}\n"]
        for i, ch in enumerate(chunk_data):
            similarity = ch.get("similarity", 0) * 100
            text = ch["text"]
            source = ch["metadata"].get("source", "unknown")
            lines.append(f"{Fore.GREEN}Chunk {i+1} - {similarity:.2f}% match:{Style.RESET_ALL}")
            lines.append(f"{Fore.YELLOW}Source: {source}{Style.RESET_ALL}")
            lines.append("  " + text.replace("\n", "\n  "))
            lines.append("")
        return "\n".join(lines)

    def chat(self, conversation_history, debug_mode=False, show_retrieved_chunks=True):
        last_user = ""
        for msg in reversed(conversation_history):
            if msg["role"] in ("user", "human"):
                last_user = msg["content"]
                break
        if not last_user:
            return "No user message found."

        return self.query(
            last_user,
            debug_mode=debug_mode,
            show_retrieved_chunks=show_retrieved_chunks,
        )

    def add_document(self, file_path: str) -> int:
        logger.info(f"Adding document: {file_path}")
        st = time.time()
        try:
            cnt = self.vectordb.add_document(file_path)
            logger.info(f"Added {cnt} chunks in {time.time()-st:.2f}s")
            return cnt
        except Exception as e:
            logger.error(f"Error adding document: {e}")
            raise

    def close(self):
        try:
            self.vectordb.close()
            self.llm_instance = None
            gc.collect()
            logger.info("Resources cleaned up")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


def main():
    parser = argparse.ArgumentParser(description="LlamaSearch - RAG + Qwen2.5-1.5B llama-cpp.")
    parser.add_argument("--document", type=str, default=None, help="Path to .md doc to process")
    parser.add_argument("--query", type=str, default=None, help="Query to run")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--debug", action="store_true", help="Debug mode (show chunk usage)")
    parser.add_argument("--persist", action="store_true", help="Don't clear the vector DB.")
    parser.add_argument("--custom-model", type=str, default=None, help="Path to custom model")
    parser.add_argument("--context-length", type=int, default=2048, help="Context window size")
    parser.add_argument("--hide-chunks", action="store_true", help="Hide retrieved chunks in output")
    parser.add_argument("--verbose", action="store_true", help="Verbose logs")
    args = parser.parse_args()

    llm = OptimizedLLM(
        persist=args.persist,
        verbose=args.verbose,
        context_length=args.context_length,
        custom_model_path=args.custom_model,
    )

    if args.document:
        import os
        if os.path.isfile(args.document):
            if not args.document.lower().endswith(".md"):
                print("Only markdown (.md) files are supported.")
            else:
                cnt = llm.add_document(args.document)
                print(f"Added {cnt} chunks from {args.document}")
        elif os.path.isdir(args.document):
            total_chunks = 0
            for f in os.listdir(args.document):
                fp = os.path.join(args.document, f)
                if f.lower().endswith(".md"):
                    c = llm.add_document(fp)
                    total_chunks += c
                    print(f"Added {c} from {fp}")
            print(f"Total {total_chunks} chunks added.")
        else:
            print(f"No file or directory found at {args.document}")

    if args.query:
        show_chunks = not args.hide_chunks
        import time
        st = time.time()
        if args.debug:
            resp, dbg, chunk_info = llm.query(args.query, debug_mode=True, show_retrieved_chunks=show_chunks)
            if show_chunks and chunk_info:
                print(chunk_info)
            print("\nResponse:\n", resp)
            print(f"Query took {time.time()-st:.2f}s")
        else:
            resp, chunk_info = llm.query(args.query, show_retrieved_chunks=show_chunks)
            if show_chunks and chunk_info:
                print(chunk_info)
            print("\nResponse:\n", resp)
            print(f"Query took {time.time()-st:.2f}s")

    if args.interactive:
        print("Interactive mode. Type 'exit' to quit.")
        show_chunks = not args.hide_chunks
        while True:
            q = input("You: ")
            if q.lower() in ("exit", "quit"):
                break
            st = time.time()
            resp, chunk_info = llm.query(q, show_retrieved_chunks=show_chunks)
            if show_chunks and chunk_info:
                print(chunk_info)
            print("\nAssistant:\n", resp)
            print(f"Took {time.time()-st:.2f}s")

    llm.close()


if __name__ == "__main__":
    main()

