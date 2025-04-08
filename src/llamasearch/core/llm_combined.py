# src/llamasearch/core/llm_combined.py
import os
import re
import time
import gc
from pathlib import Path
from typing import Dict, Any

from llamasearch.utils import setup_logging, log_query
from llamasearch.core.vectordb import VectorDB
from llamasearch.core.embedder import EnhancedEmbedder
from llamasearch.core.resource_manager import get_resource_manager
from llamasearch.core.chunker import process_directory

# Optional imports (only used if model_engine=='hf')
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from llama_cpp import Llama

logger = setup_logging(__name__)

DEFAULT_MODEL_NAME = "qwen2.5-1.5b-instruct-q4_k_m"

class LLMSearch:
    """
    A universal RAG-based search class that loads either:
      - llama-cpp (model_engine='llamacpp'), or
      - a Hugging Face Transformers model (model_engine='hf')

    The advanced RAG methods from old LlamaSearch code (add_documents_from_directory, llm_query)
    are included, so that your existing references to these methods will still work.

    The system uses a VectorDB for chunk-based retrieval, plus optional named entity queries.
    """

    def __init__(
        self,
        storage_dir: Path,
        models_dir: Path,
        model_name: str = DEFAULT_MODEL_NAME,
        model_engine: str = "llamacpp",   # "llamacpp" or "hf"
        verbose: bool = True,
        context_length: int = 4096,
        max_results: int = 3,
        custom_model_path: str = "",
        auto_optimize: bool = False,
        embedder_batch_size: int = 8,
        force_cpu: bool = False,
        max_workers: int = 4,
        debug: bool = False,
    ):
        """
        Args:
            storage_dir: Where to store the vector index and related data
            models_dir: Where to store or find the LLM model
            model_name: Name of the model (for huggingface or the default llama-cpp name)
            model_engine: Either 'llamacpp' or 'hf' (hugging face)
            verbose: Whether to log verbosely
            context_length: LLM context window (if using llama-cpp)
            max_results: Max retrieval results from VectorDB
            custom_model_path: Custom file path to load from (for llama-cpp) or HF
            auto_optimize: If True, tries to auto-optimize GPU usage
            embedder_batch_size: Batch size for embedding
            force_cpu: If True, forces CPU usage
            max_workers: Worker threads for chunking or embeddings
            debug: Extra debug logs
        """
        self.verbose = verbose
        self.context_length = context_length
        self.max_results = max_results
        self.model_name = model_name
        self.custom_model_path = custom_model_path
        self.auto_optimize = auto_optimize
        self.embedder_batch_size = embedder_batch_size
        self.force_cpu = force_cpu
        self.max_workers = max_workers
        self.debug = debug
        self.model_engine = model_engine.lower().strip()

        self.resource_manager = get_resource_manager(auto_optimize=self.auto_optimize)
        self.device = self.resource_manager.get_embedding_config()['device']

        self.models_dir = models_dir
        self.storage_dir = storage_dir
        os.makedirs(self.models_dir, exist_ok=True)

        if self.custom_model_path:
            self.model_path = self.custom_model_path
            logger.info(f"Using custom model path: {self.model_path}")
        else:
            # e.g. for llama-cpp we might do {model_name}.gguf
            # for HF we typically do a huggingface repo name
            self.model_path = os.path.join(self.models_dir, f"{self.model_name}.gguf") if self.model_engine=="llamacpp" \
                                else self.model_name
            logger.info(f"Using model: {self.model_name} => path: {self.model_path}")

        if self.model_engine=="llamacpp" and not os.path.exists(self.model_path):
            logger.warning(f"Llama-cpp model not found at {self.model_path}")
            logger.info("Please download or provide a valid .gguf model file.")

        embedding_config = {}
        if self.auto_optimize:
            embedding_config = self.resource_manager.get_embedding_config()

        # Determine embedding device based on hardware
        embedding_device = "cpu"
        if not force_cpu:
            if self.resource_manager.hardware.has_cuda:
                embedding_device = "cuda"
            elif self.resource_manager.hardware.has_mps:
                embedding_device = "mps"

        self.embedder = EnhancedEmbedder(
            device=embedding_device,
            batch_size=self.embedder_batch_size,
            auto_optimize=self.auto_optimize,
            num_workers=self.max_workers,
            embedding_config=embedding_config,
        )

        # Initialize VectorDB
        self.vectordb = VectorDB(
            embedder=self.embedder,
            similarity_threshold=0.25,
            storage_dir=self.storage_dir,
            collection_name="default",
            max_chunk_size=512,
            chunk_overlap=64,
            min_chunk_size=128,
            max_results=self.max_results,
            device=self.device,
        )

        self.llm_instance = None
        logger.info(f"Initialized LLMSearch with model_engine='{self.model_engine}'")
        logger.info(f"Max retrieval chunks: {self.max_results}")

    def _get_llm(self):
        """Load or return the underlying model, depending on self.model_engine."""
        if self.llm_instance is not None:
            return self.llm_instance

        if self.model_engine == "llamacpp":
            return self._load_llama_cpp()
        else:
            # We assume hugging face
            return self._load_huggingface_model()

    def _load_llama_cpp(self):
        """Load a llama-cpp model from a .gguf or .bin file."""
        logger.info(f"Loading llama-cpp model from: {self.model_path}")
        llm_config = self.resource_manager.get_llm_config() if self.auto_optimize else {}
        n_threads = llm_config.get("n_threads", 4)
        n_gpu_layers = llm_config.get("n_gpu_layers", 0)

        kwargs = {
            "model_path": self.model_path,
            "n_ctx": self.context_length,
            "n_threads": n_threads,
            "verbose": self.verbose,
            "chat_format": "custom",
            "chat_template": (
                "{%- if messages[0]['role'] == 'system' -%}"
                "<|im_start|>system\n{{ messages[0]['content'] }}<|im_end|>\n"
                "{%- endif -%}"
                "{%- for message in messages[1:] -%}"
                "<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n"
                "{%- endfor -%}"
                "<|im_start|>assistant\n"
            ),
        }
        if self.resource_manager.hardware.has_cuda and not self.force_cpu:
            kwargs["n_gpu_layers"] = n_gpu_layers

        try:
            llm_instance = Llama(**kwargs)
            logger.info(f"Llama-cpp model loaded (ctx={self.context_length})")
            return llm_instance
        except Exception as e:
            logger.error(f"Error loading llama-cpp model: {e}")
            raise RuntimeError(f"Failed to load llama-cpp model: {str(e)}")

    def _load_huggingface_model(self):
        """
        Load a hugging face model. By default we assume a causal LM (e.g. GPT2, LLaMA, etc.).
        If your model is a seq2seq model (like T5), you can adapt the code below or auto-detect.
        """
        try:
            logger.info(f"Loading Hugging Face model: {self.model_path}")
            # If user wants a large context, some models might not support it. We'll ignore that for now.
            tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=False)
            # pick a suitable model class
            # if you have a T5 or BART model, you'd want AutoModelForSeq2SeqLM
            # if you have a GPT- or llama-based HF model, you'd want AutoModelForCausalLM
            # For now we assume a causal LM:
            model = AutoModelForCausalLM.from_pretrained(self.model_path)
            # We'll wrap with a pipeline for convenience
            device_name = "cpu"
            if not self.force_cpu and self.resource_manager.hardware.has_cuda:
                device_name = "cuda:0"
            elif not self.force_cpu and self.resource_manager.hardware.has_mps:
                # pipeline can accept "mps" device string in some versions
                device_name = "mps"

            # We create a text-generation pipeline
            llm_instance = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device=device_name
            )
            logger.info(f"Hugging Face model loaded: {self.model_path}")
            return llm_instance
        except Exception as e:
            logger.error(f"Error loading Hugging Face model: {e}")
            raise RuntimeError(f"Failed to load huggingface model: {str(e)}")

    def _call_llamacpp(self, prompt: str, max_tokens=512, temperature=0.7, top_p=0.9, repeat_penalty=1.1):
        """Call the llama-cpp model and return a text response or error."""
        llm = self._get_llm()  # must be a Llama instance
        if not llm and not isinstance(llm, Llama):
            logger.error("No llama-cpp model loaded.")
            return "Error: no model loaded", None
        try:
            completion = llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repeat_penalty=repeat_penalty,
                stop=["<|im_end|>", "<|endoftext|>", "<|im_start|>"],
                echo=False,
                stream=False,
            )
        except Exception as e:
            logger.error(f"Error calling llama-cpp: {e}")
            return f"Error: {str(e)}", None  # no completion

        if (
            isinstance(completion, dict)
            and "choices" in completion
            and len(completion["choices"])>0
            and "text" in completion["choices"][0]
        ):
            text = completion["choices"][0]["text"].strip()
            return text, completion
        else:
            logger.warning(f"Unexpected completion format: {completion}")
            return str(completion), completion

    def _call_huggingface(self, prompt: str, max_tokens=512, temperature=0.7, top_p=0.9, repeat_penalty=1.1):
        """Call the HF pipeline and return a text response or error."""
        pipe = self.llm_instance  # must be a pipeline
        if not pipe:
            logger.error("No HF pipeline loaded.")
            return "Error: no pipeline loaded", None

        # The pipeline for text-generation usually returns a list of dict
        # e.g. [ { 'generated_text': 'some text...' } ]
        # We'll pass the arguments as needed
        try:
            # Usually the pipeline uses 'max_new_tokens' not 'max_tokens'
            # We'll just pass them as is, and hope it works
            result = pipe(prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p)
        except Exception as e:
            logger.error(f"Error calling HF pipeline: {e}")
            return f"Error: {str(e)}", None

        # The pipeline returns a list of dict e.g. [{'generated_text': 'some text...'}]
        if isinstance(result, list) and len(result)>0 and "generated_text" in result[0]:
            text = result[0]["generated_text"]
            # remove the prompt from the start if it's included
            if text.startswith(prompt):
                text = text[len(prompt):]
            return text.strip(), result
        else:
            logger.warning(f"Unexpected HF pipeline format: {result}")
            return str(result), result

    def query_model(self, prompt: str, max_tokens=512, temperature=0.7, top_p=0.9, repeat_penalty=1.1):
        """
        Unified entry point for calling the underlying model (llama-cpp or HF).
        Returns (text, raw_completion).
        """
        llm = self._get_llm()

        if self.model_engine == "llamacpp":
            return self._call_llamacpp(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repeat_penalty=repeat_penalty,
            )
        else:
            return self._call_huggingface(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repeat_penalty=repeat_penalty,
            )

    ############################################################################
    # The advanced RAG methods from older LlamaSearch code (like llm.py)
    # so that references to add_documents_from_directory() and llm_query() keep working
    ############################################################################

    def add_documents_from_directory(self, directory_path: str) -> int:
        """
        Process and add all documents from a directory to the vectordb,
        using chunkers from VectorDB. 
        """
        results = process_directory(
            directory_path=directory_path,
            markdown_chunker=self.vectordb.markdown_chunker,
            html_chunker=self.vectordb.html_chunker,
        )
        total_chunks = 0
        for file_path, chunks in results.items():
            try:
                if self.vectordb.is_document_processed(file_path):
                    logger.info(f"File already processed: {file_path}")
                    # Count how many chunks we already had
                    ccount = sum(1 for m in self.vectordb.document_metadata if m.get("source")==file_path)
                    total_chunks += ccount
                    continue
                self.vectordb.add_document_chunks(file_path, chunks)
                total_chunks += len(chunks)
                logger.info(f"Added {len(chunks)} chunks from {os.path.basename(file_path)}")
            except Exception as e:
                logger.error(f"Error adding {file_path} to vectordb: {e}")
        return total_chunks

    def llm_query(self, query_text: str, debug_mode: bool=False) -> Dict[str, Any]:
        """
        Perform RAG-based retrieval from vectordb, build a prompt, call the LLM,
        and return a dict with "response", "retrieved_display", and optional debug_info.
        """
        logger.info(f"Query: {query_text}")
        start_t = time.time()
        debug_info: Dict[str, Any] = {}

        # Step 1: Named entity detection from vectordb
        is_entity_query, target_entity = self.vectordb.is_named_entity_query(query_text)
        intent = {
            "has_greeting": bool(re.search(r"\b(hello|hi|hey|greetings)\b", query_text.lower())),
            "information_request": query_text if len(query_text.split())>2 else None,
            "requires_rag": True,
            "is_entity_query": is_entity_query,
            "target_entity": target_entity
        }
        debug_info["intent"] = intent

        final_context = ""
        retrieved_display = ""
        logger.info("Retrieving context from vectordb...")
        entity_found = False

        try:
            # Possibly adjust max_results for entity queries
            effective_max_results = self.max_results
            if is_entity_query and target_entity:
                effective_max_results = min(self.max_results + 3, 6)
                logger.info(f"Entity query for '{target_entity}', using {effective_max_results} max results")

            results = self.vectordb.vectordb_query(query_text, max_results=effective_max_results)
            docs = results.get("documents", [])
            if not docs:
                return {"response": "No relevant context found.", "debug_info": debug_info, "retrieved_display": ""}

            metas = results.get("metadatas", [])
            logger.info(f"Retrieved {len(docs)} documents")

            # Check if target entity is found
            if is_entity_query and target_entity:
                for doc in docs:
                    if target_entity.lower() in doc.lower():
                        entity_found = True
                        logger.info(f"Entity '{target_entity}' found in retrieved docs")
                        break

            # Build a textual context
            for i, doc_text in enumerate(docs):
                score = results.get("scores", [0]*len(docs))[i]
                final_context += f"[Doc {i+1} (score={score:.2f})]\n{doc_text}\n\n"
            
            for i, meta in enumerate(metas):
                source = meta.get("source", "N/A") if isinstance(meta, dict) else "N/A"
                content = docs[i]
                retrieved_display += f"Chunk {i+1} - Source: {source}\nContent: {content}\n\n"

            debug_info["chunks"] = [
                {
                    "id": results.get("ids", [])[i] if i<len(results.get("ids", [])) else f"doc_{i}",
                    "score": results.get("scores", [])[i] if i<len(results.get("scores", [])) else 0.0,
                    "metadata": meta,
                    "text": docs[i]
                }
                for i, meta in enumerate(metas)
            ]
            debug_info["retrieval_time"] = time.time()-start_t
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            debug_info["retrieval_error"] = str(e)
            return {"response": f"Error retrieving context: {str(e)}", "debug_info": debug_info, "retrieved_display": ""}

        # Step 2: Build an appropriate system instruction
        system_instruction = "You are a helpful AI assistant. Answer based on the provided context."
        if is_entity_query and target_entity:
            system_instruction = (
                f"You are a helpful AI assistant. Answer based on the provided context. "
                f"Focus specifically on information about '{target_entity}'."
            )
        # Build final prompt
        prompt = (
            f"<|im_start|>system\n{system_instruction}<|im_end|>\n"
            f"<|im_start|>user\nContext:\n{final_context}\n\nQuery: {query_text}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        # Step 3: Call the underlying model
        gen_start = time.time()
        text_response, raw_completion = self.query_model(
            prompt=prompt,
            max_tokens=512,
            temperature=0.7,
            top_p=0.9,
            repeat_penalty=1.1
        )
        generation_time = time.time() - gen_start
        total_time = time.time() - start_t
        logger.info(f"Generated response in {generation_time:.2f}s, total {total_time:.2f}s")

        # Possibly fix up text if target entity is missing
        if is_entity_query and target_entity and entity_found:
            if target_entity.lower() not in text_response.lower():
                logger.warning(f"Entity '{target_entity}' is not found in the final answer text.")
                if not text_response.startswith("I couldn't find"):
                    text_response = f"The context includes info about {target_entity}, specifically: {text_response}"
        
        # Save query logs
        debug_info["generation_time"] = generation_time
        debug_info["total_time"] = total_time
        log_file = log_query(query_text, debug_info.get("chunks", []), text_response, debug_info, full_logging=debug_mode)
        logger.info(f"Query log saved to {log_file}")

        if debug_mode:
            debug_info["raw_completion"] = raw_completion
            return {
                "response": text_response,
                "debug_info": debug_info,
                "retrieved_display": retrieved_display
            }
        else:
            return {
                "response": text_response,
                "debug_info": {},
                "retrieved_display": retrieved_display
            }

    def unload_model(self) -> None:
        """Release references to the model and force GC."""
        if self.llm_instance is not None:
            try:
                logger.info("Unloading the model instance for LLMSearch...")
                self.llm_instance = None
                gc.collect()
            except Exception as e:
                logger.warning(f"Error during model unloading: {e}")
                self.llm_instance = None
                gc.collect()

    def close(self) -> None:
        """
        Clean up resources: unload model and close embedder if possible.
        """
        self.unload_model()
        if hasattr(self.embedder, "close"):
            self.embedder.close()
        gc.collect()
        logger.info("All resources for LLMSearch have been cleaned up.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()