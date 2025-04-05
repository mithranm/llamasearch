# llamasearch/core/llm.py

import os
import subprocess
import re
import time
import gc
import argparse
import platform
import shutil
from typing import Dict, Any, Optional, Union

from llama_cpp import Llama

from ..setup_utils import find_project_root
from ..utils import setup_logging, log_query
from .vectordb import VectorDB
from .embedder import EnhancedEmbedder
from .chunker import HtmlChunker, process_directory
from .resource_manager import get_resource_manager

logger = setup_logging(__name__)

DEFAULT_MODEL_NAME = "qwen2.5-1.5b-instruct-q4_k_m"
DEFAULT_EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- GPU Helper Functions ---

def get_nvidia_gpu_info() -> Optional[Dict[str, Any]]:
    """Returns NVIDIA GPU info using nvidia-smi if available."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True
        )
        if result.returncode == 0:
            output = result.stdout.strip().split(',')
            if len(output) >= 3:
                return {
                    "name": output[0].strip(),
                    "total_memory_mb": int(output[1].strip().split()[0]),
                    "free_memory_mb": int(output[2].strip().split()[0]),
                    "compute_cap": output[3].strip() if len(output) > 3 else None
                }
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    return None

def get_gpu_info() -> Dict[str, Any]:
    """Returns a dictionary with basic GPU info."""
    gpu_info = {"available": False, "type": None, "details": None}
    nvidia_info = get_nvidia_gpu_info()
    if nvidia_info:
        gpu_info["available"] = True
        gpu_info["type"] = "cuda"
        gpu_info["details"] = nvidia_info
    return gpu_info

# --- Data Directory Clearing ---
def clear_data_directory() -> None:
    """Wipes the data directory (reverse lookup, knowledge graph, etc.) on a fresh run."""
    project_root = find_project_root()
    data_dir = os.path.join(project_root, "data")
    if os.path.exists(data_dir):
        for fname in os.listdir(data_dir):
            fpath = os.path.join(data_dir, fname)
            try:
                if os.path.isfile(fpath):
                    os.unlink(fpath)
                else:
                    shutil.rmtree(fpath)
            except Exception as e:
                logger.error(f"Failed to remove {fpath}: {e}")

# --- LlamaSearch Class ---
class LlamaSearch:
    """
    LlamaSearch integrates retrieval (vector/BM25/graph) and an LLM for RAG.
    It supports ingesting documents (HTML/Markdown) and preserving source traceability.
    """
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        persist: bool = False,
        verbose: bool = True,
        context_length: int = 2048,
        n_results: int = 5,
        custom_model_path: Optional[str] = None,
        auto_optimize: bool = True,
        embedder_batch_size: Optional[int] = None,
        chunker_batch_size: Optional[int] = None,
        force_cpu: bool = False,
        max_workers: Optional[int] = None,
        debug: bool = False,
    ):
        self.verbose = verbose
        self.persist = persist
        self.context_length = context_length
        self.n_results = n_results
        self.model_name = model_name
        self.custom_model_path = custom_model_path
        self.auto_optimize = auto_optimize
        self.embedder_batch_size = embedder_batch_size
        self.chunker_batch_size = chunker_batch_size
        self.force_cpu = force_cpu
        self.max_workers = max_workers

        # GPU info
        self.gpu_info = get_gpu_info() if not force_cpu else {"available": False, "type": None, "details": None}

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

        self.resource_manager = get_resource_manager(auto_optimize=auto_optimize)
        embedding_config = {}
        if auto_optimize:
            embedding_config = self.resource_manager.get_embedding_config()

        # Choose embedding device
        embedding_device = "cpu"
        if not force_cpu:
            if self.gpu_info["available"]:
                embedding_device = "cuda" if self.gpu_info["type"] == "cuda" else "mps"
                logger.info(f"Using {embedding_device} for embedding model")
            else:
                logger.info("No compatible GPU detected for embedding, using CPU")

        self.embedder = EnhancedEmbedder(
            model_name=DEFAULT_EMBEDDING_MODEL_NAME,
            device=embedding_device,
            batch_size=embedder_batch_size or embedding_config.get("batch_size", 8),
            auto_optimize=auto_optimize,
            num_workers=max_workers
        )

        self.chunker = HtmlChunker(
            chunk_size=250,
            text_embedding_size=512,
            overlap_size=50,
            min_chunk_size=50,
            batch_size=chunker_batch_size,
            semantic_headers_only=True,
            num_workers=max_workers,
            auto_optimize=auto_optimize,
            debug_output=debug
        )

        self.vectordb = VectorDB(
            persist=persist,
            embedder=self.embedder,
            similarity_threshold=0.25,
            storage_dir=self.storage_dir,
            use_pca=False
        )

        self.llm_instance = None
        self._detect_gpu_capabilities()

        logger.info(f"Initialized LlamaSearch with model: {self.model_name}")
        logger.info(f"Model path: {self.model_path}")
        logger.info(f"Auto-optimize: {self.auto_optimize}")

        self._report_hardware_capabilities()

    def _report_hardware_capabilities(self):
        hardware = self.resource_manager.hardware
        logger.info(f"CPU: {hardware.logical_cores} logical cores ({hardware.physical_cores} physical)")
        logger.info(f"Memory: {hardware.total_memory_gb:.1f}GB total, {hardware.available_memory_gb:.1f}GB available")
        if hardware.has_cuda:
            for i, gpu in enumerate(hardware.gpu_info):
                logger.info(f"GPU {i}: {gpu['name']}, {gpu['free_memory_gb']:.1f}GB/{gpu['total_memory_gb']:.1f}GB")
        elif hardware.has_mps:
            logger.info("GPU: Apple Silicon with MPS support")
        else:
            logger.info("GPU: None detected")
        logger.info(f"Embedding device: {self.embedder.device}, batch size: {self.embedder.batch_size}")
        logger.info(f"LLM GPU layers: {self.gpu_info.get('n_gpu_layers', 'CPU only')}")

    def _detect_gpu_capabilities(self):
        self.gpu_info = {"available": False, "type": "none", "memory_mb": 0, "n_gpu_layers": 0}
        if self.force_cpu:
            logger.info("Forced CPU mode, skipping GPU detection")
            return
        if platform.system() == "Darwin" and platform.processor() == "arm":
            self.gpu_info = {"available": True, "type": "metal", "memory_mb": 0, "n_gpu_layers": -1}
            logger.info("Detected Apple Silicon, using Metal for all layers")
            return
        nvidia_info = get_nvidia_gpu_info()
        if nvidia_info:
            free_vram_gb = nvidia_info["free_memory_mb"] / 1024
            n_gpu_layers = -1
            if free_vram_gb < 4:
                n_gpu_layers = 20
            elif free_vram_gb < 8:
                n_gpu_layers = 35
            self.gpu_info = {"available": True, "type": "cuda", "name": nvidia_info["name"], "memory_mb": nvidia_info["free_memory_mb"], "n_gpu_layers": n_gpu_layers}
            logger.info(f"Detected NVIDIA GPU: {nvidia_info['name']} with {free_vram_gb:.1f}GB free VRAM")
            logger.info(f"Using {n_gpu_layers if n_gpu_layers != -1 else 'all'} GPU layers")
            return
        logger.info("No compatible GPU detected, CPU-only inference")

    def _get_llm(self):
        if self.llm_instance is None:
            logger.info(f"Loading LLM from {self.model_path}")
            n_threads = max(1, min(self.resource_manager.hardware.physical_cores - 1, 8)) if self.auto_optimize else min(os.cpu_count() or 4, 8)
            n_gpu_layers = 0
            if self.gpu_info["available"] and not self.force_cpu:
                n_gpu_layers = self.gpu_info.get("n_gpu_layers", 0)
                if self.gpu_info["type"] == "cuda":
                    logger.info(f"Using CUDA with {n_gpu_layers if n_gpu_layers != -1 else 'all'} GPU layers")
                elif self.gpu_info["type"] == "metal":
                    logger.info("Using Metal for all layers")
            else:
                logger.info("Using CPU-only inference")
            try:
                chat_template = """
                {%- if messages[0]['role'] == 'system' -%}
                {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' -}}
                {%- endif -%}
                {%- for message in messages[1:] -%}
                {{- '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' -}}
                {%- endfor -%}
                {{- '<|im_start|>assistant\\n' -}}
                """
                self.llm_instance = Llama(
                    model_path=self.model_path,
                    n_ctx=self.context_length,
                    n_threads=n_threads,
                    n_gpu_layers=n_gpu_layers,
                    verbose=self.verbose,
                    chat_format="custom",
                    chat_template=chat_template,
                )
                logger.info(f"Model loaded with context {self.context_length}")
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                raise RuntimeError(f"Failed to load LLM: {str(e)}")
        return self.llm_instance

    def ingest_temp_dir(self) -> None:
        """Process all documents from the project's temp directory."""
        project_root = find_project_root()
        temp_dir = os.path.join(project_root, "temp")
        if not os.path.exists(temp_dir):
            logger.info(f"Temp directory {temp_dir} not found.")
            return
        logger.info(f"Ingesting files from {temp_dir}...")
        results = process_directory(temp_dir, recursive=True, chunker=self.chunker)
        total_chunks = 0
        for file_path, chunks in results.items():
            if self.persist and self.vectordb.is_document_processed(file_path):
                logger.info(f"File already processed: {file_path}")
                continue
            try:
                self.vectordb.add_document_chunks(file_path, chunks)
                total_chunks += len(chunks)
                logger.info(f"Added {len(chunks)} chunks from {file_path}")
            except Exception as e:
                logger.error(f"Error adding {file_path} to vectordb: {e}")
        logger.info(f"Ingested {len(results)} files with {total_chunks} total chunks")

    def _analyze_query_intent(self, query: str) -> Dict[str, Union[bool, str]]:
        intent = {"has_greeting": False, "information_request": None, "requires_rag": True}
        greet_pats = [r"\bhello\b", r"\bhi\b", r"\bhey\b", r"\bgreetings\b"]
        if any(re.search(p, query.lower()) for p in greet_pats):
            intent["has_greeting"] = True
        if len(query.split()) > 2:
            intent["information_request"] = query
        return intent

    def _build_prompt(self, query: str, context: str, intent: dict) -> str:
        sys_msg = "You are a helpful AI assistant. Answer based on the provided context."
        logger.info(f"Context length: {len(context)}")
        if context:
            logger.info(f"Context preview: {context[:100]}...")
        prompt = f"<|im_start|>system\n{sys_msg}<|im_end|>\n<|im_start|>user\n"
        if context.strip():
            prompt += f"Context information:\n{context}\n\n"
        else:
            logger.warning("No context for prompt injection.")
        prompt += f"Question: {query}\n<|im_end|>\n<|im_start|>assistant\n"
        logger.info(f"Final prompt (truncated): {prompt[:300]}...")
        return prompt

    def query(self, query_text: str, show_retrieved_chunks: bool = True, debug_mode: bool = False) -> Dict[str, Any]:
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
            results = self.vectordb.query(query_text, show_retrieved_chunks=show_retrieved_chunks)
            logger.info(f"Retrieved {len(results.get('documents', []))} documents")
            for i, doc_text in enumerate(results.get("documents", [])):
                score = results.get("scores", [0] * len(results.get("documents", [])))[i]
                final_context += f"[Doc {i+1} (score={score:.2f})]\n{doc_text}\n\n"
            debug_chunks = []
            for i, doc_text in enumerate(results.get("documents", [])):
                debug_chunks.append({
                    "id": results.get("ids", [])[i],
                    "score": results.get("scores", [])[i],
                    "metadata": results.get("metadatas", [])[i],
                    "text": (doc_text[:200] + "...") if len(doc_text) > 200 else doc_text
                })
            debug_info["chunks"] = debug_chunks
            if show_retrieved_chunks:
                lines = [f"Retrieved {len(results.get('documents', []))} relevant chunks:\n"]
                for i, doc_text in enumerate(results.get("documents", [])):
                    sc = results.get("scores", [])[i]
                    snippet = (doc_text[:200] + "...") if len(doc_text) > 200 else doc_text
                    source = results.get("metadatas", [{}] * len(results.get("documents", [])))[i].get("source", "N/A")
                    lines.append(f"Chunk {i+1} - score {sc:.2f}")
                    lines.append(f"  Source: {source}")
                    lines.append("  " + snippet)
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
            completion = llm(
                prompt,
                max_tokens=512,
                temperature=0.7,
                top_p=0.9,
                repeat_penalty=1.1,
                stop=["<|im_end|>", "<|endoftext|>", "<|im_start|>"],
                echo=False,
                stream=False,
            )
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
        except Exception as ex:
            logger.error(f"Error generating: {ex}")
            response = f"Error generating response: {str(ex)}"
            debug_info["generation_error"] = str(ex)
        generation_time = time.time() - gen_start
        total_time = time.time() - start_t
        logger.info(f"Generated response in {generation_time:.2f}s, total {total_time:.2f}s")
        debug_info["generation_time"] = generation_time
        debug_info["total_time"] = total_time
        log_file = log_query(query_text, debug_info.get("chunks", []), response, debug_info)
        logger.info(f"Query log saved to {log_file}")
        if debug_mode:
            print("\nChunk Sources:")
            for chunk in debug_info.get("chunks", []):
                source = chunk.get("metadata", {}).get("source", "N/A")
                print(f"  {chunk['id']}: {source}")
            return {"response": response, "debug_info": debug_info, "retrieved_display": retrieved_display}
        else:
            return {"response": response, "debug_info": {}, "retrieved_display": retrieved_display}

    def add_document(self, file_path: str) -> int:
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return 0
        if self.persist and self.vectordb.is_document_processed(file_path):
            logger.info(f"File already processed: {file_path}")
            ccount = sum(1 for m in self.vectordb.document_metadata if m.get("source") == file_path)
            return ccount
        chunks = list(self.chunker.process_file(file_path))
        try:
            self.vectordb.add_document_chunks(file_path, chunks)
            return len(chunks)
        except Exception as e:
            logger.error(f"Error adding document: {e}")
            return 0

    def add_documents_from_directory(self, directory_path: str, recursive: bool = True) -> int:
        results = process_directory(directory_path, recursive=recursive, chunker=self.chunker)
        total_chunks = 0
        for file_path, chunks in results.items():
            try:
                if self.persist and self.vectordb.is_document_processed(file_path):
                    logger.info(f"File already processed: {file_path}")
                    ccount = sum(1 for m in self.vectordb.document_metadata if m.get("source") == file_path)
                    total_chunks += ccount
                    continue
                self.vectordb.add_document_chunks(file_path, chunks)
                total_chunks += len(chunks)
                logger.info(f"Added {len(chunks)} chunks from {os.path.basename(file_path)}")
            except Exception as e:
                logger.error(f"Error adding {file_path} to vectordb: {e}")
        return total_chunks

    def add_web_content(self, url: str) -> int:
        from .extractor import extract_text_with_jina, save_to_project_tempdir
        logger.info(f"Processing URL: {url}")
        try:
            content = extract_text_with_jina(url)
            if not content:
                logger.error(f"Failed to extract content from URL: {url}")
                return 0
            temp_file = save_to_project_tempdir(content, url)
            logger.info(f"Saved content to {temp_file}")
            return self.add_document(temp_file)
        except Exception as e:
            logger.error(f"Error adding web content from {url}: {e}")
            return 0

    def close(self) -> None:
        if hasattr(self, 'vectordb'):
            self.vectordb.close()
        if hasattr(self, 'embedder'):
            self.embedder.close()
        self.llm_instance = None
        gc.collect()
        logger.info("Resources cleaned up.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def main():
    parser = argparse.ArgumentParser("LlamaSearch advanced RAG system")
    parser.add_argument("--document", type=str, default=None,
                        help="Path to document or directory of documents to add")
    parser.add_argument("--query", type=str, required=True,
                        help="Query to run against the documents")
    parser.add_argument("--url", type=str, default=None,
                        help="URL to retrieve and add to the knowledge base")
    parser.add_argument("--persist", action="store_true",
                        help="Persist the vector database between runs (do not clear data)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode with additional info")
    parser.add_argument("--custom-model", type=str, default=None,
                        help="Path to a custom model file")
    parser.add_argument("--force-cpu", action="store_true",
                        help="Force CPU-only inference even if GPU is available")
    parser.add_argument("--workers", type=int, default=None,
                        help="Maximum number of worker threads (default: auto)")
    parser.add_argument("--recursive", action="store_true",
                        help="Recursively process subdirectories")
    args = parser.parse_args()

    # Only clear the data directory if persist is NOT set.
    if not args.persist:
        clear_data_directory()

    llm = LlamaSearch(
        persist=args.persist,
        custom_model_path=args.custom_model,
        force_cpu=args.force_cpu,
        max_workers=args.workers,
        debug=args.debug
    )
    try:
        if not args.document and not args.url:
            logger.info("No --document or --url provided. Ingesting from 'temp' directory by default.")
            llm.ingest_temp_dir()
        if args.document:
            from pathlib import Path
            doc_path = Path(args.document)
            if doc_path.is_dir():
                logger.info(f"Processing directory: {doc_path}")
                added = llm.add_documents_from_directory(str(doc_path), recursive=args.recursive)
                logger.info(f"Added a total of {added} chunks from directory")
            elif doc_path.exists():
                c = llm.add_document(str(doc_path))
                logger.info(f"Added {c} chunks from {doc_path.name}")
            else:
                logger.error(f"Document path not found: {args.document}")
        if args.url:
            logger.info(f"Processing URL: {args.url}")
            added = llm.add_web_content(args.url)
            logger.info(f"Added {added} chunks from URL")
        if args.query:
            st = time.time()
            result = llm.query(args.query, debug_mode=args.debug)
            response = result.get("response", "")
            display = result.get("retrieved_display", "")
            if display.strip():
                print(display)
            else:
                print("No retrieved chunks to display.")
            if args.debug:
                debug_info = result.get("debug_info", {})
                print("\nDebug info:\n")
                for key, value in debug_info.items():
                    if key != "chunks":
                        print(f"  {key}: {value}")
                print("\nChunk Sources:")
                for chunk in debug_info.get("chunks", []):
                    source = chunk.get("metadata", {}).get("source", "N/A")
                    print(f"  {chunk['id']}: {source}")
                print("\nResponse:\n")
            else:
                print("\nResponse:\n")
            if response.strip():
                print(response)
            else:
                print("[No response text returned by LLM.]")
            print(f"\nQuery took {time.time() - st:.2f}s")
    finally:
        llm.close()

if __name__ == "__main__":
    main()
