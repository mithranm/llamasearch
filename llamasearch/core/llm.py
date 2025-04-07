import os
import re
import time
import gc
import argparse
import shutil

from typing import Dict, Any
from llama_cpp import Llama
from pathlib import Path

from llamasearch.setup_utils import find_project_root
from llamasearch.utils import setup_logging, log_query
from llamasearch.core.vectordb import VectorDB
from llamasearch.core.embedder import EnhancedEmbedder
from llamasearch.core.resource_manager import get_resource_manager

import json  # Added to dump query JSON

logger = setup_logging(__name__)

DEFAULT_MODEL_NAME = "qwen2.5-1.5b-instruct-q4_k_m"

class LlamaSearch:
    """
    LlamaSearch integrates retrieval (vector/BM25/graph) with an LLM for RAG.
    Document ingestion is managed via VectorDB.
    
    This version:
      - Triggers ingestion from the crawl_data/raw directory if --persist is not used.
      - Uses the VectorDB's built-in chunker.
      - Constructs a citation string (retrieved_display) for user visibility.
      - Enhanced handling for name queries like "tell me about Georgi"
    """
    def __init__(
        self,
        storage_dir: Path,
        model_name: str = DEFAULT_MODEL_NAME,
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
        self.resource_manager = get_resource_manager(auto_optimize=self.auto_optimize)
        self.device = self.resource_manager.get_embedding_config()['device']
        
        self.models_dir = os.path.join(storage_dir, "models")
        self.storage_dir = storage_dir
        os.makedirs(self.models_dir, exist_ok=True)

        if custom_model_path:
            self.model_path = custom_model_path
            logger.info(f"Using custom model at: {self.model_path}")
        else:
            self.model_path = os.path.join(self.models_dir, f"{self.model_name}.gguf")
            logger.info(f"Using model: {self.model_name} ({self.model_path})")

        if not os.path.exists(self.model_path):
            logger.warning(f"Model not found at {self.model_path}")
            logger.info("Please download or provide a valid model path.")

        embedding_config = {}
        if self.auto_optimize:
            embedding_config = self.resource_manager.get_embedding_config()

        # Determine embedding device based on resource_manager's hardware info.
        embedding_device = "cpu"
        if not force_cpu:
            embedding_device = "cuda" if self.resource_manager.hardware.has_cuda else (
                "mps" if self.resource_manager.hardware.has_mps else "cpu"
            )

        self.embedder = EnhancedEmbedder(
            device=embedding_device,
            batch_size=self.embedder_batch_size,
            auto_optimize=self.auto_optimize,
            num_workers=self.max_workers,
            embedding_config=embedding_config,
        )

        # Initialize VectorDB (which instantiates its own chunkers).
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

        logger.info(f"Initialized LlamaSearch with model: {self.model_name}")
        logger.info(f"Model path: {self.model_path}")
        logger.info(f"Auto-optimize: {self.auto_optimize}")
        logger.info(f"Chunks for retrieval: {self.max_results}")

    def _get_llm(self):
        if self.llm_instance is None:
            logger.info(f"Loading LLM from {self.model_path}")
            llm_config = self.resource_manager.get_llm_config() if self.auto_optimize else {}
            n_threads = llm_config.get("n_threads", 4)
            n_gpu_layers = llm_config.get("n_gpu_layers", 0)
            kwargs = {
                "model_path": self.model_path,
                "n_ctx": self.context_length,
                "n_threads": n_threads,
                "verbose": self.verbose,
            }
            if self.resource_manager.hardware.has_cuda and not self.force_cpu:
                kwargs["n_gpu_layers"] = n_gpu_layers
            chat_template = (
                "{%- if messages[0]['role'] == 'system' -%}"
                "<|im_start|>system\n{{ messages[0]['content'] }}<|im_end|>\n"
                "{%- endif -%}"
                "{%- for message in messages[1:] -%}"
                "<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n"
                "{%- endfor -%}"
                "<|im_start|>assistant\n"
            )
            kwargs["chat_format"] = "custom"
            kwargs["chat_template"] = chat_template
            try:
                self.llm_instance = Llama(**kwargs)
                logger.info(f"LLM loaded with context length {self.context_length}")
            except Exception as e:
                logger.error(f"Error loading LLM: {e}")
                raise RuntimeError(f"Failed to load LLM: {str(e)}")
        return self.llm_instance

    def ingest_crawl_data(self) -> None:
        """
        Ingests all documents from the crawl_data/raw directory.
        Processes each file with the VectorDB's chunker and indexes the resulting chunks.
        """
        project_root = find_project_root()
        crawl_data_dir = os.path.join(project_root, "crawl_data", "raw")
        if not os.path.exists(crawl_data_dir):
            logger.info(f"Crawl data directory {crawl_data_dir} not found.")
            return
        logger.info(f"Ingesting files from {crawl_data_dir}")
        self.add_documents_from_directory(crawl_data_dir)

    def llm_query(self, query_text: str, debug_mode: bool = False) -> Dict[str, Any]:
        logger.info(f"Query: {query_text}")
        start_t = time.time()
        debug_info: Dict[str, Any] = {}
        
        # Use VectorDB's NER-based named entity detection
        is_entity_query, target_entity = self.vectordb.is_named_entity_query(query_text)
        
        intent = {
            "has_greeting": bool(re.search(r"\b(hello|hi|hey|greetings)\b", query_text.lower())),
            "information_request": query_text if len(query_text.split()) > 2 else None,
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
            # Adjust max_results for entity queries to improve recall
            effective_max_results = self.max_results
            if is_entity_query:
                # For entity queries, we want more results to ensure we find relevant information
                effective_max_results = min(self.max_results + 3, 6)
                logger.info(f"Entity query detected for '{target_entity}', using {effective_max_results} max results")
            
            results = self.vectordb.vectordb_query(
                query_text, 
                max_results=effective_max_results
            )
            
            if debug_mode:
                logger.info("Query JSON: " + json.dumps(results, indent=2))
                
            docs = results.get("documents", [])
            if not docs:
                return {"response": "No relevant context found.", "debug_info": debug_info, "retrieved_display": ""}
                
            logger.info(f"Retrieved {len(docs)} documents")
            
            # Check if target entity is found in results (for entity queries)
            entity_found = False
            if is_entity_query and target_entity:
                for doc in docs:
                    if target_entity.lower() in doc.lower():
                        entity_found = True
                        logger.info(f"Target entity '{target_entity}' found in retrieved documents")
                        break
                        
                if not entity_found:
                    logger.warning(f"Target entity '{target_entity}' NOT found in retrieved documents")
            
            # Format context with scores
            for i, doc_text in enumerate(docs):
                score = results.get("scores", [0] * len(docs))[i]
                final_context += f"[Doc {i+1} (score={score:.2f})]\n{doc_text}\n\n"
                
            metas = results.get("metadatas", [])
            for i, meta in enumerate(metas):
                source = meta.get("source", "N/A") if isinstance(meta, dict) else "N/A"
                content = docs[i]
                retrieved_display += f"Chunk {i+1} - Source: {source}\nContent: {content}\n\n"
                
            debug_info["chunks"] = [
                {
                    "id": results.get("ids", [])[i] if i < len(results.get("ids", [])) else f"doc_{i}",
                    "score": results.get("scores", [])[i] if i < len(results.get("scores", [])) else 0.0,
                    "metadata": meta,
                    "text": docs[i]
                } for i, meta in enumerate(metas)
            ]
            
            debug_info["retrieval_time"] = time.time() - start_t
            logger.info(f"Context retrieved in {debug_info['retrieval_time']:.2f}s")
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            debug_info["retrieval_error"] = str(e)
        
        # Enhanced prompt for entity queries with explicit target highlighting
        system_instruction = "You are a helpful AI assistant. Answer based on the provided context."
        
        if is_entity_query and target_entity:
            # Enhanced system instruction for entity queries
            system_instruction = (
                f"You are a helpful AI assistant. Answer based on the provided context. "
                f"Focus specifically on information about '{target_entity}', their details, background, "
                f"achievements, and any other relevant information. "
                f"If '{target_entity}' is mentioned in the context, make sure to include this information "
                f"in your response, even if it seems minimal."
            )
            
            # Verification for debugging
            if entity_found:
                logger.info(f"Confirmed '{target_entity}' is present in context documents")
            else:
                logger.warning(f"'{target_entity}' may not be explicitly mentioned in context")
        
        prompt = (
            f"<|im_start|>system\n{system_instruction}<|im_end|>\n"
            f"<|im_start|>user\nContext information:\n{final_context}\n\nQuestion: {query_text}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        
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
                
                # For entity queries, verify the entity is mentioned in the response
                if is_entity_query and target_entity and entity_found:
                    if target_entity.lower() not in response.lower():
                        logger.warning(f"Entity '{target_entity}' found in docs but not in response")
                        # Add a note if the entity is missing from response
                        if not response.startswith("I couldn't find"):
                            response = f"The information about {target_entity} in the context includes: {response}"
            else:
                response = "Unexpected response format from LLM."
                logger.error(f"Unexpected response format: {completion}")
        except Exception as ex:
            logger.error(f"Error generating response: {ex}")
            response = f"Error generating response: {str(ex)}"
            debug_info["generation_error"] = str(ex)
            
        generation_time = time.time() - gen_start
        total_time = time.time() - start_t
        logger.info(f"Generated response in {generation_time:.2f}s, total {total_time:.2f}s")
        debug_info["generation_time"] = generation_time
        debug_info["total_time"] = total_time
        
        # Use the optimized log_query function that saves disk space
        log_file = log_query(
            query_text, 
            debug_info.get("chunks", []), 
            response, 
            debug_info,
            full_logging=debug_mode  # Only use full logging in debug mode
        )
        logger.info(f"Query log saved to {log_file}")
        
        if debug_mode:
            print("\nChunk Sources:")
            for chunk in debug_info.get("chunks", []):
                source = chunk.get("metadata", {}).get("source", "N/A")
                print(f"  {chunk['id']}: {source}")
            return {"response": response, "debug_info": debug_info, "retrieved_display": retrieved_display}
        else:
            return {"response": response, "debug_info": {}, "retrieved_display": retrieved_display}

    def add_documents_from_directory(self, directory_path: str) -> int:
        from llamasearch.core.chunker import process_directory
        results = process_directory(directory_path=directory_path, markdown_chunker=self.vectordb.markdown_chunker, html_chunker=self.vectordb.html_chunker)
        total_chunks = 0
        for file_path, chunks in results.items():
            try:
                if self.vectordb.is_document_processed(file_path):
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

    def close(self) -> None:
        if hasattr(self.embedder, "close"):
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
    parser.add_argument("--query", type=str, required=True, help="Query to run against the documents")
    parser.add_argument("--persist", action="store_true", help="Persist the vector database between runs (do not clear data)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with additional info")
    parser.add_argument("--custom-model", type=str, default=None, help="Path to a custom model file")
    parser.add_argument("--force-cpu", action="store_true", help="Force CPU-only inference even if GPU is available")
    parser.add_argument("--workers", type=int, default=1, help="Maximum number of worker threads (default: auto)")
    parser.add_argument("--recursive", action="store_true", help="Recursively process subdirectories")
    args = parser.parse_args()
    
    storage_dir = Path(os.path.join(find_project_root(), "index"))
    
    st = time.time()
    if args.persist:
        logger.info("Persisting vector database")
    else:
        logger.info(f"Clearing vector database: {storage_dir}")

        # Check if the directory exists before attempting to delete it
        if os.path.exists(storage_dir):
            logger.info(f"Vector database directory exists at: {storage_dir}")

            # Log the contents of the directory
            try:
                contents = os.listdir(storage_dir)
                logger.info(f"Contents of {storage_dir}: {contents}")
            except Exception as e:
                logger.error(f"Error listing contents of {storage_dir}: {e}")

            try:
                shutil.rmtree(storage_dir, ignore_errors=False)
                logger.info(f"Successfully cleared vector database at {storage_dir}")
            except Exception as e:
                logger.error(f"Error clearing vector database at {storage_dir}: {e}")

            # Check if the directory still exists after attempting to delete it
            if os.path.exists(storage_dir):
                logger.warning(f"Vector database directory STILL EXISTS at {storage_dir} after deletion attempt!")
            else:
                logger.info(f"Vector database directory successfully deleted at {storage_dir}")
        else:
            logger.info(f"Vector database directory does not exist at: {storage_dir}, skipping deletion.")
    
    llm = LlamaSearch(
        custom_model_path=args.custom_model,
        force_cpu=args.force_cpu,
        max_workers=args.workers,
        debug=args.debug,
        storage_dir=storage_dir
    )
    llm.ingest_crawl_data()
    result = llm.llm_query(args.query, debug_mode=args.debug)
    response = result.get("response", "")
    retrieved_display = result.get("retrieved_display", "")
    if retrieved_display.strip():
        print(retrieved_display)
    else:
        print("No retrieved chunks to display.")
    if args.debug:
        debug_info = result.get("debug_info", {})
        print("\nDebug info:\n")
        for key, value in debug_info.items():
            if key != "chunks":
                logger.info(f"  {key}: {value}")
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

if __name__ == "__main__":
    main()