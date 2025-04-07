import os
import re
import time
import gc
import argparse
import shutil
import json
from typing import Dict, Any, List

from llamasearch.core.teapotai import TeapotAI, TeapotAISettings
from transformers import AutoTokenizer

from llamasearch.setup_utils import find_project_root
from llamasearch.utils import setup_logging, log_query
from llamasearch.core.vectordb import VectorDB
from llamasearch.core.embedder import EnhancedEmbedder
from llamasearch.core.resource_manager import get_resource_manager

logger = setup_logging(__name__)

class TeapotSearch:
    """
    TeapotSearch integrates sophisticated retrieval (vector/BM25/graph) with the lightweight TeapotLLM.
    Optimized for CPU usage and improved entity recognition.
    """
    def __init__(
        self,
        verbose: bool = True,
        max_results: int = 3,
        auto_optimize: bool = False,
        embedder_batch_size: int = 8,
        force_cpu: bool = True,
        max_workers: int = 4,
        debug: bool = False,
    ):
        self.verbose = verbose
        self.max_results = max_results
        self.auto_optimize = auto_optimize
        self.embedder_batch_size = embedder_batch_size
        self.force_cpu = force_cpu
        self.max_workers = max_workers
        self.debug = debug
        self.resource_manager = get_resource_manager(auto_optimize=self.auto_optimize)
        self.device = self.resource_manager.get_embedding_config()['device']
        
        project_root = find_project_root()
        self.storage_dir = os.path.join(project_root, "index")

        embedding_config = {}
        if self.auto_optimize:
            embedding_config = self.resource_manager.get_embedding_config()

        # Determine embedding device based on resource_manager's hardware info
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

        # Initialize VectorDB (which instantiates its own chunkers)
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

        # Initialize tokenizer separately for token counting
        self.tokenizer = AutoTokenizer.from_pretrained("teapotai/teapotllm")
        
        # Initialize TeapotAI with custom settings
        self.teapot_settings = TeapotAISettings(
            use_rag=False,  # We'll handle RAG ourselves
            verbose=self.verbose,
            max_context_length=768,  # Leave some room for prompt and query
            context_chunking=True,
            log_level="info" if self.verbose else "warning"
        )
        
        self.llm_instance = None
        
        logger.info("Initialized TeapotSearch")
        logger.info(f"Auto-optimize: {self.auto_optimize}")
        logger.info(f"Chunks for retrieval: {self.max_results}")

    def _get_llm(self):
        """Initialize or return the TeapotAI instance"""
        if self.llm_instance is None:
            logger.info("Loading TeapotLLM")
            self.llm_instance = TeapotAI(settings=self.teapot_settings)
            logger.info("TeapotLLM loaded")
        return self.llm_instance

    def _extract_entities(self, text: str) -> List[str]:
        """Extract named entities from text using the VectorDB's NER pipeline"""
        try:
            if hasattr(self.vectordb, '_extract_entities_from_text'):
                return self.vectordb._extract_entities_from_text(text)
            return []
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return []

    def _count_tokens(self, text: str) -> int:
        """Count tokens in a string using TeapotLLM's tokenizer"""
        return len(self.tokenizer.encode(text))

    def _format_context_with_entities(self, context: str, query: str) -> str:
        """Format context to highlight query entities and add an entity summary"""
        try:
            # Extract entities from query and context
            query_entities = self._extract_entities(query)
            context_entities = self._extract_entities(context)
            
            # Add entity summary at the beginning
            all_entities = set(context_entities)
            entity_summary = "Important entities: " + ", ".join(all_entities) + "\n\n"
            
            # Highlight query entities in context
            highlighted_context = context
            for entity in query_entities:
                if entity in highlighted_context:
                    highlighted_context = highlighted_context.replace(entity, f"**{entity}**")
            
            # Combine with summary
            formatted_context = entity_summary + highlighted_context
            
            return formatted_context
        except Exception as e:
            logger.error(f"Error formatting context with entities: {e}")
            return context

    def _optimize_context_for_token_limit(self, docs: List[str], meta: List[Dict], query: str, max_tokens: int = 768) -> str:
        """
        Optimize the context to fit within token limits while prioritizing relevant information
        
        Args:
            docs: List of document texts
            meta: List of document metadata
            query: The query text
            max_tokens: Maximum tokens allowed for context
            
        Returns:
            Optimized context string
        """
        # Extract entities from the query to prioritize chunks with matching entities
        query_entities = self._extract_entities(query)
        
        # Create a list of (doc, meta, score) tuples for ranking
        ranked_chunks = []
        for i, (doc, meta_info) in enumerate(zip(docs, meta)):
            # Base score from retrieval
            score = meta_info.get("score", 0.5)
            
            # Entity match bonus
            entity_match_score = 0.0
            chunk_entities = meta_info.get("entities", [])
            if chunk_entities and query_entities:
                matches = set(chunk_entities) & set(query_entities)
                if matches:
                    entity_match_score = len(matches) * 0.2  # 0.2 bonus per entity match
            
            # Position bonus (prioritize earlier chunks slightly)
            position_score = max(0, 0.1 - (i * 0.02))
            
            # Calculate final score
            final_score = score + entity_match_score + position_score
            
            ranked_chunks.append((doc, meta_info, final_score))
        
        # Sort chunks by score (descending)
        ranked_chunks.sort(key=lambda x: x[2], reverse=True)
        
        # Build optimized context within token limit
        context_parts = []
        current_tokens = 0
        
        # First, add an entity summary line if we have entities
        all_entities = set()
        for _, meta_info, _ in ranked_chunks:
            chunk_entities = meta_info.get("entities", [])
            all_entities.update(chunk_entities)
        
        if all_entities:
            entity_summary = "Key entities in context: " + ", ".join(all_entities)
            entity_tokens = self._count_tokens(entity_summary)
            if entity_tokens < max_tokens * 0.1:  # Use no more than 10% for entity summary
                context_parts.append(entity_summary)
                current_tokens += entity_tokens
        
        # Prioritize chunks with query entity matches first
        query_matched_chunks = []
        other_chunks = []
        
        for doc, meta_info, score in ranked_chunks:
            chunk_entities = meta_info.get("entities", [])
            matches = set(chunk_entities) & set(query_entities)
            if matches:
                query_matched_chunks.append((doc, meta_info, score))
            else:
                other_chunks.append((doc, meta_info, score))
        
        # Process chunks in order: query-matched chunks first, then others
        for chunk_list in [query_matched_chunks, other_chunks]:
            for doc, meta_info, _ in chunk_list:
                # Check if adding this chunk would exceed our token limit
                chunk_tokens = self._count_tokens(doc)
                if current_tokens + chunk_tokens <= max_tokens:
                    # Format the chunk with source information
                    source = meta_info.get("source", "Unknown")
                    formatted_chunk = f"[Source: {os.path.basename(source)}]\n{doc}"
                    
                    # Highlight any query entities in this chunk
                    for entity in query_entities:
                        if entity in formatted_chunk:
                            formatted_chunk = formatted_chunk.replace(entity, f"**{entity}**")
                    
                    context_parts.append(formatted_chunk)
                    current_tokens += chunk_tokens
                else:
                    # If we can't fit the full chunk, see if we can fit the first few sentences
                    if current_tokens < max_tokens * 0.9:  # Only try this if we have at least 10% space left
                        sentences = re.split(r'(?<=[.!?])\s+', doc)
                        partial_chunk = ""
                        for sentence in sentences:
                            sentence_tokens = self._count_tokens(sentence + " ")
                            if current_tokens + self._count_tokens(partial_chunk) + sentence_tokens <= max_tokens:
                                partial_chunk += sentence + " "
                            else:
                                break
                        
                        if partial_chunk:
                            source = meta_info.get("source", "Unknown")
                            formatted_partial = f"[Source: {os.path.basename(source)} (partial)]\n{partial_chunk}"
                            context_parts.append(formatted_partial)
                            current_tokens += self._count_tokens(formatted_partial)
                    break
        
        # Join the context parts with double newlines
        optimized_context = "\n\n".join(context_parts)
        logger.info(f"Optimized context: {current_tokens} tokens, {len(context_parts)} chunks")
        
        return optimized_context

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
        """
        Process a query using the TeapotLLM with optimized context management and entity handling
        
        Args:
            query_text: The query text
            debug_mode: Whether to enable debug mode
            
        Returns:
            Dictionary with response and debug information
        """
        logger.info(f"Query: {query_text}")
        start_t = time.time()
        debug_info: Dict[str, Any] = {}
        intent = {
            "has_greeting": bool(re.search(r"\b(hello|hi|hey|greetings)\b", query_text.lower())),
            "information_request": query_text if len(query_text.split()) > 2 else None,
            "requires_rag": True
        }
        debug_info["intent"] = intent

        # Extract entities from query for improved retrieval and highlighting
        query_entities = self._extract_entities(query_text)
        debug_info["query_entities"] = query_entities
        
        # Retrieve relevant context using our sophisticated retrieval system
        logger.info("Retrieving context from vectordb...")
        try:
            results = self.vectordb.vectordb_query(query_text, max_results=self.max_results)
            if debug_mode:
                logger.info("Query JSON: " + json.dumps(results, indent=2))
            
            docs = results.get("documents", [])
            if not docs:
                return {"response": "No relevant context found.", "debug_info": debug_info, "retrieved_display": ""}
            
            logger.info(f"Retrieved {len(docs)} documents")
            
            # Get metadata for retrieved documents
            metas = results.get("metadatas", [])
            
            # Format retrieved chunks for display to the user
            retrieved_display = ""
            for i, (content, meta) in enumerate(zip(docs, metas)):
                source = meta.get("source", "N/A") if isinstance(meta, dict) else "N/A"
                retrieved_display += f"Chunk {i+1} - Source: {source}\nContent: {content}\n\n"
            
            # Optimize the context to fit within TeapotLLM's token limit
            optimized_context = self._optimize_context_for_token_limit(
                docs=docs,
                meta=metas,
                query=query_text,
                max_tokens=768  # Leave room for system prompt and query
            )
            
            # Save optimization info for debugging
            debug_info["optimized_context_tokens"] = self._count_tokens(optimized_context)
            debug_info["chunks"] = [
                {
                    "id": results.get("ids", [])[i] if i < len(results.get("ids", [])) else f"chunk_{i}",
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
            return {"response": f"Error retrieving information: {str(e)}", "debug_info": debug_info, "retrieved_display": ""}
        
        # Define our custom system prompt optimized for entity extraction
        system_prompt = """You are a helpful AI assistant. Answer based ONLY on the provided context. Pay special attention to any entities marked with ** or mentioned in the context. When asked about a specific person, carefully examine the entire context for ANY mention of that person, even brief mentions. If information exists but is limited, provide what is available and acknowledge its limitations. Always cite your sources."""
        
        # Generate response using TeapotLLM
        llm = self._get_llm()
        gen_start = time.time()
        response = ""
        
        try:
            # Generate the response using TeapotAI
            response = llm.query(
                query=query_text,
                context=optimized_context
            )
            
            # Check if response acknowledges important entities
            if query_entities:
                # Extract what seems to be the main entity from the query
                main_entity = query_entities[0] if query_entities else ""
                
                # Check if the main entity is mentioned in the response
                if main_entity and main_entity not in response and main_entity in optimized_context:
                    # If main entity is missing but present in context, try again with explicit mention
                    logger.info(f"Main entity {main_entity} missing from response, trying again with explicit mention")
                    
                    # Add explicit instruction to include the entity
                    enhanced_prompt = system_prompt + f" Make sure to address information about {main_entity} if present in the context."
                    
                    # Try again
                    response = llm.query(
                        query=f"Tell me about {main_entity} based on the context",
                        context=optimized_context,
                        system_prompt=enhanced_prompt
                    )
                
        except Exception as ex:
            logger.error(f"Error generating response: {ex}")
            response = f"Error generating response: {str(ex)}"
            debug_info["generation_error"] = str(ex)
        
        generation_time = time.time() - gen_start
        total_time = time.time() - start_t
        
        logger.info(f"Generated response in {generation_time:.2f}s, total {total_time:.2f}s")
        debug_info["generation_time"] = generation_time
        debug_info["total_time"] = total_time
        
        # Log the query and response for analysis
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

    def add_documents_from_directory(self, directory_path: str) -> int:
        """
        Process and add all documents from a directory to the vectordb
        
        Args:
            directory_path: Path to the directory with documents
            
        Returns:
            Number of chunks added
        """
        from llamasearch.core.chunker import process_directory
        results = process_directory(
            directory_path=directory_path, 
            markdown_chunker=self.vectordb.markdown_chunker, 
            html_chunker=self.vectordb.html_chunker
        )
        
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
        """Release resources"""
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
    parser = argparse.ArgumentParser("TeapotSearch - CPU-optimized RAG system")
    parser.add_argument("--query", type=str, required=True, help="Query to run against the documents")
    parser.add_argument("--persist", action="store_true", help="Persist the vector database between runs (do not clear data)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with additional info")
    parser.add_argument("--workers", type=int, default=1, help="Maximum number of worker threads (default: auto)")
    parser.add_argument("--recursive", action="store_true", help="Recursively process subdirectories")
    args = parser.parse_args()

    llm = TeapotSearch(
        force_cpu=True,
        max_workers=args.workers,
        debug=args.debug
    )
    
    st = time.time()
    if args.persist:
        logger.info("Persisting vector database")
        # If persisting, we assume data is already in the index.
    else:
        logger.info("Clearing vector database")
        shutil.rmtree(llm.storage_dir, ignore_errors=True)
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