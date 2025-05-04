# src/llamasearch/core/llmsearch.py

import os
import time
import gc
from pathlib import Path
from typing import Dict, Any

from llamasearch.utils import setup_logging
from llamasearch.core.vectordb import VectorDB
from llamasearch.core.embedder import EnhancedEmbedder
from llamasearch.core.chunker import process_directory

from killeraiagent.models import LLM

logger = setup_logging(__name__)

def convert_to_markdown_if_needed(file_path: Path) -> Path:
    """
    If docx/pdf => convert with pandoc. Return final path (md or original).
    """
    import subprocess
    ext = file_path.suffix.lower()
    if ext in [".md", ".markdown"]:
        return file_path
    elif ext in [".docx", ".doc", ".pdf"]:
        mdfile = file_path.with_suffix(".md")
        cmd = ["pandoc", str(file_path), "-o", str(mdfile)]
        logger.info(f"Converting {file_path.name} => {mdfile.name} with {cmd}")
        try:
            res = subprocess.run(cmd, capture_output=True)
            if res.returncode != 0:
                logger.error(f"Conversion error => {res.stderr.decode('utf-8','ignore')}")
                return file_path
            return mdfile
        except Exception as e:
            logger.error(f"Conversion except => {e}")
            return file_path
    else:
        logger.warning(f"No known conversion rule for {file_path.suffix}")
        return file_path

class LLMSearch:
    """
    A universal RAG-based search class with new add_document method for single-file indexing.
    """

    def __init__(
        self,
        model: LLM,
        storage_dir: Path,
        models_dir: Path,
        verbose: bool = True,
        context_length: int = 4096,
        max_results: int = 3,
        auto_optimize: bool = True,
        embedder_batch_size: int = 8,
        force_cpu: bool = False,
        max_workers: int = 1,
        debug: bool = False
    ):
        self.verbose = verbose
        self.context_length = context_length
        self.max_results = max_results
        self.auto_optimize = auto_optimize
        self.embedder_batch_size = embedder_batch_size
        self.force_cpu = force_cpu
        self.max_workers = max_workers
        self.debug = debug
        self.model = model

        self.models_dir = models_dir
        self.storage_dir = storage_dir
        os.makedirs(self.models_dir, exist_ok=True)

        self.logger = logger
        if self.force_cpu:
            self.device = "cpu"

        self.embedder = EnhancedEmbedder(
            device=self.device,
            batch_size=self.embedder_batch_size,
            auto_optimize=self.auto_optimize,
            num_workers=self.max_workers,
        )
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
        if self.model:
            self.logger.info(f"Using LLM => {self.model.model_info.model_id}")
        else:
            raise ValueError("No LLM provided.")
        
    def add_document(self, file_path: Path) -> int:
        """
        Single-file indexing. Convert if needed, chunk, pass to vectordb.add_document_chunks.
        """
        if not file_path.exists():
            self.logger.error(f"File not found => {file_path}")
            return 0
        final_path = convert_to_markdown_if_needed(file_path)
        if final_path.is_dir():
            # if conversion created a directory? Unlikely. Just skip
            return 0
        
        # chunk
        results = process_directory(
            directory_path=final_path.parent,
            markdown_chunker=self.vectordb.markdown_chunker,
            html_chunker=self.vectordb.html_chunker
        )
        if str(final_path) not in results:
            self.logger.warning(f"No chunks for => {final_path}")
            return 0
        chunks = results[str(final_path)]
        added = self.vectordb.add_document_chunks(str(final_path), chunks)
        return added

    def add_documents_from_directory(self, directory_path: Path) -> int:
        """
        For each file in dir, do the doc->md conversion if needed, then chunk + add.
        We'll accumulate total chunks added.
        """
        if not directory_path.is_dir():
            self.logger.warning(f"{directory_path} is not a directory.")
            return 0
        total_chunks = 0
        for root, dirs, files in os.walk(directory_path):
            rp = Path(root)
            for fn in files:
                p = rp / fn
                total_chunks += self.add_document(p)
        return total_chunks

    def llm_query(self, query_text: str, debug_mode: bool=False) -> Dict[str, Any]:
        """
        RAG-based retrieval + LLM generation
        """
        debug_info: Dict[str, Any] = {}
        final_context = ""
        retrieved_display = ""

        try:
            results = self.vectordb.vectordb_query(query_text, max_out=self.max_results)
            docs = results.get("documents",[])
            metas= results.get("metadatas",[])
            if not docs:
                return {"response":"No relevant context found.","formatted_response":"No relevant context found."}
            for i, doc_text in enumerate(docs):
                sc = results["scores"][i] if i<len(results["scores"]) else 0
                src = metas[i].get("source","N/A")
                final_context += f"[Doc {i+1}, src={src}, sc={sc:.2f}]\n{doc_text}\n\n"
                retrieved_display += f"Chunk {i+1} => Source={src}\nContent: {doc_text}\n\n"
        except Exception as e:
            return {"response":f"Error => {e}","formatted_response":f"Error => {e}"}

        system_instruction = "You are a helpful AI. Use context, no fabrications."
        prompt = (
            f"<|im_start|>system\n{system_instruction}<|im_end|>\n"
            f"<|im_start|>user\nContext:\n{final_context}\n\nQ: {query_text}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        gen_start= time.time()
        text_response, raw = self.model.generate(
            prompt=prompt,
            max_tokens=512,
            temperature=0.7,
            top_p=0.9,
            repeat_penalty=1.1
        )
        gen_time= time.time()-gen_start
        if debug_mode:
            debug_info["gen_time"]= gen_time
            debug_info["retrieved_display"]= retrieved_display
        formatted = f"## AI Summary\n{text_response}\n\n## Retrieved Chunks\n{retrieved_display}"
        return {
            "response": text_response,
            "debug_info": debug_info,
            "retrieved_display": retrieved_display,
            "formatted_response": formatted
        }

    def close(self)->None:
        if hasattr(self,'model'):
            del self.model
        if hasattr(self,'embedder'):
            self.embedder.close()
            del self.embedder
        if hasattr(self,'vectordb'):
            self.vectordb.close()
            del self.vectordb
        gc.collect()

    def __enter__(self):
        return self
    def __exit__(self,exc_type,exc_val,exc_tb):
        self.close()
