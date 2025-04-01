# llamasearch/core/llm.py

import os
import argparse
import time
import gc
import logging
import re
import platform
from typing import List, Dict, Any, Tuple, Optional, Union

from llama_cpp import Llama

from ..utils import find_project_root, setup_logging, log_query
from .vectordb import VectorDB

logger = setup_logging(__name__)

class OptimizedLLM:
    """
    LLM orchestrator that uses advanced chunker + vectordb with knowledge graph and name expansions.
    Does not do batch generation (no n_batch param).
    """

    def __init__(
        self,
        model_name:str="qwen2.5-1.5b-instruct-q4_k_m",
        persist:bool=False,
        verbose:bool=True,
        context_length:int=2048,
        n_results:int=5,
        custom_model_path:Optional[str]=None
    ):
        self.verbose=verbose
        self.persist=persist
        self.context_length=context_length
        self.n_results=n_results
        self.model_name=model_name
        self.custom_model_path=custom_model_path

        project_root=find_project_root()
        self.models_dir=os.path.join(project_root,"models")
        os.makedirs(self.models_dir, exist_ok=True)

        if custom_model_path:
            self.model_path=custom_model_path
            logger.info(f"Using custom model at: {self.model_path}")
        else:
            self.model_path=os.path.join(self.models_dir,f"{model_name}.gguf")
            logger.info(f"Using model: {model_name} ({self.model_path})")

        if not os.path.exists(self.model_path):
            logger.warning(f"Model not found at {self.model_path}")
            logger.info("Please download or provide a valid model path.")

        # init vectordb
        from .embedding import Embedder
        self.storage_dir=os.path.join(project_root,"vector_db")
        self.vectordb= VectorDB(
            persist=persist,
            chunk_size=250,
            text_embedding_size=512,
            chunk_overlap=50,
            min_chunk_size=50,
            batch_size=2,
            similarity_threshold=0.25,
            storage_dir=self.storage_dir,
            use_pca=False
        )

        self._process_temp_docs()

        self.llm_instance=None
        logger.info(f"Initialized OptimizedLLM with model: {self.model_name}")
        logger.info(f"Model path: {self.model_path}")

    def _process_temp_docs(self):
        proj_root=find_project_root()
        temp_dir=os.path.join(proj_root,"temp")
        if not os.path.exists(temp_dir):
            logger.info(f"Temp dir not found: {temp_dir}")
            return
        logger.info(f"Processing docs from {temp_dir}")
        for fn in os.listdir(temp_dir):
            fp=os.path.join(temp_dir, fn)
            if os.path.isfile(fp):
                logger.info(f"Processing file: {fn}")
                try:
                    self.vectordb.add_document(fp)
                except Exception as e:
                    logger.error(f"Error processing {fn}: {e}")

    def _get_llm(self):
        if self.llm_instance is None:
            logger.info(f"Loading LLM from {self.model_path}")
            n_threads=min(os.cpu_count() or 4, 8)
            n_gpu_layers=0
            if platform.system()=="Darwin" and platform.processor()=="arm":
                n_gpu_layers=-1
                logger.info("Detected Apple Silicon, using Metal acceleration")
            try:
                # no batch generation
                self.llm_instance= Llama(
                    model_path=self.model_path,
                    n_ctx=self.context_length,
                    n_threads=n_threads,
                    n_gpu_layers=n_gpu_layers,
                    verbose=self.verbose
                )
                logger.info(f"Model loaded with context length {self.context_length}")
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                raise RuntimeError(str(e))
        return self.llm_instance

    def _analyze_query_intent(self, query:str)->Dict[str,Union[bool,str]]:
        intent={
            "has_greeting":False,
            "information_request":None,
            "requires_rag":False
        }
        greet_pats=[r"\bhello\b", r"\bhi\b", r"\bhey\b", r"\bgreetings\b"]
        if any(re.search(p, query.lower()) for p in greet_pats):
            intent["has_greeting"]=True
        if len(query.split())>3:
            intent["information_request"]=query
            intent["requires_rag"]=True
        return intent

    def _build_prompt(self, query:str, context:str, intent:dict)->str:
        sys_msg="You are Qwen, a helpful AI assistant. Provide direct answers using context if relevant."
        prompt=(
            f"<|im_start|>system\n{sys_msg}<|im_end|>\n"
            f"<|im_start|>user\n"
        )
        if intent["requires_rag"] and context.strip():
            prompt+=f"Context:\n{context}\n\n"
        prompt+=f"Q: {query}\n<|im_end|>\n<|im_start|>assistant\n"
        return prompt

    def query(self, query_text:str, show_retrieved_chunks=True, debug_mode=False)->Union[str,Tuple[str,Dict]]:
        logger.info(f"Query: {query_text}")
        start_t=time.time()
        debug_info={}
        retrieved_display=""

        intent=self._analyze_query_intent(query_text)
        debug_info["intent"]=intent

        final_context=""
        if intent["requires_rag"]:
            logger.info("Retrieving context from vectordb...")
            st=time.time()
            try:
                results=self.vectordb.query(query_text, n_results=self.n_results)
                # top 3 for final context
                final_context=""
                for i, doc_text in enumerate(results["documents"][:3]):
                    final_context+= doc_text+"\n\n---\n\n"
                chunk_data=[]
                for i, doc_text in enumerate(results["documents"]):
                    chunk_data.append({
                        "id": results["ids"][i],
                        "score": results["scores"][i],
                        "metadata": results["metadatas"][i],
                        "text": doc_text[:200]+"..." if len(doc_text)>200 else doc_text
                    })
                debug_info["chunks"]=chunk_data
                if show_retrieved_chunks:
                    lines=[f"Retrieved {len(results['documents'])} relevant chunks:\n"]
                    for i, doc_text in enumerate(results["documents"]):
                        sc=results["scores"][i]
                        lines.append(f"Chunk {i+1} - score {sc:.2f}")
                        lines.append("  "+ (doc_text[:200]+"..." if len(doc_text)>200 else doc_text))
                        lines.append("")
                    retrieved_display="\n".join(lines)
                debug_info["retrieval_time"]= time.time()-st
                logger.info(f"Context retrieved in {debug_info['retrieval_time']:.2f}s")
            except Exception as e:
                logger.error(f"Error retrieving context: {e}")
                debug_info["retrieval_error"]=str(e)

        prompt=self._build_prompt(query_text, final_context, intent)
        token_est=len(prompt.split())*2
        logger.info(f"Estimated prompt tokens: {token_est}")

        llm=self._get_llm()
        gen_start=time.time()
        response=""
        try:
            out=llm(
                prompt,
                max_tokens=128,
                temperature=0.7,
                top_p=0.9,
                repeat_penalty=1.1,
                stop=["<|im_end|>","<|endoftext|>","<|im_start|>"],
                echo=False
            )
            response= out["choices"][0]["text"].strip()
        except Exception as e:
            logger.error(f"Error generating: {e}")
            response=f"Error generating: {str(e)}"
            debug_info["generation_error"]=str(e)

        generation_time=time.time()-gen_start
        total_time=time.time()-start_t
        logger.info(f"Generated response in {generation_time:.2f}s, total {total_time:.2f}s")
        debug_info["generation_time"]=generation_time
        debug_info["total_time"]=total_time

        from ..utils import log_query
        log_file= log_query(query_text, debug_info.get("chunks",[]), response, debug_info)
        logger.info(f"Query log saved to {log_file}")

        if debug_mode:
            return (response, debug_info, retrieved_display)
        else:
            return (response, retrieved_display)

    def close(self):
        self.vectordb.close()
        self.llm_instance=None
        gc.collect()
        logger.info("Resources cleaned up")

def main():
    import sys
    import argparse
    parser=argparse.ArgumentParser("LlamaSearch advanced setup")
    parser.add_argument("--document", type=str, default=None)
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--persist", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--custom-model", type=str, default=None)
    args=parser.parse_args()

    llm=OptimizedLLM(
        persist=args.persist,
        custom_model_path=args.custom_model
    )
    if args.document:
        import os
        if os.path.isfile(args.document):
            if not args.document.lower().endswith(".md"):
                print("Only .md files supported.")
            else:
                c=llm.vectordb.add_document(args.document)
                print(f"Added {c} chunks from {args.document}")
        elif os.path.isdir(args.document):
            total=0
            for f in os.listdir(args.document):
                if f.lower().endswith(".md"):
                    fp=os.path.join(args.document,f)
                    c=llm.vectordb.add_document(fp)
                    total+=c
            print(f"Added {total} chunks from {args.document}")
        else:
            print(f"No file/dir found: {args.document}")

    if args.query:
        st=time.time()
        if args.debug:
            resp, dbg, disp=llm.query(args.query, debug_mode=True)
            print(disp)
            print("\nResponse:\n", resp)
        else:
            resp, disp= llm.query(args.query)
            print(disp)
            print("\nResponse:\n", resp)
        print(f"\nQuery took {time.time()-st:.2f}s")

    llm.close()

if __name__=="__main__":
    main()
