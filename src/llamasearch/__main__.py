#!/usr/bin/env python3
"""
__main__.py - Consolidated entry point for LlamaSearch.

This module supports three primary modes:
  • GUI mode: Launch the full graphical interface.
  • MCP mode: Run as a Model Context Protocol server.
  • CLI mode: Run headless commands to crawl, build an index, search, export, or set storage directories.

Crawl archives and indices are separate. A crawl archive is a tar.gz file containing raw scraped content and 
a manifest (including root URL and query phrase). The indexing process (build-index) reads the raw crawl archive 
and computes embeddings and other index artifacts, which can then be exported for lightweight query devices.
  
Usage examples:

  # GUI (default)
  python -m llamasearch
  
  # MCP mode (requires uvicorn)
  python -m llamasearch --mode mcp
  
  # CLI mode:
  ## Crawl a website (creates a new crawl archive)
  python -m llamasearch --mode cli crawl --url "https://gnu.org" --target-links 15 --max-depth 2 --phrase "python" --api-type jina
  
  ## Build an index from a crawl archive (extracted directory or tar.gz)
  python -m llamasearch --mode cli build-index --crawl-archive /path/to/extracted/crawl_folder
  
  ## Search the index (use --generate for a generated AI response)
  python -m llamasearch --mode cli search --query "What is free software?" --generate
  
  ## Export an index (compress the index directory into a tar.gz)
  python -m llamasearch --mode cli export-index --output index_export_20230409.tar.gz
  
  ## Change a storage directory (e.g., for crawl_data)
  python -m llamasearch --mode cli set --key crawl_data --path "/new/path/to/crawl_data"
"""

import argparse
import sys
from pathlib import Path

import asyncio
# Import GUI, MCP, and app logic modules:
from killeraiagent.models import NullModel
from llamasearch.utils import get_llamasearch_dir
from llamasearch.core.crawler import ConcurrentAsyncCrawler
from llamasearch.core.llmsearch import LLMSearch

def main():
    parser = argparse.ArgumentParser(
        description="LlamaSearch: Crawl, index, and search documents with a modular interface."
    )
    parser.add_argument("--mode", choices=["gui", "mcp", "cli"], default="gui",
                        help="Run mode: 'gui' (default), 'mcp', or 'cli'.")
    subparsers = parser.add_subparsers(dest="command", help="CLI subcommands (if mode==cli)")

    # Crawl subcommand: perform crawl, produce a crawl archive.
    crawl_parser = subparsers.add_parser("crawl", help="Crawl a website and create a crawl archive")
    crawl_parser.add_argument("--url", type=str, required=True, help="Root URL to crawl")
    crawl_parser.add_argument("--target-links", type=int, default=15, help="Max links to collect")
    crawl_parser.add_argument("--max-depth", type=int, default=2, help="Max crawl depth")
    crawl_parser.add_argument("--phrase", type=str, default="", help="Query phrase to target (optional)")
    crawl_parser.add_argument("--api-type", choices=["jina", "mithran"], default="jina", help="API to use")
    crawl_parser.add_argument("--key-id", type=str, help="API key ID (for mithran)")
    crawl_parser.add_argument("--private-key", type=str, help="Path to RSA private key (for mithran)")

    # Build-index subcommand: build (or rebuild) an index from an existing crawl archive (directory)
    build_index_parser = subparsers.add_parser("build-index", help="Build an index from an existing crawl archive")
    build_index_parser.add_argument("--crawl-archive", type=str, required=True,
                                    help="Path to an extracted crawl archive (folder with manifest, raw/ etc.)")

    # Search subcommand
    search_parser = subparsers.add_parser("search", help="Query the index and optionally generate a response")
    search_parser.add_argument("--query", type=str, required=True, help="Search query")
    search_parser.add_argument("--generate", action="store_true",
                               help="Generate an AI response (instantiates a real LLM)")

    # Export-index subcommand: pack the current index into a tar.gz archive.
    export_index_parser = subparsers.add_parser("export-index", help="Export the current index as a tar.gz archive")
    export_index_parser.add_argument("--output", type=str, required=True, help="Output file name")

    # Set storage directory subcommand
    set_parser = subparsers.add_parser("set", help="Set a storage directory dynamically")
    set_parser.add_argument("--key", type=str, required=True, choices=["crawl_data", "index", "models", "logs"],
                            help="Data directory key to update")
    set_parser.add_argument("--path", type=str, required=True, help="New path for the directory")

    args = parser.parse_args()

    if args.mode == "gui":
        from llamasearch.ui.main import main as gui_main
        gui_main()
    elif args.mode == "mcp":
        try:
            import uvicorn
            uvicorn.run("llamasearch.mcp_server:app", host="0.0.0.0", port=8001, log_level="info")
        except ImportError:
            print("uvicorn is required. Install with 'pip install uvicorn'")
    elif args.mode == "cli":
        llamasearch_dir = get_llamasearch_dir()
        # Create a LlamaSearchApp instance. For tasks that do not require generation, use NullModel.
        if args.command in ["crawl", "build-index", "set"]:
            model = NullModel()
        elif args.command == "search":
            if args.generate:
                # Instantiate a real LLM (adjust model path as needed)
                # TODO: Find a better way to specify the model path.
                raise NotImplementedError("Model generation not implemented in CLI mode.")
                # model_path = get_llamasearch_dir() / "models"
                # model = create_llm(model_path_or_id=model_path, backend="hf", verbose=True)
            else:
                model = NullModel()
                
        llmsearch_instance = LLMSearch(
                model=model,
                storage_dir=llamasearch_dir / "index",
                models_dir=llamasearch_dir / "models",
                debug=True,
            )

        if args.command == "crawl":
            try:
                def print_crawl(url, content, crawl_folder):
                    print(f"Fetched {url} ({len(content)} bytes) to {crawl_folder}")

                crawler = ConcurrentAsyncCrawler(
                    root_urls=[args.url],
                    base_crawl_dir=llamasearch_dir / "crawl_data",
                    on_fetch_page=print_crawl
                )
                asyncio.run(crawler.run_crawl())
            except Exception as e:
                print(f"Error during crawl: {e}")
                exit(1)
            print(f"Crawl completed. Archive created at {crawler.export_archive()}")
            exit(0)
        elif args.command == "build-index":
            # Build index from an existing crawl archive folder.
            llmsearch_instance.add_documents_from_directory(Path(args.crawl_archive))
        elif args.command == "search":
            res = llmsearch_instance.llm_query(args.query)
            print(res)
        elif args.command == "export-index":
            # Compress the index directory into a tar.gz archive.
            from tarfile import open as taropen
            index_dir = llamasearch_dir / "index"
            crawl_dir = llamasearch_dir / "crawl_data"
            if not index_dir.exists():
                print(f"Index directory {index_dir} does not exist.")
                exit(1)
            if not crawl_dir.exists():
                print(f"Crawl data directory {crawl_dir} does not exist.")
                exit(1)
            if not index_dir.is_dir():
                print(f"Index path {index_dir} is not a directory.")
                exit(1)
            if not crawl_dir.is_dir():
                print(f"Crawl data path {crawl_dir} is not a directory.")
                exit(1)
            if not args.output.endswith(".tar.gz"):
                print(f"Output file {args.output} must end with .tar.gz")
                exit(1)
            if Path(args.output).exists():
                print(f"Output file {args.output} already exists. Will not Overwrite.")
                exit(1)
            output = args.output
            with taropen(output, "w:gz") as tar:
                tar.add(str(index_dir), arcname=index_dir.name)
            print(f"Index exported to {output}")
        elif args.command == "set":
            from llamasearch.data_manager import data_manager
            try:
                data_manager.set_data_path(args.key, args.path)
                print(f"Set {args.key} to {args.path}")
            except Exception as e:
                print(f"Error: {e}")
        else:
            print("No valid CLI subcommand specified. Use --help for usage information.")
    else:
        parser.print_help()

if __name__ == "__main__":
    try:
        if not sys.platform.startswith("darwin"):
            import signal
            signal.signal(signal.SIGINT, lambda s,f: sys.exit(0))
            signal.signal(signal.SIGTERM, lambda s,f: sys.exit(0))
    except Exception:
        pass
    main()
