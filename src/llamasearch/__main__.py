#!/usr/bin/env python3
"""
__main__.py - Consolidated entry point for LlamaSearch (runtime).

Handles GUI, MCP, and CLI modes for crawling, indexing, searching, etc.
Requires models to be present. Run `llamasearch-setup` first if needed.
"""

import argparse
import sys
from pathlib import Path
import asyncio
import tarfile
import logging
import json

# Import GUI, MCP, and app logic modules:
from llamasearch.utils import setup_logging

# --- Use the revised Crawl4AICrawler ---
from llamasearch.core.crawler import Crawl4AICrawler

from llamasearch.core.llmsearch import LLMSearch
from llamasearch.exceptions import ModelNotFoundError
from llamasearch.data_manager import data_manager

logger = setup_logging(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="LlamaSearch: Crawl, index, and search documents. Requires setup (`llamasearch-setup`)."
    )
    parser.add_argument(
        "--mode", choices=["gui", "mcp", "cli"], default="gui", help="Run mode"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    subparsers = parser.add_subparsers(
        dest="command", help="CLI subcommands (if mode==cli)"
    )

    # --- Updated CRAWL PARSER ---
    crawl_parser = subparsers.add_parser(
        "crawl", help="Crawl website using Crawl4AI (custom priority)"
    )
    crawl_parser.add_argument("--url", type=str, required=True, help="Root URL")
    crawl_parser.add_argument("--target-links", type=int, default=15, help="Max unique pages to save")
    crawl_parser.add_argument("--max-depth", type=int, default=2, help="Max crawl depth")
    # Removed --phrase and API args
    # Added optional keywords arg for relevance scoring
    crawl_parser.add_argument(
        "--keywords", type=str, nargs='*', help="Optional keywords to guide crawl relevance (space-separated)"
    )
    crawl_parser.add_argument(
        "--output-dir", type=str, help="Base directory for crawl data (defaults to settings 'crawl_data')"
    )
    # --- END CRAWL PARSER MODIFICATION ---

    # Build-index subcommand (no changes needed here)
    build_index_parser = subparsers.add_parser(
        "build-index", help="Build index from crawl data" # Updated help slightly
    )
    # --- MODIFIED build-index input: Point to base crawl dir ---
    build_index_parser.add_argument(
        "--crawl-data-dir",
        type=str,
        help="Path to crawl data directory containing 'raw' subdir and 'reverse_lookup.json' (defaults to settings 'crawl_data')",
    )
    # --- END MODIFICATION ---
    build_index_parser.add_argument(
        "--index-dir", type=str, help="Store index data here (defaults to settings)"
    )

    # Search subcommand (no changes needed here)
    search_parser = subparsers.add_parser("search", help="Query the index")
    search_parser.add_argument("--query", type=str, required=True, help="Search query")
    search_parser.add_argument(
        "--generate", action="store_true", default=True, help="Generate AI response"
    )
    search_parser.add_argument(
        "--index-dir", type=str, help="Index data directory (defaults to settings)"
    )

    # Export-index subcommand (no changes needed here)
    export_index_parser = subparsers.add_parser(
        "export-index", help="Export index as tar.gz"
    )
    export_index_parser.add_argument(
        "--output", type=str, required=True, help="Output .tar.gz filename"
    )
    export_index_parser.add_argument(
        "--index-dir", type=str, help="Index directory to export (defaults to settings)"
    )

    # Set storage directory subcommand (no changes needed here)
    set_parser = subparsers.add_parser("set", help="Set a storage directory")
    set_parser.add_argument(
        "--key",
        type=str,
        required=True,
        choices=["crawl_data", "index", "models", "logs"],
        help="Directory key",
    )
    set_parser.add_argument("--path", type=str, required=True, help="New path")

    args = parser.parse_args()

    data_paths = data_manager.get_data_paths()

    # Determine crawl data base directory
    crawl_dir_base_path_str = data_paths["crawl_data"]
    if args.command == "crawl" and args.output_dir:
        crawl_dir_base_path_str = args.output_dir
    elif args.command == "build-index" and args.crawl_data_dir:
         crawl_dir_base_path_str = args.crawl_data_dir # Use crawl_data_dir for build-index source
    crawl_dir_base = Path(crawl_dir_base_path_str)

    # Determine index base directory
    index_dir_base_path_str = data_paths["index"]
    if args.command in ["build-index", "search", "export-index"] and args.index_dir:
        index_dir_base_path_str = args.index_dir
    index_dir_base = Path(index_dir_base_path_str)

    if args.debug:
        logger.setLevel(logging.DEBUG)
        for handler in logger.handlers:
            handler.setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled.")
        logger.debug(
            f"Crawl Dir Base: {crawl_dir_base}, Index Dir Base: {index_dir_base}"
        )

    # --- Mode Handling ---
    if args.mode == "gui":
        try:
            from llamasearch.ui.main import main as gui_main

            logger.info("Starting GUI mode...")
            gui_main()
        except ModelNotFoundError as e:
            logger.error(f"GUI Error: {e}")
            logger.error("Please run 'llamasearch-setup' first.")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Failed to start GUI: {e}", exc_info=args.debug)
            sys.exit(1)

    elif args.mode == "mcp":
        logger.warning("MCP mode implementation pending.")
        sys.exit(1)

    elif args.mode == "cli":
        llmsearch_instance = None
        try:
            if args.command == "crawl":
                try:
                    # --- Use Revised Crawl4AICrawler ---
                    logger.info("Starting crawl with Crawl4AI (custom priority)...")
                    crawler = Crawl4AICrawler(
                        root_urls=[args.url],
                        base_crawl_dir=crawl_dir_base,
                        target_links=args.target_links,
                        max_depth=args.max_depth,
                        relevance_keywords=args.keywords # Pass optional keywords
                    )
                    # run_crawl is now async
                    collected_urls = asyncio.run(crawler.run_crawl())
                    # --- REMOVED ARCHIVE EXPORT ---
                    # archive_path = crawler.export_archive() # This method no longer exists
                    logger.info(f"Crawl completed. Collected {len(collected_urls)} pages.")
                    # --- End Crawl4AICrawler usage ---
                except Exception as e:
                    logger.error(f"Crawl error: {e}", exc_info=args.debug)
                    sys.exit(1)

            elif args.command in ["build-index", "search"]:
                # Logic remains the same, but depends on the crawl output format
                # LLMSearch initialization handles model checks.
                logger.info(f"Initializing LLMSearch for '{args.command}'...")
                llmsearch_instance = LLMSearch(
                    storage_dir=index_dir_base, debug=args.debug, verbose=args.debug
                )
                logger.info("LLMSearch initialized.")

                if args.command == "build-index":
                    # --- MODIFIED: Build index directly from crawl data dir ---
                    target_dir = crawl_dir_base / "raw" # Index the 'raw' subdir
                    reverse_lookup_file = crawl_dir_base / "reverse_lookup.json"

                    if not target_dir.is_dir():
                         logger.error(f"Crawl 'raw' directory not found: {target_dir}. Did the crawl run?")
                         sys.exit(1)
                    if not reverse_lookup_file.is_file():
                         logger.error(f"Crawl 'reverse_lookup.json' not found in: {crawl_dir_base}")
                         # Optional: Could proceed without it, but source info will be missing
                         sys.exit(1)
                    # --- END MODIFICATION ---

                    logger.info(f"Building index from: {target_dir}")
                    # Ensure add_documents can handle the .md files + reverse_lookup.json structure
                    # We assume LLMSearch/VectorDB is adapted or can handle this structure
                    # TODO: Potentially pass reverse_lookup_file path to add_documents_from_directory if needed
                    added = llmsearch_instance.add_documents_from_directory(target_dir)
                    logger.info(f"Index build finished. Added {added} new chunks.")
                    # No temp dir cleanup needed now

                elif args.command == "search":
                    if not args.generate:
                        logger.warning(
                            "CLI search currently always generates response."
                        )
                    res = llmsearch_instance.llm_query(
                        args.query, debug_mode=args.debug
                    )
                    print("\n--- AI Answer ---")
                    print(res.get("response", "N/A."))
                    if args.debug:
                        print("\n--- Retrieved Context ---")
                        # Use the raw context if available for CLI debug
                        retrieved_context_str = res.get("retrieved_context", "N/A.")
                        # Simple heuristic to check if it's the formatted string or raw chunks
                        if isinstance(retrieved_context_str, str) and retrieved_context_str.startswith("--- Chunk"):
                             print(retrieved_context_str)
                        elif "debug_info" in res and "vector_results" in res["debug_info"]:
                             # If raw context isn't directly in result, reconstruct from debug
                             print("Raw Chunks (from debug_info):")
                             docs = res["debug_info"]["vector_results"].get("documents", [])
                             metas = res["debug_info"]["vector_results"].get("metadatas", [])
                             scores = res["debug_info"]["vector_results"].get("scores", [])
                             for i, doc in enumerate(docs):
                                  meta_str = json.dumps(metas[i]) if i < len(metas) else "{}"
                                  score_str = f"{scores[i]:.4f}" if i < len(scores) else "N/A"
                                  print(f"--- Chunk {i+1} | Score: {score_str} | Meta: {meta_str} ---")
                                  print(doc)
                                  print("-"*(15+len(str(i+1)))) # Separator
                        else:
                             print("Raw retrieved context details not available in debug info.")

                        print("\n--- Debug Info ---")
                        print(json.dumps(res.get("debug_info", {}), indent=2))


            elif args.command == "export-index":
                # No changes needed
                output_path = Path(args.output).resolve()
                if not str(output_path).endswith(".tar.gz"):
                    logger.error("Output must be .tar.gz")
                    sys.exit(1)
                if output_path.exists():
                    logger.error(f"Output exists: {output_path}")
                    sys.exit(1)
                if not index_dir_base.is_dir():
                    logger.error(f"Index dir not found: {index_dir_base}")
                    sys.exit(1)
                try:
                    logger.info(
                        f"Exporting index from {index_dir_base} to {output_path}..."
                    )
                    with tarfile.open(output_path, "w:gz") as tar:
                        # Use arcname to place index contents directly in archive root or specific dir
                        tar.add(str(index_dir_base), arcname=index_dir_base.name)
                    logger.info(f"Index exported to {output_path}")
                except Exception as e:
                    logger.error(f"Export error: {e}", exc_info=args.debug)
                    sys.exit(1)

            elif args.command == "set":
                # No changes needed
                try:
                    data_manager.set_data_path(args.key, args.path)
                    logger.info(f"Set '{args.key}' path to '{args.path}'.")
                except Exception as e:
                    logger.error(f"Set path error: {e}", exc_info=args.debug)
                    sys.exit(1)
            else:
                logger.error("No valid CLI subcommand provided.")
                # Show help for the main parser if no command is given
                parser.print_help()
                sys.exit(1)

        except ModelNotFoundError as e:
            logger.error(f"CLI Error: {e}")
            logger.error("Please run 'llamasearch-setup' first.")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Unexpected CLI error: {e}", exc_info=args.debug)
            sys.exit(1)
        finally:
            if llmsearch_instance:
                llmsearch_instance.close()
                logger.debug("LLMSearch closed.")

    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    try:  # Graceful exit setup
        if not sys.platform.startswith("darwin"):
            import signal

            signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))
            signal.signal(signal.SIGTERM, lambda s, f: sys.exit(0))
    except ImportError:
        logger.warning("Signal handling unavailable.")
    except Exception as e:
        logger.warning(f"Signal handler setup error: {e}")
    main()