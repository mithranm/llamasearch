# src/llamasearch/__main__.py

#!/usr/bin/env python3
"""
__main__.py - Consolidated entry point for LlamaSearch.
Handles GUI and CLI modes for crawling, indexing, searching.
"""

import argparse
import asyncio
import json
import logging
import logging.handlers  # Ensure handlers submodule is imported if needed directly (less common)
import sys
import tarfile
import threading
import time
from pathlib import Path
import signal
from typing import Optional

from llamasearch.core.crawler import Crawl4AICrawler
from llamasearch.core.llmsearch import LLMSearch  # Now orchestrates ChromaDB+BM25
from llamasearch.data_manager import data_manager
from llamasearch.exceptions import ModelNotFoundError, SetupError
from llamasearch.utils import setup_logging

# Use root logger setup (ensure setup_logging is called early if needed globally)
logger = setup_logging()  # Get root logger configured by utils

# --- Global variable for graceful shutdown ---
shutdown_requested = False
llmsearch_instance_global: Optional[LLMSearch] = None
crawler_instance_global: Optional[Crawl4AICrawler] = None


def handle_signal(sig, frame):
    global shutdown_requested, llmsearch_instance_global, crawler_instance_global
    if not shutdown_requested:
        logger.warning(f"Received signal {sig}. Initiating graceful shutdown...")
        shutdown_requested = True
        # Signal components to stop
        if llmsearch_instance_global and hasattr(
            llmsearch_instance_global, "_shutdown_event"
        ):
            shutdown_event = getattr(llmsearch_instance_global, "_shutdown_event", None)
            if shutdown_event:
                logger.debug("Setting LLMSearch shutdown event.")
                shutdown_event.set()
            else:
                logger.warning("LLMSearch instance exists but has no _shutdown_event.")
        if crawler_instance_global and hasattr(crawler_instance_global, "abort"):
            logger.debug("Calling crawler abort.")
            crawler_instance_global.abort()
        # If GUI is running, try to quit it gracefully
        try:
            from PySide6.QtWidgets import QApplication

            app_instance = QApplication.instance()
            # <<< FIX: Check if instance exists before calling quit >>>
            if app_instance:
                logger.info("Requesting Qt Application quit.")
                # Use QTimer or invokeMethod for thread safety if called from signal handler?
                # For simplicity here, direct call, but be cautious in complex scenarios.
                app_instance.quit()
        except ImportError:
            pass  # GUI not available or Qt not installed
        except Exception as e:
            logger.warning(f"Error requesting Qt Application quit: {e}")

    else:
        logger.warning("Shutdown already requested. Force exiting...")
        sys.exit(1)


# Register signal handlers early
signal.signal(signal.SIGINT, handle_signal)  # Ctrl+C
signal.signal(signal.SIGTERM, handle_signal)  # Termination signal


# --- Main Function ---
def main():
    global llmsearch_instance_global, crawler_instance_global  # Allow modification

    parser = argparse.ArgumentParser(
        description="LlamaSearch: Crawl, index, and search documents. Requires setup (`llamasearch-setup`)."
    )
    parser.add_argument(
        "--mode",
        choices=["gui", "cli"],
        default="gui",
        help="Run mode (GUI or Command Line)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    subparsers = parser.add_subparsers(
        dest="command", help="CLI subcommands (if mode==cli)"
    )

    # --- CRAWL+INDEX PARSER --- (Combined crawl and index)
    crawl_index_parser = subparsers.add_parser(
        "crawl-index", help="Crawl website(s) and index the content"
    )
    crawl_index_parser.add_argument(
        "--url",
        type=str,
        required=True,
        nargs="+",
        help="Root URL(s) to crawl (space-separated)",
    )
    crawl_index_parser.add_argument(
        "--target-links", type=int, default=20, help="Max unique pages per root URL"
    )
    crawl_index_parser.add_argument(
        "--max-depth", type=int, default=2, help="Max crawl depth relative to root"
    )
    crawl_index_parser.add_argument(
        "--keywords",
        type=str,
        nargs="*",
        help="Optional keywords to guide crawl relevance (space-separated)",
    )

    # --- ADD PARSER --- (Manually index local files/dirs)
    add_parser = subparsers.add_parser(
        "add", help="Manually add and index local file or directory"
    )
    add_parser.add_argument(
        "path", type=str, help="Path to the file or directory to add"
    )

    # --- SEARCH PARSER ---
    search_parser = subparsers.add_parser("search", help="Query the index")
    search_parser.add_argument("--query", type=str, required=True, help="Search query")

    # --- LIST PARSER ---
    list_parser = subparsers.add_parser(  # noqa: F841 (ruff rule for unused variable)
        "list", help="List indexed sources"
    )

    # --- REMOVE PARSER ---
    remove_parser = subparsers.add_parser("remove", help="Remove an indexed source")
    remove_parser.add_argument(
        "source_path", type=str, help="The exact source path string to remove"
    )

    # --- EXPORT PARSER --- (Export index data)
    export_parser = subparsers.add_parser(
        "export-index", help="Export index data as tar.gz"
    )
    export_parser.add_argument(
        "--output", type=str, required=True, help="Output .tar.gz filename"
    )

    # --- SET PATH PARSER ---
    set_parser = subparsers.add_parser("set", help="Set a storage directory path")
    set_parser.add_argument(
        "--key",
        required=True,
        choices=["crawl_data", "index", "models", "logs"],
        help="Directory key",
    )
    set_parser.add_argument("--path", required=True, help="New path for the key")

    args = parser.parse_args()

    # --- Configure Logging Level ---
    log_level = logging.DEBUG if args.debug else logging.INFO
    root_logger = logging.getLogger("llamasearch")  # Use the fixed root logger name
    root_logger.setLevel(log_level)  # Set root level first

    # <<< FIX: Access handlers via the logger instance >>>
    for handler in root_logger.handlers:
        if isinstance(
            handler, (logging.FileHandler, logging.handlers.RotatingFileHandler)
        ):
            # Set file handler level based on debug flag
            handler.setLevel(log_level)
        elif isinstance(handler, logging.StreamHandler):
            # Keep console INFO unless debug flag is explicitly set
            handler.setLevel(log_level if args.debug else logging.INFO)
        else:
            # Set level for other handlers (like Qt) based on debug flag
            handler.setLevel(log_level)
        # Specific QtLogHandler check (ensure type exists before using isinstance)
        try:
            from llamasearch.ui.qt_logging import QtLogHandler

            if QtLogHandler and isinstance(handler, QtLogHandler):
                handler.setLevel(log_level)  # Set Qt handler based on debug flag
        except ImportError:
            pass

    logger.info(f"LlamaSearch starting in {args.mode} mode.")
    if args.debug:
        logger.debug("Debug logging enabled.")

    # --- Mode Handling ---
    if args.mode == "gui":
        # --- GUI Mode ---
        try:
            logger.info("Verifying essential models before starting GUI...")
            models_dir = Path(data_manager.get_data_paths()["models"])
            active_dir = models_dir / "active_teapot"
            onnx_dir = active_dir / "onnx"
            if not active_dir.is_dir() or not onnx_dir.is_dir():
                raise ModelNotFoundError(
                    "Active Teapot directory not found or incomplete."
                )
            logger.info("Basic model structure check passed.")

            from llamasearch.ui.main import main as gui_main

            logger.info("Starting GUI...")
            gui_main()
            logger.info("GUI finished.")

        except ModelNotFoundError as e:
            logger.error(f"GUI Error: {e}")
            logger.error(
                "Essential models not found. Please run 'llamasearch-setup' first."
            )
            sys.exit(1)
        except SetupError as e:
            logger.error(f"GUI Error: {e}")
            logger.error(
                "There was a setup issue. Please run 'llamasearch-setup' again."
            )
            sys.exit(1)
        except ImportError as e:
            if "PySide6" in str(e):
                logger.error("GUI Error: PySide6 (Qt bindings) not installed.")
                logger.error(
                    "To use the GUI, please install LlamaSearch with GUI extras:"
                )
                logger.error('  pip install "llamasearch[gui]"')
                sys.exit(1)
            else:
                logger.error(
                    f"GUI Error: Failed to import UI components: {e}",
                    exc_info=args.debug,
                )
                sys.exit(1)
        except Exception as e:
            logger.error(f"Failed to start GUI: {e}", exc_info=args.debug)
            sys.exit(1)

    elif args.mode == "cli":
        # --- CLI Mode ---
        cli_shutdown_event = threading.Event()

        # Redefine handle_signal locally for CLI context to use cli_shutdown_event
        def handle_cli_signal(sig, frame):
            global \
                shutdown_requested, \
                llmsearch_instance_global, \
                crawler_instance_global
            if not shutdown_requested:
                logger.warning(f"Received signal {sig}. Initiating CLI shutdown...")
                shutdown_requested = True
                cli_shutdown_event.set()  # Set the CLI-specific event
                # Signal components
                if (
                    llmsearch_instance_global
                    and hasattr(llmsearch_instance_global, "_shutdown_event")
                ):
                    shutdown_event = getattr(llmsearch_instance_global, "_shutdown_event")
                    if shutdown_event is not None:
                        shutdown_event.set()
                if crawler_instance_global:
                    crawler_instance_global.abort()
            else:
                logger.warning("Shutdown already requested. Force exiting...")
                sys.exit(1)

        # Re-register signals for CLI context
        signal.signal(signal.SIGINT, handle_cli_signal)
        signal.signal(signal.SIGTERM, handle_cli_signal)

        # --- Initialize LLMSearch for relevant commands ---
        if args.command in [
            "crawl-index",
            "add",
            "search",
            "list",
            "remove",
            "export-index",
        ]:
            try:
                index_dir = Path(data_manager.get_data_paths()["index"])
                logger.info(
                    f"Initializing LLMSearch for CLI command '{args.command}'..."
                )
                llmsearch_instance_global = LLMSearch(
                    storage_dir=index_dir,
                    shutdown_event=cli_shutdown_event,  # Pass CLI event
                    debug=args.debug,
                    verbose=args.debug,
                )
                logger.info("LLMSearch initialized for CLI.")
            except (ModelNotFoundError, SetupError) as e:
                logger.error(f"CLI Error: {e}")
                logger.error("Please run 'llamasearch-setup' first.")
                sys.exit(1)
            except Exception as e:
                logger.error(
                    f"Unexpected error initializing LLMSearch for CLI: {e}",
                    exc_info=args.debug,
                )
                sys.exit(1)

        # --- Execute CLI Command ---
        try:
            if args.command == "crawl-index":
                if not llmsearch_instance_global:
                    logger.error("LLMSearch instance not available for crawl-index.")
                    sys.exit(1)
                crawl_dir_base = Path(data_manager.get_data_paths()["crawl_data"])
                logger.info(f"Starting crawl & index. Crawl Output: {crawl_dir_base}")
                try:
                    crawler_instance_global = Crawl4AICrawler(
                        root_urls=args.url,
                        base_crawl_dir=crawl_dir_base,
                        target_links=args.target_links,
                        max_depth=args.max_depth,
                        relevance_keywords=args.keywords,
                        shutdown_event=cli_shutdown_event,
                    )
                    try:
                        loop = asyncio.get_event_loop_policy().get_event_loop()
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                    collected_urls = loop.run_until_complete(
                        crawler_instance_global.run_crawl()
                    )
                    if shutdown_requested:
                        logger.warning("Crawl was interrupted.")
                        sys.exit(130)
                    logger.info(
                        f"Crawl completed. Collected {len(collected_urls)} pages. Starting indexing..."
                    )
                    raw_markdown_dir = crawl_dir_base / "raw"
                    if raw_markdown_dir.is_dir():
                        reverse_lookup_path = crawl_dir_base / "reverse_lookup.json"
                        reverse_lookup = {}
                        if reverse_lookup_path.exists():
                            try:
                                with open(
                                    reverse_lookup_path, "r", encoding="utf-8"
                                ) as f:
                                    reverse_lookup = json.load(f)
                            except Exception as e:
                                logger.warning(f"Could not load reverse lookup: {e}")
                        added_total, processed_files = 0, 0
                        all_files = list(raw_markdown_dir.glob("*.md"))
                        logger.info(f"Found {len(all_files)} markdown files to index.")
                        for md_file in all_files:
                            if shutdown_requested:
                                logger.warning("Indexing interrupted.")
                                break
                            file_hash = md_file.stem
                            original_url = reverse_lookup.get(file_hash, str(md_file))
                            logger.info(f"Indexing: {original_url} ({md_file.name})")
                            # add_source returns a tuple (count, success_flag)
                            added_count_result, _ = llmsearch_instance_global.add_source(
                                str(md_file)
                            )
                            added_total += added_count_result
                            processed_files += 1
                        logger.info(
                            f"Indexing complete. Added {added_total} new chunks from {processed_files} files."
                        )
                    else:
                        logger.warning(
                            f"Crawl 'raw' dir not found: {raw_markdown_dir}. No indexing."
                        )
                except KeyboardInterrupt:
                    logger.warning("CLI crawl-index aborted by user.")
                    sys.exit(130)
                except Exception as e:
                    logger.error(f"Crawl/Index error: {e}", exc_info=args.debug)
                    sys.exit(1)

            elif args.command == "add":
                if not llmsearch_instance_global:
                    logger.error("LLMSearch not available for add.")
                    sys.exit(1)
                target_path = Path(args.path).resolve()
                logger.info(f"Adding and indexing source: {target_path}")
                try:
                    added_count = llmsearch_instance_global.add_source(str(target_path))
                    logger.info(f"Finished adding. Added {added_count} new chunks.")
                    if shutdown_requested:
                        logger.warning("Add command possibly interrupted.")
                except KeyboardInterrupt:
                    logger.warning("CLI add aborted by user.")
                    sys.exit(130)
                except Exception as e:
                    logger.error(f"Add error: {e}", exc_info=args.debug)
                    sys.exit(1)

            elif args.command == "search":
                if not llmsearch_instance_global:
                    logger.error("LLMSearch not available for search.")
                    sys.exit(1)
                logger.info(f"Searching for: '{args.query}'")
                try:
                    res = llmsearch_instance_global.llm_query(
                        args.query, debug_mode=args.debug
                    )
                    print("\n--- AI Answer ---")
                    print(res.get("response", "N/A."))
                    if args.debug:
                        print("\n--- Retrieved Context ---")
                        print(res.get("retrieved_context", "N/A."))
                        print("\n--- Debug Info ---")
                        debug_data = res.get("debug_info", {})
                        print(
                            json.dumps(debug_data, indent=2, default=str)
                        )  # Use default=str for non-serializable
                    if shutdown_requested:
                        logger.warning("Search command interrupted.")
                except KeyboardInterrupt:
                    logger.warning("CLI search aborted by user.")
                    sys.exit(130)
                except Exception as e:
                    logger.error(f"Search error: {e}", exc_info=args.debug)
                    sys.exit(1)

            elif args.command == "list":
                if not llmsearch_instance_global:
                    logger.error("LLMSearch not available for list.")
                    sys.exit(1)
                logger.info("Listing indexed sources...")
                try:
                    sources = llmsearch_instance_global.get_indexed_sources()
                    if not sources:
                        print("No sources currently indexed.")
                    else:
                        print(f"\n--- Indexed Sources ({len(sources)}) ---")
                        for item in sources:
                            print(f"- Path: {item.get('source_path', 'N/A')}")
                            filename = item.get("filename", "N/A")
                            if filename == "N/A" and item.get("source_path"):
                                try:
                                    filename = Path(item["source_path"]).name
                                except Exception:
                                    filename = "N/A"
                            print(f"  Filename: {filename}")
                            print(f"  Chunks: {item.get('chunk_count', 'N/A')}")
                            mtime_val = item.get("mtime")
                            mtime_str = (
                                time.strftime(
                                    "%Y-%m-%d %H:%M:%S", time.localtime(mtime_val)
                                )
                                if mtime_val is not None
                                else "N/A"
                            )
                            print(f"  Modified: {mtime_str}")
                            print("-" * 10)
                except Exception as e:
                    logger.error(f"List error: {e}", exc_info=args.debug)
                    sys.exit(1)

            elif args.command == "remove":
                if not llmsearch_instance_global:
                    logger.error("LLMSearch not available for remove.")
                    sys.exit(1)
                logger.info(f"Removing source: {args.source_path}")
                try:
                    success = llmsearch_instance_global.remove_source(args.source_path)
                    if success:
                        logger.info(f"Successfully removed source: {args.source_path}")
                    else:
                        logger.warning(
                            f"Source not found or removal failed: {args.source_path}"
                        )
                except Exception as e:
                    logger.error(f"Remove error: {e}", exc_info=args.debug)
                    sys.exit(1)

            elif args.command == "export-index":
                index_dir_base = Path(data_manager.get_data_paths()["index"])
                output_path = Path(args.output).resolve()
                if not str(output_path).endswith(".tar.gz"):
                    logger.error("Output filename must end with .tar.gz")
                    sys.exit(1)
                if output_path.exists():
                    logger.warning(f"Output file exists: {output_path}. Overwriting.")
                if not index_dir_base.is_dir() or not any(index_dir_base.iterdir()):
                    logger.error(
                        f"Index directory '{index_dir_base}' not found or empty."
                    )
                    sys.exit(1)
                try:
                    logger.info(
                        f"Exporting index from {index_dir_base} to {output_path}..."
                    )
                    if llmsearch_instance_global:
                        logger.info("Closing LLMSearch before export...")
                        llmsearch_instance_global.close()
                        llmsearch_instance_global = None
                    with tarfile.open(output_path, "w:gz") as tar:
                        tar.add(str(index_dir_base), arcname=index_dir_base.name)
                    logger.info(f"Index exported successfully to {output_path}")
                except Exception as e:
                    logger.error(f"Export error: {e}", exc_info=args.debug)
                    sys.exit(1)

            elif args.command == "set":
                try:
                    data_manager.set_data_path(args.key, args.path)
                    logger.info(
                        f"Set '{args.key}' path to '{args.path}'. Changes saved."
                    )
                except Exception as e:
                    logger.error(f"Set path error: {e}", exc_info=args.debug)
                    sys.exit(1)
            else:
                if args.command:
                    logger.error(f"Invalid CLI subcommand: '{args.command}'.")
                else:
                    logger.error("No valid CLI subcommand provided.")
                parser.print_help()
                sys.exit(1)

        except KeyboardInterrupt:
            logger.warning("CLI operation aborted by user (Ctrl+C).")
            sys.exit(130)
        except SystemExit as e:
            raise e  # Allow clean exits
        except Exception as e:
            logger.error(f"Unexpected CLI error: {e}", exc_info=args.debug)
            sys.exit(1)
        finally:
            if llmsearch_instance_global:
                logger.debug("Closing LLMSearch in CLI finally block...")
                llmsearch_instance_global.close()
                logger.debug("LLMSearch closed.")

    else:
        logger.error("Invalid mode selected.")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
