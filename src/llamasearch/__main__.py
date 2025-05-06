#!/usr/bin/env python3
"""
__main__.py - Consolidated entry point for LlamaSearch.
Handles GUI and CLI modes for crawling, indexing, searching.
"""

import argparse
import asyncio
import json
import logging
import logging.handlers
import signal
import sys
import tarfile
import threading
import time
from pathlib import Path
from typing import Optional

# Core components
from llamasearch.core.crawler import Crawl4AICrawler
from llamasearch.core.search_engine import LLMSearch  # Updated import
# Utilities and Data Management
from llamasearch.data_manager import data_manager
from llamasearch.exceptions import ModelNotFoundError, SetupError
from llamasearch.utils import setup_logging

# Setup root logger early
logger = setup_logging()

# --- Global variables for signal handling and instance access ---
shutdown_requested = False
llmsearch_instance_global: Optional[LLMSearch] = None
crawler_instance_global: Optional[Crawl4AICrawler] = None


# --- Signal Handling ---
def handle_signal(sig, frame):
    """Gracefully handle SIGINT and SIGTERM."""
    global shutdown_requested, llmsearch_instance_global, crawler_instance_global
    if not shutdown_requested:
        logger.warning(f"Received signal {sig}. Initiating graceful shutdown...")
        shutdown_requested = True  # Set flag first

        # Signal LLMSearch to shut down (via its internal event)
        if llmsearch_instance_global and hasattr(
            llmsearch_instance_global, "_shutdown_event"
        ):
            shutdown_event = getattr(llmsearch_instance_global, "_shutdown_event", None)
            if shutdown_event:
                logger.debug("Setting LLMSearch shutdown event.")
                shutdown_event.set()
            else:
                # Should not happen if LLMSearch is initialized correctly
                logger.warning("LLMSearch instance exists but has no _shutdown_event.")

        # Signal Crawler to abort
        if crawler_instance_global and hasattr(crawler_instance_global, "abort"):
            logger.debug("Calling crawler abort.")
            crawler_instance_global.abort()

        # Attempt to quit Qt application gracefully if running
        try:
            # Import dynamically only when needed
            from PySide6.QtCore import QTimer
            from PySide6.QtWidgets import QApplication

            app_instance = QApplication.instance()
            if app_instance:
                logger.info("Requesting Qt Application quit.")
                # Use QTimer for thread safety if called from non-GUI thread (signals can be tricky)
                QTimer.singleShot(0, app_instance.quit)
        except ImportError:
            # GUI not available or Qt not installed
            pass
        except Exception as e:
            # Catch potential errors during Qt quit request
            logger.warning(f"Error requesting Qt Application quit: {e}")

    else:
        # If shutdown already requested, force exit
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
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging level",
    )
    subparsers = parser.add_subparsers(
        dest="command",
        help="CLI subcommands (required if mode==cli)",
        # Make subcommand required in CLI mode implicitly by not setting a default
    )

    # --- CLI Subparsers ---

    # Crawl & Index Command
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
        "--target-links",
        type=int,
        default=20,
        help="Max unique pages per root URL (default: 20)",
    )
    crawl_index_parser.add_argument(
        "--max-depth",
        type=int,
        default=2,
        help="Max crawl depth relative to root (default: 2)",
    )
    crawl_index_parser.add_argument(
        "--keywords",
        type=str,
        nargs="*",
        help="Optional keywords to guide crawl relevance (space-separated)",
    )

    # Add Local Content Command
    add_parser = subparsers.add_parser(
        "add", help="Manually add and index a local file or directory"
    )
    add_parser.add_argument(
        "path", type=str, help="Path to the file or directory to add"
    )

    # Search Command
    search_parser = subparsers.add_parser("search", help="Query the index")
    search_parser.add_argument("--query", type=str, required=True, help="Search query")

    # List Indexed Sources Command
    # Variable assigned but not used, suppressed via noqa
    list_parser = subparsers.add_parser("list", help="List indexed sources")  # noqa: F841

    # Remove Source Command
    remove_parser = subparsers.add_parser("remove", help="Remove an indexed source")
    remove_parser.add_argument(
        "source_identifier",
        type=str,
        help="The exact source identifier (URL or path) to remove",
    )

    # Export Index Command
    export_parser = subparsers.add_parser(
        "export-index", help="Export index data as tar.gz archive"
    )
    export_parser.add_argument(
        "--output", type=str, required=True, help="Output .tar.gz filename"
    )

    # Set Data Path Command
    set_parser = subparsers.add_parser("set", help="Set a storage directory path")
    set_parser.add_argument(
        "--key",
        required=True,
        choices=["crawl_data", "index", "models", "logs"],
        help="Directory key to set",
    )
    set_parser.add_argument(
        "--path", required=True, help="New path for the specified key"
    )

    args = parser.parse_args()

    # --- Configure Logging Level ---
    log_level = logging.DEBUG if args.debug else logging.INFO
    # Get the root logger instance configured by setup_logging
    root_logger = logging.getLogger("llamasearch")
    # Set root level first - handlers will filter based on their own levels
    root_logger.setLevel(log_level)

    # Adjust handler levels based on debug flag
    for handler in root_logger.handlers:
        if isinstance(
            handler, (logging.FileHandler, logging.handlers.RotatingFileHandler)
        ):
            # File handler always captures DEBUG level and up
            handler.setLevel(logging.DEBUG)
        elif isinstance(handler, logging.StreamHandler):
            # Keep console INFO unless debug flag is explicitly set
            handler.setLevel(log_level if args.debug else logging.INFO)
        else:
            # Set level for other handlers (like Qt) based on debug flag
            handler.setLevel(log_level)
        # Specific QtLogHandler check
        try:
            # Dynamically import Qt handler only if needed
            from llamasearch.ui.qt_logging import QtLogHandler

            if QtLogHandler and isinstance(handler, QtLogHandler):
                # Set Qt handler level based on debug flag
                handler.setLevel(log_level)
        except ImportError:
            # Qt components not available, ignore
            pass

    logger.info(f"LlamaSearch starting in {args.mode} mode.")
    if args.debug:
        logger.debug("Debug logging enabled.")

    # --- Mode Handling ---
    if args.mode == "gui":
        # --- GUI Mode ---
        try:
            logger.info("Verifying essential models before starting GUI...")
            models_dir_path_str = data_manager.get_data_paths().get("models")
            if not models_dir_path_str:
                raise SetupError("Models directory path not configured in settings.")
            models_dir = Path(models_dir_path_str)
            # Check for the 'active_model' directory as a proxy for setup completion
            active_dir = models_dir / "active_model"
            onnx_dir = active_dir / "onnx"  # Check for ONNX subdir too
            if not active_dir.is_dir() or not onnx_dir.is_dir():
                raise ModelNotFoundError(
                    "Active model directory not found or incomplete. Run 'llamasearch-setup'."
                )
            logger.info("Basic model structure check passed.")

            # Import and run the GUI main function
            # Import dynamically to avoid Qt dependency in CLI mode if possible
            from llamasearch.ui.main import main as gui_main

            logger.info("Starting GUI...")
            gui_main()  # This blocks until the GUI closes
            logger.info("GUI finished.")

        except ModelNotFoundError as e:
            logger.error(f"GUI Error: {e}")
            logger.error(
                "Essential models not found. Please run 'llamasearch-setup' first."
            )
            sys.exit(1)
        except SetupError as e:
            # Catch other setup-related errors
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
                # Handle other import errors potentially related to UI components
                logger.error(
                    f"GUI Error: Failed to import UI components: {e}",
                    exc_info=args.debug,
                )
                sys.exit(1)
        except Exception as e:
            # Catch any other unexpected errors during GUI launch
            logger.error(f"Failed to start GUI: {e}", exc_info=args.debug)
            sys.exit(1)

    elif args.mode == "cli":
        # --- CLI Mode ---
        # Check if a command was provided for CLI mode
        if not args.command:
            parser.error(
                "A command is required for CLI mode (e.g., search, add, list)."
            )
            sys.exit(2)  # Use different exit code for argument error

        # Setup separate shutdown event for CLI context
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

                # Signal components (same logic as global handler)
                if llmsearch_instance_global and hasattr(
                    llmsearch_instance_global, "_shutdown_event"
                ):
                    shutdown_event = getattr(
                        llmsearch_instance_global, "_shutdown_event"
                    )
                    if shutdown_event is not None:
                        shutdown_event.set()
                if crawler_instance_global:
                    crawler_instance_global.abort()
            else:
                logger.warning("Shutdown already requested. Force exiting...")
                sys.exit(1)

        # Re-register signals for CLI context to use the local handler
        signal.signal(signal.SIGINT, handle_cli_signal)
        signal.signal(signal.SIGTERM, handle_cli_signal)

        # --- Initialize LLMSearch for relevant commands ---
        needs_llmsearch = args.command in [
            "crawl-index",
            "add",
            "search",
            "list",
            "remove",
            "export-index",
        ]
        if needs_llmsearch:
            try:
                index_dir_path_str = data_manager.get_data_paths().get("index")
                if not index_dir_path_str:
                    raise SetupError("Index directory path not configured in settings.")
                index_dir = Path(index_dir_path_str)
                logger.info(
                    f"Initializing LLMSearch for CLI command '{args.command}'..."
                )
                # Pass the CLI-specific shutdown event
                llmsearch_instance_global = LLMSearch(
                    storage_dir=index_dir,
                    shutdown_event=cli_shutdown_event,
                    debug=args.debug,
                    verbose=args.debug,  # Keep verbose in sync with debug for CLI
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
                crawl_dir_base_path_str = data_manager.get_data_paths().get(
                    "crawl_data"
                )
                if not crawl_dir_base_path_str:
                    raise SetupError("Crawl data directory path not configured.")
                crawl_dir_base = Path(crawl_dir_base_path_str)
                logger.info(
                    f"Starting crawl & index. Crawl Output Dir: {crawl_dir_base}"
                )
                try:
                    # Instantiate crawler
                    crawler_instance_global = Crawl4AICrawler(
                        root_urls=args.url,
                        base_crawl_dir=crawl_dir_base,
                        target_links=args.target_links,
                        max_depth=args.max_depth,
                        relevance_keywords=args.keywords,
                        shutdown_event=cli_shutdown_event,  # Pass CLI event
                        verbose_logging=args.debug,  # Pass debug flag
                    )
                    # Setup asyncio loop
                    try:
                        loop = asyncio.get_event_loop_policy().get_event_loop()
                    except RuntimeError:
                        logger.debug("No current asyncio event loop, creating new one.")
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)

                    # Run crawl
                    collected_urls = loop.run_until_complete(
                        crawler_instance_global.run_crawl()
                    )

                    # Check for interruption after crawl
                    if shutdown_requested:
                        logger.warning("Crawl was interrupted by signal.")
                        sys.exit(130)  # Standard exit code for Ctrl+C termination

                    url_count = len(collected_urls) if collected_urls is not None else 0
                    logger.info(
                        f"Crawl completed. Collected {url_count} pages. Starting indexing..."
                    )

                    # Indexing phase
                    raw_markdown_dir = crawl_dir_base / "raw"
                    if raw_markdown_dir.is_dir():
                        # Load reverse lookup for URL mapping
                        reverse_lookup_path = crawl_dir_base / "reverse_lookup.json"
                        reverse_lookup = {}
                        if reverse_lookup_path.exists():
                            try:
                                with open(
                                    reverse_lookup_path, "r", encoding="utf-8"
                                ) as f:
                                    reverse_lookup = json.load(f)
                            except Exception as e:
                                logger.warning(
                                    f"Could not load reverse lookup file: {e}"
                                )

                        added_total, processed_files = 0, 0
                        all_files = list(raw_markdown_dir.glob("*.md"))
                        logger.info(f"Found {len(all_files)} markdown files to index.")

                        for md_file in all_files:
                            if shutdown_requested:
                                logger.warning("Indexing interrupted by signal.")
                                break  # Exit loop if interrupted
                            file_hash = md_file.stem
                            # Use path as fallback URL if hash not in lookup
                            original_url = reverse_lookup.get(
                                file_hash, md_file.as_uri()
                            )
                            logger.info(f"Indexing: {original_url} ({md_file.name})")
                            # Use internal_call=True for crawled files
                            added_count_result, was_blocked = (
                                llmsearch_instance_global.add_source(
                                    str(md_file), internal_call=True
                                )
                            )
                            if not was_blocked:
                                added_total += added_count_result
                            else:
                                # This case should ideally not happen for internal calls
                                logger.warning(
                                    f"Indexing blocked for internal call? {md_file.name}"
                                )
                            processed_files += 1  # Count as processed

                        logger.info(
                            f"Indexing complete. Added {added_total} new chunks from {processed_files} files."
                        )
                    else:
                        logger.warning(
                            f"Crawl 'raw' directory not found: {raw_markdown_dir}. No indexing performed."
                        )

                except KeyboardInterrupt:  # User Ctrl+C
                    logger.warning(
                        "CLI crawl-index aborted by user (KeyboardInterrupt)."
                    )
                    sys.exit(130)
                except Exception as e:
                    logger.error(f"Crawl/Index error: {e}", exc_info=args.debug)
                    sys.exit(1)

            elif args.command == "add":
                if not llmsearch_instance_global:
                    logger.error("LLMSearch instance not available for add.")
                    sys.exit(1)
                target_path = Path(args.path).resolve()
                logger.info(f"Adding and indexing source: {target_path}")
                try:
                    # add_source returns (count, blocked)
                    added_count, was_blocked = llmsearch_instance_global.add_source(
                        str(target_path)
                    )
                    if was_blocked:
                        logger.warning(
                            f"Adding source '{target_path}' skipped: Cannot add from managed crawl directory."
                        )
                    else:
                        logger.info(f"Finished adding. Added {added_count} new chunks.")

                    if shutdown_requested:
                        logger.warning("Add command possibly interrupted by signal.")
                except KeyboardInterrupt:
                    logger.warning("CLI add aborted by user (KeyboardInterrupt).")
                    sys.exit(130)
                except Exception as e:
                    logger.error(f"Add error: {e}", exc_info=args.debug)
                    sys.exit(1)

            elif args.command == "search":
                if not llmsearch_instance_global:
                    logger.error("LLMSearch instance not available for search.")
                    sys.exit(1)
                logger.info(f"Searching for: '{args.query}'")
                try:
                    # llm_query returns a dict with response, debug info etc.
                    res = llmsearch_instance_global.llm_query(
                        args.query, debug_mode=args.debug
                    )
                    print("\n--- AI Answer ---")
                    print(res.get("response", "N/A."))

                    # Print debug info if requested
                    if args.debug:
                        print("\n--- Retrieved Context (HTML) ---")
                        # 'retrieved_context' contains HTML string from query_processor
                        print(res.get("retrieved_context", "N/A."))
                        print("\n--- Debug Info ---")
                        debug_data = res.get("debug_info", {})
                        # Use default=str for non-serializable types like Path
                        print(json.dumps(debug_data, indent=2, default=str))

                    if shutdown_requested:
                        logger.warning("Search command interrupted by signal.")
                except KeyboardInterrupt:
                    logger.warning("CLI search aborted by user (KeyboardInterrupt).")
                    sys.exit(130)
                except Exception as e:
                    logger.error(f"Search error: {e}", exc_info=args.debug)
                    sys.exit(1)

            elif args.command == "list":
                if not llmsearch_instance_global:
                    logger.error("LLMSearch instance not available for list.")
                    sys.exit(1)
                logger.info("Listing indexed sources...")
                try:
                    sources = llmsearch_instance_global.get_indexed_sources()
                    if not sources:
                        print("No sources currently indexed.")
                    else:
                        print(f"\n--- Indexed Sources ({len(sources)}) ---")
                        for item in sources:
                            ident = item.get("identifier", "N/A")
                            is_url = item.get("is_url_source", False)
                            path_disp = item.get("source_path", "N/A")
                            url_disp = item.get("original_url")  # Can be None
                            filename = item.get("filename", "N/A")

                            print(f"- Identifier: {ident}")
                            print(f"  Type: {'URL' if is_url else 'Local Path'}")
                            # Show URL if available and it's a URL source
                            if is_url and url_disp:
                                print(f"  URL: {url_disp}")
                            # Show path if available (might be present for URL sources too)
                            if path_disp != "N/A":
                                print(f"  Path: {path_disp}")
                            print(f"  Filename: {filename}")
                            print(f"  Chunks: {item.get('chunk_count', 'N/A')}")
                            mtime_val = item.get("mtime")
                            mtime_str = "N/A"
                            if mtime_val is not None:
                                try:
                                    mtime_str = time.strftime(
                                        "%Y-%m-%d %H:%M:%S", time.localtime(mtime_val)
                                    )
                                except (OSError, ValueError):
                                    mtime_str = (
                                        "(Invalid Date)"  # Handle potential errors
                                    )
                            print(f"  Modified: {mtime_str}")
                            print("-" * 10)
                except Exception as e:
                    logger.error(f"List error: {e}", exc_info=args.debug)
                    sys.exit(1)

            elif args.command == "remove":
                if not llmsearch_instance_global:
                    logger.error("LLMSearch instance not available for remove.")
                    sys.exit(1)
                # Use the correct argument name from the parser
                source_id_to_remove = args.source_identifier
                logger.info(f"Removing source identifier: {source_id_to_remove}")
                try:
                    # remove_source returns (removed_bool, blocked_bool)
                    success, _ = llmsearch_instance_global.remove_source(
                        source_id_to_remove
                    )
                    if success:
                        logger.info(
                            f"Successfully removed source: {source_id_to_remove}"
                        )
                    else:
                        # Could be not found or another error during removal
                        logger.warning(
                            f"Source not found or removal failed: {source_id_to_remove}"
                        )
                except Exception as e:
                    logger.error(f"Remove error: {e}", exc_info=args.debug)
                    sys.exit(1)

            elif args.command == "export-index":
                index_dir_path_str = data_manager.get_data_paths().get("index")
                if not index_dir_path_str:
                    raise SetupError("Index dir path not configured.")
                index_dir_base = Path(index_dir_path_str)
                output_path = Path(args.output).resolve()

                if not str(output_path).endswith(".tar.gz"):
                    logger.error("Output filename must end with .tar.gz")
                    sys.exit(1)
                if output_path.exists():
                    logger.warning(f"Output file exists: {output_path}. Overwriting.")
                # Check if index directory exists and is not empty
                if not index_dir_base.is_dir() or not any(index_dir_base.iterdir()):
                    logger.error(
                        f"Index directory '{index_dir_base}' not found or is empty."
                    )
                    sys.exit(1)

                try:
                    logger.info(
                        f"Exporting index from {index_dir_base} to {output_path}..."
                    )
                    # Close LLMSearch before exporting to ensure files are not locked
                    if llmsearch_instance_global:
                        logger.info("Closing LLMSearch instance before export...")
                        llmsearch_instance_global.close()
                        # Clear global ref after closing
                        llmsearch_instance_global = None

                    # Create the tar.gz archive
                    with tarfile.open(output_path, "w:gz") as tar:
                        # Add the contents of the index directory
                        # arcname ensures the directory itself is the root inside the tar
                        tar.add(str(index_dir_base), arcname=index_dir_base.name)
                    logger.info(f"Index exported successfully to {output_path}")
                except Exception as e:
                    logger.error(f"Export error: {e}", exc_info=args.debug)
                    sys.exit(1)

            elif args.command == "set":
                try:
                    # Use data_manager to set and save the path
                    data_manager.set_data_path(args.key, args.path)
                    logger.info(
                        f"Set '{args.key}' path to '{args.path}'. Changes saved."
                    )
                except ValueError as e:  # Catch specific error for invalid key
                    logger.error(f"Set path error: {e}")
                    sys.exit(1)
                except Exception as e:
                    logger.error(f"Set path error: {e}", exc_info=args.debug)
                    sys.exit(1)
            # else: # This case should be unreachable due to check at start of CLI mode
            #     logger.error(f"Invalid CLI subcommand: '{args.command}'.")
            #     parser.print_help()
            #     sys.exit(1)

        except KeyboardInterrupt:
            # Catch Ctrl+C during command execution
            logger.warning("CLI operation aborted by user (KeyboardInterrupt).")
            sys.exit(130)
        except SystemExit as e:
            # Allow clean exits (e.g., from sys.exit calls within command logic)
            raise e
        except Exception as e:
            # Catch any other unexpected errors during CLI command execution
            logger.error(f"Unexpected CLI error: {e}", exc_info=args.debug)
            sys.exit(1)
        finally:
            # Ensure LLMSearch is closed if it was initialized for CLI
            if llmsearch_instance_global:
                logger.debug("Closing LLMSearch in CLI finally block...")
                llmsearch_instance_global.close()
                logger.debug("LLMSearch closed.")

    else:
        # Should not be reachable if mode choices are enforced by argparse
        logger.error("Invalid mode selected.")
        parser.print_help()
        sys.exit(1)


# Entry point guard
if __name__ == "__main__":
    main()
