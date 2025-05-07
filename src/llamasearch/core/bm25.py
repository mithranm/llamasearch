# src/llamasearch/core/bm25.py

import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

# --- Whoosh Imports ---
from whoosh import index as whoosh_index
from whoosh.analysis import StandardAnalyzer
from whoosh.fields import ID, TEXT, SchemaClass
from whoosh.qparser import QueryParser  # Using simple QueryParser
from whoosh.scoring import BM25F  # BM25 Scoring algorithm

from llamasearch.utils import setup_logging

# Removed AsyncWriter and FileLock imports


# Use the global logger setup
logger = setup_logging(__name__, use_qt_handler=True)


# --- Whoosh Schema Definition ---
class BM25Schema(SchemaClass):
    chunk_id = ID(unique=True, stored=True)
    content = TEXT(analyzer=StandardAnalyzer(), stored=False)


DEFAULT_WRITER_TIMEOUT = 60.0  # Timeout for acquiring writer lock


class WhooshBM25Retriever:
    """
    BM25Retriever implementation using the Whoosh library for efficient indexing and search.
    Relies on Whoosh's internal file locking via the writer.
    """

    def __init__(self, storage_dir: Path) -> None:
        self.storage_dir = storage_dir
        logger.debug("BM25: Creating storage directory if needed...")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        logger.debug("BM25: Instantiating BM25Schema...")
        self.schema = BM25Schema()
        logger.debug("BM25: BM25Schema instantiated.")
        self.index_path = self.storage_dir / "whoosh_bm25_index"
        self.ix: Optional[whoosh_index.Index] = None
        self.parser: Optional[QueryParser] = None
        self._open_or_create_index()

    def _open_or_create_index(self) -> None:
        """Opens an existing Whoosh index or creates a new one."""
        # Locking is handled by the writer context manager later
        try:
            logger.debug(f"BM25: Checking if index exists at: {self.index_path}")
            index_exists = whoosh_index.exists_in(str(self.index_path))
            logger.debug(f"BM25: Index exists: {index_exists}")

            if index_exists:
                logger.info(f"Opening existing Whoosh index at: {self.index_path}")
                try:
                    logger.debug("BM25: Attempting whoosh_index.open_dir...")
                    self.ix = whoosh_index.open_dir(
                        str(self.index_path), schema=self.schema
                    )
                    logger.debug("BM25: whoosh_index.open_dir successful.")
                except whoosh_index.EmptyIndexError:
                    logger.warning(
                        f"Whoosh index at {self.index_path} exists but is empty/corrupt. Recreating."
                    )
                    shutil.rmtree(self.index_path)  # Remove corrupt index
                    self.index_path.mkdir(parents=True, exist_ok=True)  # Recreate dir
                    logger.debug("BM25: Attempting whoosh_index.create_in...")
                    self.ix = whoosh_index.create_in(str(self.index_path), self.schema)
                    logger.debug("BM25: whoosh_index.create_in successful.")
                except Exception as open_err:
                    logger.error(
                        f"Error opening existing Whoosh index {self.index_path}, attempting recreation: {open_err}",
                        exc_info=True,
                    )
                    try:
                        shutil.rmtree(self.index_path)
                        self.index_path.mkdir(
                            parents=True, exist_ok=True
                        )  # Recreate dir
                        logger.debug("BM25: Attempting whoosh_index.create_in...")
                        self.ix = whoosh_index.create_in(
                            str(self.index_path), self.schema
                        )
                        logger.debug("BM25: whoosh_index.create_in successful.")
                        logger.info(f"Recreated Whoosh index at: {self.index_path}")
                    except Exception as recreate_err:
                        logger.critical(
                            f"FATAL: Could not recreate Whoosh index after open failure: {recreate_err}",
                            exc_info=True,
                        )
                        raise RuntimeError(
                            "Failed to open or recreate Whoosh index"
                        ) from recreate_err
            else:
                logger.info(f"Creating new Whoosh index at: {self.index_path}")
                logger.debug("BM25: Ensuring index directory exists...")
                self.index_path.mkdir(parents=True, exist_ok=True)  # Ensure dir exists
                logger.debug("BM25: Attempting whoosh_index.create_in...")
                self.ix = whoosh_index.create_in(str(self.index_path), self.schema)
                logger.debug("BM25: whoosh_index.create_in successful.")

            logger.debug("BM25: Initializing QueryParser...")
            self.parser = QueryParser("content", schema=self.ix.schema)  # type: ignore
            logger.debug("BM25: QueryParser initialized.")
            logger.info(f"Whoosh index ready. Doc count: {self.get_doc_count()}")

        except Exception as e:
            logger.error(
                f"Failed to open or create Whoosh index at {self.index_path}: {e}",
                exc_info=True,
            )
            self.ix = None
            self.parser = None
            raise RuntimeError("Failed to initialize Whoosh index") from e

    def add_document(self, text: str, doc_id: str) -> bool:
        """Adds a document chunk to the Whoosh index."""
        if self.ix is None:
            logger.error("Cannot add document, Whoosh index is not initialized.")
            return False
        if not text or not doc_id:
            logger.warning(
                f"Skipping add_document: Empty text or doc_id provided (ID: '{doc_id}')."
            )
            return False

        try:
            # Use the index writer directly; it handles locking.
            writer = self.ix.writer(timeout=DEFAULT_WRITER_TIMEOUT)
            logger.debug("BM25: Attempting writer.update_document...")
            with writer:  # Context manager handles commit/cancel and locking
                writer.update_document(chunk_id=doc_id, content=text)
            logger.debug("BM25: writer.update_document successful.")
            logger.debug(f"Added/Updated document chunk_id '{doc_id}' in Whoosh index.")
            return True
        except whoosh_index.LockError as lock_err:
            logger.error(
                f"Failed to acquire lock for adding document chunk_id '{doc_id}': {lock_err}"
            )
            return False
        except Exception as e:
            logger.error(
                f"Failed to add document chunk_id '{doc_id}' to Whoosh index: {e}",
                exc_info=True,
            )
            return False

    def remove_document(self, doc_id: str) -> bool:
        """Removes a document from the Whoosh index by its unique chunk ID."""
        if self.ix is None:
            logger.error("Cannot remove document, Whoosh index is not initialized.")
            return False
        if not doc_id:
            logger.warning("Skipping remove_document: Empty doc_id provided.")
            return False

        try:
            writer = self.ix.writer(timeout=DEFAULT_WRITER_TIMEOUT)
            logger.debug("BM25: Attempting writer.delete_by_term...")
            num_deleted = 0
            with writer:  # Context manager handles commit/cancel and locking
                num_deleted = writer.delete_by_term("chunk_id", doc_id)
            logger.debug("BM25: writer.delete_by_term successful.")
            logger.debug(
                f"Attempted removal of document chunk_id '{doc_id}' from Whoosh index (deleted {num_deleted} segment docs)."
            )
            return True  # Return true even if 0 were deleted
        except whoosh_index.LockError as lock_err:
            logger.error(
                f"Failed to acquire lock for removing document chunk_id '{doc_id}': {lock_err}"
            )
            return False
        except Exception as e:
            logger.error(
                f"Failed to remove document chunk_id '{doc_id}' from Whoosh index: {e}",
                exc_info=True,
            )
            return False

    def query(self, query_text: str, n_results: int = 5) -> Dict[str, Any]:
        """Performs a BM25F query using Whoosh."""
        empty_res = {"query": query_text, "ids": [], "scores": [], "documents": []}
        if self.ix is None or self.parser is None:
            logger.error("Cannot query, Whoosh index or parser not initialized.")
            return empty_res
        if not query_text:
            logger.debug("BM25 query text is empty.")
            return empty_res

        try:
            query_obj = self.parser.parse(query_text)

            with self.ix.searcher(weighting=BM25F()) as searcher:
                logger.debug("BM25: Attempting searcher.search...")
                results = searcher.search(query_obj, limit=n_results)
                logger.debug("BM25: searcher.search successful.")
                top_ids: List[str] = []
                top_scores: List[float] = []
                top_docs: List[None] = []  # Documents are not stored

                if results:
                    for hit in results:
                        chunk_id = hit.get("chunk_id")
                        # --- Check hit.score before appending ---
                        score = hit.score
                        if chunk_id and score is not None:
                            top_ids.append(chunk_id)
                            top_scores.append(float(score))  # Cast score to float
                            top_docs.append(None)
                        # ----------------------------------------
                        else:
                            logger.warning(
                                f"Whoosh hit missing 'chunk_id' or 'score': {hit}"
                            )

                logger.debug(f"Whoosh BM25 query returned {len(top_ids)} results.")
                return {
                    "query": query_text,
                    "ids": top_ids,
                    "scores": top_scores,
                    "documents": top_docs,  # List of None
                }
        except Exception as e:
            logger.error(f"Whoosh query failed: {e}", exc_info=True)
            return empty_res

    def get_doc_count(self) -> int:
        """Returns the number of documents currently in the Whoosh index."""
        if self.ix is None:
            return 0
        try:
            # Use the main index doc count for a quick estimate
            logger.debug("BM25: Attempting ix.doc_count...")
            doc_count = self.ix.doc_count()
            logger.debug("BM25: ix.doc_count successful.")
            return doc_count
        except Exception as e:
            logger.error(f"Failed to get Whoosh doc count: {e}")
            return 0

    def save(self) -> None:
        """Whoosh handles persistence automatically on commit. This is a no-op."""
        logger.debug("Whoosh index persistence is handled internally on add/remove.")
        pass

    def close(self) -> None:
        """Closes the Whoosh index."""
        logger.info("Closing Whoosh index...")
        if self.ix is not None:
            try:
                logger.debug("BM25: Attempting ix.close...")
                self.ix.close()
                logger.debug("BM25: ix.close successful.")
            except Exception as e:
                logger.error(f"Error closing Whoosh index: {e}")
            finally:  # Ensure these are set to None even if close() fails or succeeds
                self.ix = None
                self.parser = None
                logger.info("Whoosh index resources released/nulled.")
        else:
            logger.info("Whoosh index was already None or closed.")
