"""
Vector store management for ChromaDB.

Handles initialization, embedding, and persistence of the movie/series dataset.
"""

import json
import logging
import inspect
from pathlib import Path
from typing import List, Dict, Any, Optional


def _patch_posthog_capture() -> None:
    """
    Patch posthog.capture to accept the legacy (distinct_id, event, properties) signature
    expected by chromadb, even when newer posthog versions only accept keyword args.
    """
    try:
        import posthog  # type: ignore

        signature = inspect.signature(posthog.capture)
        params = list(signature.parameters.values())
        # Newer posthog releases expose capture(event, **kwargs). Only patch in that case.
        if not params or params[0].name != "event":
            return
        if getattr(posthog.capture, "_patched_for_chromadb", False):
            return
        original_capture = posthog.capture

        def capture_compat(*args, **kwargs):
            # Skip telemetry entirely if no API key is configured (common when telemetry disabled)
            if not getattr(posthog, "project_api_key", None):
                return None
            # Legacy chromadb expects capture(distinct_id, event, properties)
            if len(args) == 3:
                distinct_id, event, properties = args
                kwargs.setdefault("distinct_id", distinct_id)
                kwargs.setdefault("properties", properties)
                return original_capture(event, **kwargs)
            if len(args) == 2:
                distinct_id, event = args
                kwargs.setdefault("distinct_id", distinct_id)
                return original_capture(event, **kwargs)
            return original_capture(*args, **kwargs)

        capture_compat._patched_for_chromadb = True  # type: ignore[attr-defined]
        posthog.capture = capture_compat
    except Exception:
        # If posthog is unavailable or signature inspection fails, skip patching.
        pass


_patch_posthog_capture()

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Manages ChromaDB vector store for content recommendations."""

    def __init__(self,
                 persist_directory: str = "./data/chroma_db",
                 collection_name: str = "content_collection",
                 openai_api_key: Optional[str] = None):
        """
        Initialize the vector store manager.

        Args:
            persist_directory: Directory for ChromaDB persistence
            collection_name: Name of the collection
            openai_api_key: OpenAI API key for embeddings
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name

        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Initialize embedding function
        if openai_api_key:
            self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
                api_key=openai_api_key,
                model_name="text-embedding-3-small"
            )
        else:
            # Use default embedding function for testing
            self.embedding_function = embedding_functions.DefaultEmbeddingFunction()

        self.collection = None

    def create_collection(self, reset: bool = False) -> None:
        """
        Create or get the collection.

        Args:
            reset: If True, delete existing collection and create new one
        """
        if reset:
            try:
                self.client.delete_collection(name=self.collection_name)
                logger.info(f"Deleted existing collection: {self.collection_name}")
            except Exception as e:
                logger.debug(f"No existing collection to delete: {e}")

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"Collection ready: {self.collection_name}")

    def create_document_text(self, item: Dict[str, Any]) -> str:
        """
        Create a rich text representation for embedding.

        Args:
            item: Content item dictionary

        Returns:
            Formatted text for embedding
        """
        parts = [
            f"Title: {item['title']}",
            f"Type: {item['type']}",
            f"Overview: {item['overview']}" if item.get('overview') else "",
            f"Genres: {', '.join(item['genres'])}" if item.get('genres') else "",
            f"Moods: {', '.join(item['moods'])}" if item.get('moods') else "",
            f"Year: {item['year']}" if item.get('year') else "",
            f"Rating: {item['tmdb_rating']}/10" if item.get('tmdb_rating') else "",
            f"Cast: {', '.join(item['cast'][:3])}" if item.get('cast') else "",
            f"Crew: {', '.join(item['crew'])}" if item.get('crew') else "",
        ]

        # Add tagline if available
        if item.get('tagline'):
            parts.append(f"Tagline: {item['tagline']}")

        return " | ".join([p for p in parts if p])

    def create_metadata(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create metadata for ChromaDB storage.

        ChromaDB supports: str, int, float, bool
        Lists and nested objects must be serialized.

        Args:
            item: Content item dictionary

        Returns:
            Metadata dictionary
        """
        metadata = {
            "content_id": str(item['id']),
            "type": item['type'],
            "title": item['title'],
            "year": str(item.get('year', '')),
            "tmdb_rating": float(item.get('tmdb_rating', 0.0)),
            "vote_count": int(item.get('vote_count', 0)),
            "runtime_minutes": int(item.get('runtime_minutes', 0)),
            "total_hours": float(item.get('total_hours', 0.0)),
            "language": item.get('language', ''),
            "popularity": float(item.get('popularity', 0.0)),
            "content_rating": item.get('content_rating', '') or '',
        }

        # Serialize complex fields as JSON strings
        metadata["genres"] = json.dumps(item.get('genres', []))
        metadata["moods"] = json.dumps(item.get('moods', []))
        metadata["keywords"] = json.dumps(item.get('keywords', []))
        metadata["context_suitability"] = json.dumps(item.get('context_suitability', {}))
        metadata["cast"] = json.dumps(item.get('cast', []))
        metadata["crew"] = json.dumps(item.get('crew', []))

        # Series-specific fields
        if item['type'] == 'series':
            metadata["seasons"] = int(item.get('seasons', 0))
            metadata["episodes"] = int(item.get('episodes', 0))

        return metadata

    def index_content(self, content_data: List[Dict[str, Any]], batch_size: int = 100) -> None:
        """
        Index content into ChromaDB.

        Args:
            content_data: List of content items
            batch_size: Number of items to process at once
        """
        if self.collection is None:
            raise ValueError("Collection not initialized. Call create_collection() first.")

        logger.info(f"Indexing {len(content_data)} items...")

        # Process in batches
        for i in range(0, len(content_data), batch_size):
            batch = content_data[i:i + batch_size]

            ids = [str(item['id']) for item in batch]
            documents = [self.create_document_text(item) for item in batch]
            metadatas = [self.create_metadata(item) for item in batch]

            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )

            logger.info(f"Indexed batch {i // batch_size + 1}/{(len(content_data) - 1) // batch_size + 1}")

        logger.info(f"Indexing complete. Total items: {self.collection.count()}")

    def search(self,
               query: str,
               n_results: int = 10,
               where: Optional[Dict[str, Any]] = None,
               where_document: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Search the vector store.

        Args:
            query: Search query text
            n_results: Number of results to return
            where: Metadata filter (e.g., {"type": "movie"})
            where_document: Document content filter

        Returns:
            Search results dictionary with ids, documents, metadatas, distances
        """
        if self.collection is None:
            raise ValueError("Collection not initialized.")

        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where,
            where_document=where_document
        )

        return results

    def get_by_ids(self, ids: List[str]) -> Dict[str, Any]:
        """
        Retrieve items by their IDs.

        Args:
            ids: List of content IDs

        Returns:
            Dictionary with ids, documents, metadatas
        """
        if self.collection is None:
            raise ValueError("Collection not initialized.")

        return self.collection.get(ids=ids)

    def count(self) -> int:
        """Get total number of items in collection."""
        if self.collection is None:
            return 0
        return self.collection.count()

    def reset(self) -> None:
        """Reset the collection (delete all data)."""
        if self.collection is not None:
            self.client.delete_collection(name=self.collection_name)
            logger.info("Collection deleted")
            self.collection = None


def load_processed_data(data_path: str = "./data/processed/content_dataset.json") -> List[Dict[str, Any]]:
    """
    Load processed content dataset.

    Args:
        data_path: Path to the processed JSON file

    Returns:
        List of content items
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def initialize_vector_store(openai_api_key: str,
                           data_path: str = "./data/processed/content_dataset.json",
                           reset: bool = False) -> VectorStoreManager:
    """
    Initialize and populate vector store with content data.

    Args:
        openai_api_key: OpenAI API key
        data_path: Path to processed data
        reset: Whether to reset existing collection

    Returns:
        Initialized VectorStoreManager
    """
    # Load data
    content_data = load_processed_data(data_path)
    logger.info(f"Loaded {len(content_data)} items from {data_path}")

    # Initialize vector store
    vector_store = VectorStoreManager(openai_api_key=openai_api_key)
    vector_store.create_collection(reset=reset)

    # Check if already populated
    if vector_store.count() > 0 and not reset:
        logger.info(f"Collection already contains {vector_store.count()} items. Skipping indexing.")
        return vector_store

    # Index content
    vector_store.index_content(content_data)

    return vector_store


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    # Load environment variables
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment")
        exit(1)

    # Initialize vector store
    logging.basicConfig(level=logging.INFO)
    vs = initialize_vector_store(api_key, reset=True)

    # Test search
    results = vs.search("funny family movie for kids", n_results=5)
    print("\nTest Search Results:")
    for i, (doc_id, metadata) in enumerate(zip(results['ids'][0], results['metadatas'][0])):
        print(f"{i+1}. {metadata['title']} ({metadata['year']}) - {metadata['type']}")
        print(f"   Rating: {metadata['tmdb_rating']}, Genres: {json.loads(metadata['genres'])}")
