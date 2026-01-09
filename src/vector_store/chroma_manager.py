"""
ChromaDB manager for vector storage and retrieval.
Handles persistence, indexing, and similarity search.
"""

from typing import List, Dict, Any, Optional
from langchain_chroma import Chroma
from langchain.schema import Document
from config import config
from src.vector_store.embeddings import embedding_model
import logging
import os

logger = logging.getLogger(__name__)

class ChromaManager:
    """Manages ChromaDB operations for document storage and retrieval."""

    def __init__(self):
        """Initialize ChromaDB with persistent storage."""
        # Ensure persist directory exists
        os.makedirs(config.vector_store.persist_dir, exist_ok=True)

        self.embeddings = embedding_model.get_embeddings()
        self.collection_name = config.vector_store.collection_name

        # Initialize vector store
        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=config.vector_store.persist_dir
        )

        logger.info(f"ChromaDB initialized at {config.vector_store.persist_dir}")

    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add documents to the vector store.

        Args:
            documents: List of Document objects to add

        Returns:
            List of document IDs
        """
        if not documents:
            logger.warning("No documents to add")
            return []

        try:
            ids = self.vector_store.add_documents(documents)
            logger.info(f"Added {len(documents)} documents to vector store")
            return ids
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            raise

    def similarity_search(
        self,
        query: str,
        k: int = None,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Perform similarity search.

        Args:
            query: Search query
            k: Number of results (defaults to config)
            filter: Metadata filter dictionary

        Returns:
            List of most similar documents
        """
        k = k or config.retrieval.top_k

        try:
            results = self.vector_store.similarity_search(
                query,
                k=k,
                filter=filter
            )
            logger.info(f"Retrieved {len(results)} documents for query")
            return results
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            return []

    def similarity_search_with_score(
        self,
        query: str,
        k: int = None,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[tuple]:
        """
        Perform similarity search with relevance scores.

        Args:
            query: Search query
            k: Number of results
            filter: Metadata filter

        Returns:
            List of (document, score) tuples
        """
        k = k or config.retrieval.top_k

        try:
            results = self.vector_store.similarity_search_with_score(
                query,
                k=k,
                filter=filter
            )

            # Filter by similarity threshold
            threshold = config.retrieval.similarity_threshold
            filtered_results = [
                (doc, score) for doc, score in results
                if score >= threshold
            ]

            logger.info(
                f"Retrieved {len(filtered_results)} documents above "
                f"threshold {threshold}"
            )
            return filtered_results
        except Exception as e:
            logger.error(f"Error in similarity search with score: {str(e)}")
            return []

    def delete_collection(self):
        """Delete the entire collection (use with caution)."""
        try:
            self.vector_store.delete_collection()
            logger.warning(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error deleting collection: {str(e)}")
            raise

    def get_collection_count(self) -> int:
        """Get the number of documents in the collection."""
        try:
            collection = self.vector_store._collection
            return collection.count()
        except Exception as e:
            logger.error(f"Error getting collection count: {str(e)}")
            return 0
