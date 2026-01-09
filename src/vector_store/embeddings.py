"""
Embedding model wrapper using HuggingFace sentence-transformers.
Provides caching and error handling.
"""

from langchain_huggingface import HuggingFaceEmbeddings
from config import config
import logging

logger = logging.getLogger(__name__)

class EmbeddingModel:
    """Singleton wrapper for embedding model."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        logger.info(f"Loading embedding model: {config.embedding.model_name}")

        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.embedding.model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        self._initialized = True
        logger.info("Embedding model loaded successfully")

    def get_embeddings(self):
        """Get the embeddings instance."""
        return self.embeddings

# Global instance
embedding_model = EmbeddingModel()
