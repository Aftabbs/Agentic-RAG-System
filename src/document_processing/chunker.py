"""
Text chunking strategies for optimal retrieval.
Uses recursive character splitting with configurable overlap.
"""

from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import logging
from config import config

logger = logging.getLogger(__name__)

class DocumentChunker:
    """Handles intelligent document chunking with metadata preservation."""

    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None
    ):
        """
        Initialize chunker with configuration.

        Args:
            chunk_size: Size of each chunk (defaults to config)
            chunk_overlap: Overlap between chunks (defaults to config)
        """
        self.chunk_size = chunk_size or config.chunking.chunk_size
        self.chunk_overlap = chunk_overlap or config.chunking.chunk_overlap

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Chunk documents while preserving metadata.

        Args:
            documents: List of LangChain Document objects

        Returns:
            List of chunked Document objects with enhanced metadata
        """
        chunks = self.text_splitter.split_documents(documents)

        # Add chunk-specific metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata['chunk_id'] = i
            chunk.metadata['chunk_size'] = len(chunk.page_content)

        logger.info(
            f"Created {len(chunks)} chunks from {len(documents)} documents "
            f"(size={self.chunk_size}, overlap={self.chunk_overlap})"
        )

        return chunks
