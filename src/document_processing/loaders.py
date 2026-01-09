"""
Document loaders for PDF, DOCX, and TXT files.
Handles extraction with error handling and metadata preservation.
"""

from typing import List, Dict, Any
from pathlib import Path
import logging
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader
)
from langchain.schema import Document

logger = logging.getLogger(__name__)

class DocumentLoader:
    """Unified document loader supporting multiple formats."""

    SUPPORTED_EXTENSIONS = {'.pdf', '.docx', '.txt'}

    @staticmethod
    def load_document(file_path: str) -> List[Document]:
        """
        Load a document and return LangChain Document objects.

        Args:
            file_path: Path to the document

        Returns:
            List of Document objects with content and metadata
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        extension = path.suffix.lower()

        if extension not in DocumentLoader.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {extension}")

        try:
            if extension == '.pdf':
                loader = PyPDFLoader(str(path))
            elif extension == '.docx':
                loader = Docx2txtLoader(str(path))
            else:  # .txt
                loader = TextLoader(str(path), encoding='utf-8')

            documents = loader.load()

            # Enrich metadata
            for doc in documents:
                doc.metadata['source_file'] = path.name
                doc.metadata['file_type'] = extension[1:]  # Remove dot
                doc.metadata['file_path'] = str(path.absolute())

            logger.info(f"Loaded {len(documents)} pages from {path.name}")
            return documents

        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
            raise

    @staticmethod
    def load_multiple_documents(file_paths: List[str]) -> List[Document]:
        """
        Load multiple documents and combine them.

        Args:
            file_paths: List of file paths

        Returns:
            Combined list of Document objects
        """
        all_documents = []

        for file_path in file_paths:
            try:
                docs = DocumentLoader.load_document(file_path)
                all_documents.extend(docs)
            except Exception as e:
                logger.warning(f"Skipping {file_path}: {str(e)}")
                continue

        return all_documents
