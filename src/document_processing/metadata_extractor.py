"""
Metadata extraction and enrichment for documents.
Adds timestamps, file info, and custom metadata.
"""

from typing import Dict, Any, List
from datetime import datetime
from langchain.schema import Document
import hashlib

class MetadataExtractor:
    """Extracts and enriches document metadata."""

    @staticmethod
    def add_timestamp(documents: List[Document]) -> List[Document]:
        """Add upload timestamp to documents."""
        timestamp = datetime.now().isoformat()

        for doc in documents:
            doc.metadata['upload_timestamp'] = timestamp

        return documents

    @staticmethod
    def add_document_hash(documents: List[Document]) -> List[Document]:
        """Add content hash for deduplication."""
        for doc in documents:
            content_hash = hashlib.md5(
                doc.page_content.encode()
            ).hexdigest()
            doc.metadata['content_hash'] = content_hash

        return documents

    @staticmethod
    def extract_page_info(document: Document) -> Dict[str, Any]:
        """Extract page number information if available."""
        metadata = document.metadata

        page_info = {
            'page_number': metadata.get('page', 'N/A'),
            'source_file': metadata.get('source_file', 'Unknown'),
            'file_type': metadata.get('file_type', 'Unknown')
        }

        return page_info

    @staticmethod
    def enrich_all_metadata(documents: List[Document]) -> List[Document]:
        """Apply all metadata enrichment steps."""
        documents = MetadataExtractor.add_timestamp(documents)
        documents = MetadataExtractor.add_document_hash(documents)
        return documents
