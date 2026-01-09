"""
Tool definitions for RAG, LLM knowledge, and internet search.
"""

from typing import List, Dict, Any
from langchain.schema import Document
from langchain.tools import Tool
from src.vector_store.chroma_manager import ChromaManager
from src.llm.groq_client import groq_client
from src.search.serper_client import serper_client
import logging

logger = logging.getLogger(__name__)

class AgentTools:
    """Manages all tools available to the agent."""

    def __init__(self, chroma_manager: ChromaManager):
        """
        Initialize agent tools.

        Args:
            chroma_manager: ChromaDB manager instance
        """
        self.chroma = chroma_manager
        self.llm = groq_client.get_llm()
        self.search = serper_client

    def rag_tool(self, query: str) -> Dict[str, Any]:
        """
        RAG tool: Search uploaded documents.

        Args:
            query: Search query

        Returns:
            Dictionary with documents and metadata
        """
        try:
            results = self.chroma.similarity_search_with_score(query)

            documents = [doc for doc, score in results]
            scores = [score for doc, score in results]

            sources = []
            for doc, score in results:
                sources.append({
                    'type': 'document',
                    'file': doc.metadata.get('source_file', 'Unknown'),
                    'page': doc.metadata.get('page', 'N/A'),
                    'score': score,
                    'content': doc.page_content[:200]  # Preview
                })

            logger.info(f"RAG tool retrieved {len(documents)} documents")

            return {
                'documents': documents,
                'scores': scores,
                'sources': sources,
                'success': len(documents) > 0
            }
        except Exception as e:
            logger.error(f"RAG tool error: {str(e)}")
            return {
                'documents': [],
                'scores': [],
                'sources': [],
                'success': False,
                'error': str(e)
            }

    def llm_tool(self, query: str) -> Dict[str, Any]:
        """
        LLM tool: Use model's internal knowledge.

        Args:
            query: User query

        Returns:
            Dictionary with response and metadata
        """
        try:
            prompt = f"""You are a helpful AI assistant. Answer the following question using your knowledge. Be concise and accurate.

Question: {query}

Answer:"""

            response = self.llm.invoke(prompt)

            return {
                'response': response.content,
                'sources': [{
                    'type': 'llm_knowledge',
                    'model': 'Groq LLM',
                    'note': 'Response based on model training data'
                }],
                'success': True
            }
        except Exception as e:
            logger.error(f"LLM tool error: {str(e)}")
            return {
                'response': '',
                'sources': [],
                'success': False,
                'error': str(e)
            }

    def search_tool(self, query: str) -> Dict[str, Any]:
        """
        Search tool: Internet search via Serper.

        Args:
            query: Search query

        Returns:
            Dictionary with search results and metadata
        """
        try:
            results = self.search.search_with_metadata(query)

            sources = []
            for result in results:
                sources.append({
                    'type': 'internet_search',
                    'title': result['title'],
                    'url': result['link'],
                    'snippet': result['snippet']
                })

            # Combine snippets for context
            combined_context = "\n\n".join([
                f"{r['title']}: {r['snippet']}" for r in results
            ])

            logger.info(f"Search tool retrieved {len(results)} results")

            return {
                'results': results,
                'context': combined_context,
                'sources': sources,
                'success': len(results) > 0
            }
        except Exception as e:
            logger.error(f"Search tool error: {str(e)}")
            return {
                'results': [],
                'context': '',
                'sources': [],
                'success': False,
                'error': str(e)
            }
