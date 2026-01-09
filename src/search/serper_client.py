"""
Serper API client for internet search.
"""

from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.tools import Tool
from config import config
from tenacity import retry, stop_after_attempt, wait_exponential
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class SerperClient:
    """Wrapper for Serper API with error handling."""

    def __init__(self):
        """Initialize Serper client."""
        self.search = GoogleSerperAPIWrapper(
            serper_api_key=config.api.serper_api_key,
            k=5  # Top 5 results
        )
        logger.info("Serper client initialized")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def search_internet(self, query: str) -> str:
        """
        Perform internet search.

        Args:
            query: Search query

        Returns:
            Search results as formatted string
        """
        try:
            results = self.search.run(query)
            logger.info(f"Search completed for query: {query[:50]}...")
            return results
        except Exception as e:
            logger.error(f"Serper API error: {str(e)}")
            raise

    def search_with_metadata(self, query: str) -> List[Dict[str, Any]]:
        """
        Perform search and return structured results with URLs.

        Args:
            query: Search query

        Returns:
            List of result dictionaries with title, link, snippet
        """
        try:
            results = self.search.results(query)

            # Extract organic results
            organic_results = results.get('organic', [])

            formatted_results = []
            for result in organic_results[:5]:
                formatted_results.append({
                    'title': result.get('title', ''),
                    'link': result.get('link', ''),
                    'snippet': result.get('snippet', '')
                })

            logger.info(f"Retrieved {len(formatted_results)} search results")
            return formatted_results
        except Exception as e:
            logger.error(f"Serper API error: {str(e)}")
            return []

    def as_tool(self) -> Tool:
        """Return Serper as LangChain Tool."""
        return Tool(
            name="Internet Search",
            func=self.search_internet,
            description="Search the internet for current information, news, and facts not in the knowledge base"
        )

# Global instance
serper_client = SerperClient()
