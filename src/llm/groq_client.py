"""
Groq API client wrapper with retry logic and error handling.
"""

from langchain_groq import ChatGroq
from config import config
from tenacity import retry, stop_after_attempt, wait_exponential
import logging

logger = logging.getLogger(__name__)

class GroqClient:
    """Wrapper for Groq LLM with robust error handling."""

    def __init__(self):
        """Initialize Groq client."""
        self.llm = ChatGroq(
            groq_api_key=config.api.groq_api_key,
            model_name=config.api.groq_model,
            temperature=0.1,  # Lower temperature for factual responses
            max_tokens=2048
        )
        logger.info(f"Groq client initialized with model: {config.api.groq_model}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def invoke(self, prompt: str) -> str:
        """
        Invoke the LLM with retry logic.

        Args:
            prompt: Input prompt

        Returns:
            LLM response as string
        """
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            logger.error(f"Groq API error: {str(e)}")
            raise

    def get_llm(self):
        """Get the underlying LLM instance for LangChain integration."""
        return self.llm

# Global instance
groq_client = GroqClient()
