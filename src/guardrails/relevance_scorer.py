"""
Relevance scoring to determine if retrieved documents answer the query.
"""

from typing import List, Dict, Any
from langchain.schema import Document
from src.llm.groq_client import groq_client
from config import config
import logging

logger = logging.getLogger(__name__)

class RelevanceScorer:
    """Scores relevance of retrieved documents to query."""

    def __init__(self):
        self.llm = groq_client.get_llm()
        self.threshold = config.guardrails.relevance_threshold

    def score_relevance(
        self,
        query: str,
        documents: List[Document]
    ) -> Dict[str, Any]:
        """
        Score if documents are relevant to answering the query.

        Args:
            query: User query
            documents: Retrieved documents

        Returns:
            Dictionary with score and is_relevant boolean
        """
        if not documents:
            return {'score': 0.0, 'is_relevant': False, 'reason': 'No documents'}

        # Combine document content
        context = "\n\n".join([doc.page_content for doc in documents[:3]])

        prompt = f"""You are a relevance evaluator. Determine if the provided context can answer the user's question.

Question: {query}

Context:
{context}

Rate the relevance on a scale of 0.0 to 1.0 where:
- 1.0 = Context directly answers the question
- 0.5 = Context is somewhat related but doesn't fully answer
- 0.0 = Context is completely unrelated

Respond with ONLY a number between 0.0 and 1.0, nothing else."""

        try:
            response = self.llm.invoke(prompt)
            score = float(response.content.strip())
            score = max(0.0, min(1.0, score))  # Clamp to [0, 1]

            is_relevant = score >= self.threshold

            logger.info(f"Relevance score: {score:.2f} (threshold: {self.threshold})")

            return {
                'score': score,
                'is_relevant': is_relevant,
                'reason': f'Score {score:.2f} vs threshold {self.threshold}'
            }
        except (ValueError, AttributeError) as e:
            logger.error(f"Error parsing relevance score: {str(e)}")
            # Default to relevant to avoid false negatives
            return {
                'score': 0.5,
                'is_relevant': True,
                'reason': 'Error in scoring, defaulting to relevant'
            }
