"""
Hallucination detection to ensure responses are grounded in context.
"""

from typing import Dict, Any, List, Optional
from langchain.schema import Document
from src.llm.groq_client import groq_client
from config import config
import logging

logger = logging.getLogger(__name__)

class HallucinationDetector:
    """Detects if LLM responses are grounded in provided context."""

    def __init__(self):
        self.llm = groq_client.get_llm()
        self.threshold = config.guardrails.hallucination_threshold

    def check_grounding(
        self,
        query: str,
        response: str,
        context: Optional[str] = None,
        source_type: str = "rag"
    ) -> Dict[str, Any]:
        """
        Check if response is grounded in context.

        Args:
            query: Original user query
            response: Generated response
            context: Source context (if applicable)
            source_type: One of "rag", "llm", "search"

        Returns:
            Dictionary with is_grounded boolean and confidence score
        """
        # Skip grounding check for LLM knowledge and search (they're inherently ungrounded)
        if source_type in ["llm", "search"]:
            return {
                'is_grounded': True,
                'confidence': 1.0,
                'reason': f'{source_type.upper()} responses are not grounded in local context'
            }

        # For RAG, context is required
        if not context:
            logger.warning("No context provided for grounding check")
            return {
                'is_grounded': False,
                'confidence': 0.0,
                'reason': 'No context available'
            }

        prompt = f"""You are a factual grounding evaluator. Determine if the response is fully supported by the provided context.

Question: {query}

Context:
{context}

Response to evaluate:
{response}

Evaluate if EVERY claim in the response can be directly found in or reasonably inferred from the context.

Rate grounding on a scale of 0.0 to 1.0 where:
- 1.0 = All claims are fully supported by context
- 0.5 = Some claims are supported, some are not
- 0.0 = Response contains information not in context (hallucination)

Respond with ONLY a number between 0.0 and 1.0, nothing else."""

        try:
            result = self.llm.invoke(prompt)
            confidence = float(result.content.strip())
            confidence = max(0.0, min(1.0, confidence))

            is_grounded = confidence >= self.threshold

            logger.info(
                f"Grounding confidence: {confidence:.2f} "
                f"(threshold: {self.threshold})"
            )

            return {
                'is_grounded': is_grounded,
                'confidence': confidence,
                'reason': f'Confidence {confidence:.2f} vs threshold {self.threshold}'
            }
        except (ValueError, AttributeError) as e:
            logger.error(f"Error in grounding check: {str(e)}")
            return {
                'is_grounded': True,
                'confidence': 0.5,
                'reason': 'Error in grounding check, defaulting to grounded'
            }
