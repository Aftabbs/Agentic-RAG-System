"""
Input validation to filter malicious, inappropriate, or unsafe queries.
"""

import re
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class InputValidator:
    """Validates user queries for safety and appropriateness."""

    # Patterns to detect potential issues
    SQL_INJECTION_PATTERNS = [
        r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|EXECUTE)\b)",
        r"(;|\-\-|\/\*|\*\/|xp_|sp_)"
    ]

    PII_PATTERNS = [
        r"\b\d{3}-\d{2}-\d{4}\b",  # SSN
        r"\b\d{16}\b",  # Credit card
        r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",  # Email (basic)
    ]

    PROMPT_INJECTION_PATTERNS = [
        r"ignore (previous|above|all) (instructions|rules)",
        r"you are now",
        r"system prompt",
        r"disregard"
    ]

    MAX_QUERY_LENGTH = 1000

    @staticmethod
    def validate_query(query: str) -> Dict[str, Any]:
        """
        Validate user query for safety issues.

        Args:
            query: User query string

        Returns:
            Dictionary with 'valid' boolean and 'reason' if invalid
        """
        # Check length
        if len(query) > InputValidator.MAX_QUERY_LENGTH:
            return {
                'valid': False,
                'reason': f'Query too long (max {InputValidator.MAX_QUERY_LENGTH} characters)'
            }

        # Check for empty query
        if not query.strip():
            return {
                'valid': False,
                'reason': 'Query cannot be empty'
            }

        # Check for SQL injection attempts
        for pattern in InputValidator.SQL_INJECTION_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                logger.warning(f"SQL injection attempt detected: {query[:50]}")
                return {
                    'valid': False,
                    'reason': 'Query contains potentially malicious SQL patterns'
                }

        # Check for PII (warn but don't block - might be legitimate)
        for pattern in InputValidator.PII_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                logger.warning(f"PII detected in query: {query[:50]}")
                # Note: In production, you might want to block or redact

        # Check for prompt injection
        for pattern in InputValidator.PROMPT_INJECTION_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                logger.warning(f"Prompt injection attempt: {query[:50]}")
                return {
                    'valid': False,
                    'reason': 'Query contains suspicious prompt manipulation patterns'
                }

        return {'valid': True, 'reason': ''}

    @staticmethod
    def sanitize_query(query: str) -> str:
        """
        Sanitize query by removing potentially harmful characters.

        Args:
            query: Raw query string

        Returns:
            Sanitized query
        """
        # Remove excessive whitespace
        query = ' '.join(query.split())

        # Remove control characters
        query = ''.join(char for char in query if ord(char) >= 32 or char == '\n')

        return query.strip()
