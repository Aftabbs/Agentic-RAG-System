"""
Logging system for queries, responses, and metrics.
"""

import logging
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, Any
import os

class QueryLogger:
    """Logs all queries and responses for evaluation."""

    def __init__(self, log_dir: str = './logs'):
        """
        Initialize query logger.

        Args:
            log_dir: Directory for log files
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.query_log_file = self.log_dir / 'queries.jsonl'
        self.eval_log_file = self.log_dir / 'evaluations.jsonl'

    def log_query(self, query_data: Dict[str, Any]):
        """
        Log a query and its response.

        Args:
            query_data: Dictionary with query, response, and metadata
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'query': query_data.get('query', ''),
            'response': query_data.get('response', ''),
            'source_type': query_data.get('source_type', ''),
            'sources': query_data.get('sources', []),
            'processing_time': query_data.get('processing_time', 0.0),
            'is_grounded': query_data.get('is_grounded', True),
            'grounding_confidence': query_data.get('grounding_confidence', 1.0),
            'relevance_score': query_data.get('relevance_score', 0.0),
            'error': query_data.get('error', None)
        }

        with open(self.query_log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry) + '\n')

    def log_evaluation(self, eval_data: Dict[str, Any]):
        """
        Log evaluation metrics.

        Args:
            eval_data: Dictionary with evaluation metrics
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            **eval_data
        }

        with open(self.eval_log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry) + '\n')
