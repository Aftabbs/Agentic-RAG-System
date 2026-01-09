"""
Metrics calculation for evaluation.
"""

from typing import List, Dict, Any
import json
from pathlib import Path
from datetime import datetime, timedelta

class MetricsCalculator:
    """Calculates evaluation metrics from logs."""

    @staticmethod
    def load_logs(log_file: str) -> List[Dict[str, Any]]:
        """Load logs from JSONL file."""
        logs = []

        if not Path(log_file).exists():
            return logs

        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    logs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        return logs

    @staticmethod
    def calculate_metrics(log_file: str, time_window_hours: int = 24) -> Dict[str, Any]:
        """
        Calculate metrics from query logs.

        Args:
            log_file: Path to query log file
            time_window_hours: Time window for metrics calculation

        Returns:
            Dictionary with calculated metrics
        """
        logs = MetricsCalculator.load_logs(log_file)

        if not logs:
            return {'error': 'No logs available'}

        # Filter by time window
        cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
        recent_logs = [
            log for log in logs
            if datetime.fromisoformat(log['timestamp']) >= cutoff_time
        ]

        if not recent_logs:
            return {'error': f'No logs in last {time_window_hours} hours'}

        # Calculate metrics
        total_queries = len(recent_logs)
        avg_response_time = sum(log['processing_time'] for log in recent_logs) / total_queries

        source_distribution = {}
        for log in recent_logs:
            source = log.get('source_type', 'unknown')
            source_distribution[source] = source_distribution.get(source, 0) + 1

        rag_logs = [log for log in recent_logs if log.get('source_type') == 'rag']
        avg_relevance = (
            sum(log.get('relevance_score', 0) for log in rag_logs) / len(rag_logs)
            if rag_logs else 0.0
        )

        avg_grounding = sum(
            log.get('grounding_confidence', 0) for log in recent_logs
        ) / total_queries

        error_count = sum(1 for log in recent_logs if log.get('error'))

        return {
            'time_window_hours': time_window_hours,
            'total_queries': total_queries,
            'avg_response_time_seconds': round(avg_response_time, 2),
            'source_distribution': source_distribution,
            'avg_relevance_score': round(avg_relevance, 2),
            'avg_grounding_confidence': round(avg_grounding, 2),
            'error_count': error_count,
            'error_rate': round(error_count / total_queries, 2) if total_queries > 0 else 0
        }
