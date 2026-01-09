"""
State management for LangGraph agent workflow.
Defines the state schema that flows through the graph.
"""

from typing import TypedDict, List, Dict, Any, Optional
from langchain.schema import Document

class AgentState(TypedDict):
    """State object that flows through the agent graph."""

    # Input
    query: str

    # Validation
    is_valid: bool
    validation_reason: str

    # Query analysis
    query_intent: Optional[str]  # "document", "knowledge", "search", "unclear"
    confidence: float

    # Routing decision
    selected_tool: Optional[str]  # "rag", "llm", "search"

    # Retrieved context
    retrieved_documents: List[Document]
    relevance_score: float
    is_relevant: bool

    # Search results
    search_results: List[Dict[str, Any]]

    # Response generation
    response: str
    source_type: str  # "rag", "llm", "search"
    sources: List[Dict[str, Any]]  # Source attribution

    # Guardrails
    is_grounded: bool
    grounding_confidence: float

    # Fallback tracking
    attempted_tools: List[str]
    needs_fallback: bool

    # Metadata
    processing_time: float
    error: Optional[str]
    doc_count: int  # Number of documents in vector store
