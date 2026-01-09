"""
LangGraph workflow definition.
Orchestrates the agent nodes into a coherent pipeline.
"""

from langgraph.graph import StateGraph, END
from src.agents.state import AgentState
from src.agents.nodes import AgentNodes
from src.agents.tools import AgentTools
from src.vector_store.chroma_manager import ChromaManager
import logging
import time

logger = logging.getLogger(__name__)

class AgentGraph:
    """Manages the LangGraph agent workflow."""

    def __init__(self, chroma_manager: ChromaManager):
        """
        Initialize the agent graph.

        Args:
            chroma_manager: ChromaDB manager instance
        """
        self.tools = AgentTools(chroma_manager)
        self.nodes = AgentNodes(self.tools)
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("input_validation", self.nodes.input_validation_node)
        workflow.add_node("query_analysis", self.nodes.query_analysis_node)
        workflow.add_node("router", self.nodes.router_node)
        workflow.add_node("rag", self.nodes.rag_node)
        workflow.add_node("llm", self.nodes.llm_node)
        workflow.add_node("search", self.nodes.search_node)
        workflow.add_node("relevance_check", self.nodes.relevance_check_node)
        workflow.add_node("response_synthesis", self.nodes.response_synthesis_node)
        workflow.add_node("hallucination_check", self.nodes.hallucination_check_node)

        # Set entry point
        workflow.set_entry_point("input_validation")

        # Input validation routing
        workflow.add_conditional_edges(
            "input_validation",
            lambda state: "end" if not state['is_valid'] else "query_analysis",
            {
                "end": END,
                "query_analysis": "query_analysis"
            }
        )

        # Query analysis -> Router
        workflow.add_edge("query_analysis", "router")

        # Router to tools
        workflow.add_conditional_edges(
            "router",
            lambda state: state['selected_tool'],
            {
                "rag": "rag",
                "llm": "llm",
                "search": "search"
            }
        )

        # RAG -> Relevance Check
        workflow.add_edge("rag", "relevance_check")

        # Relevance check routing
        workflow.add_conditional_edges(
            "relevance_check",
            lambda state: "synthesis" if state['is_relevant'] else "fallback",
            {
                "synthesis": "response_synthesis",
                "fallback": "llm"  # Fallback to LLM if not relevant
            }
        )

        # Response synthesis -> Hallucination check
        workflow.add_edge("response_synthesis", "hallucination_check")

        # LLM and Search -> Hallucination check
        workflow.add_edge("llm", "hallucination_check")
        workflow.add_edge("search", "hallucination_check")

        # Hallucination check -> END
        workflow.add_edge("hallucination_check", END)

        return workflow.compile()

    def invoke(self, query: str) -> AgentState:
        """
        Execute the agent graph for a query.

        Args:
            query: User query

        Returns:
            Final agent state with response
        """
        start_time = time.time()

        # Get document count from vector store
        try:
            doc_count = self.tools.chroma.get_collection_count()
        except:
            doc_count = 0

        initial_state = AgentState(
            query=query,
            is_valid=True,
            validation_reason='',
            query_intent=None,
            confidence=0.0,
            selected_tool=None,
            retrieved_documents=[],
            relevance_score=0.0,
            is_relevant=False,
            search_results=[],
            response='',
            source_type='',
            sources=[],
            is_grounded=True,
            grounding_confidence=1.0,
            attempted_tools=[],
            needs_fallback=False,
            processing_time=0.0,
            error=None,
            doc_count=doc_count
        )

        try:
            result = self.graph.invoke(initial_state)
            result['processing_time'] = time.time() - start_time

            logger.info(f"Query processed in {result['processing_time']:.2f}s")
            return result
        except Exception as e:
            logger.error(f"Error in graph execution: {str(e)}")
            initial_state['error'] = str(e)
            initial_state['response'] = "An error occurred while processing your query."
            initial_state['processing_time'] = time.time() - start_time
            return initial_state
