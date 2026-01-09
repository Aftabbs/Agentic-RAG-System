"""
Individual nodes for the LangGraph agent workflow.
Each node performs a specific step in the agentic pipeline.
"""

from typing import Dict, Any
from src.agents.state import AgentState
from src.agents.tools import AgentTools
from src.guardrails.input_validator import InputValidator
from src.guardrails.relevance_scorer import RelevanceScorer
from src.guardrails.hallucination_detector import HallucinationDetector
from src.llm.groq_client import groq_client
import logging
import time

logger = logging.getLogger(__name__)

class AgentNodes:
    """All nodes for the LangGraph workflow."""

    def __init__(self, tools: AgentTools):
        """
        Initialize nodes with tools and guardrails.

        Args:
            tools: AgentTools instance
        """
        self.tools = tools
        self.validator = InputValidator()
        self.relevance_scorer = RelevanceScorer()
        self.hallucination_detector = HallucinationDetector()
        self.llm = groq_client.get_llm()

    def input_validation_node(self, state: AgentState) -> AgentState:
        """Validate user input for safety."""
        logger.info("Node: Input Validation")

        validation_result = self.validator.validate_query(state['query'])

        state['is_valid'] = validation_result['valid']
        state['validation_reason'] = validation_result['reason']

        if not validation_result['valid']:
            logger.warning(f"Query rejected: {validation_result['reason']}")

        return state

    def query_analysis_node(self, state: AgentState) -> AgentState:
        """Analyze query to determine intent."""
        logger.info("Node: Query Analysis")

        query = state['query']

        prompt = f"""Analyze the following query and determine the best source to answer it.

Query: {query}

Classify into ONE of these categories:
- "document": Query is asking about specific people, names, documents, files, uploaded content, or specific details that likely require document lookup
- "knowledge": Query is asking general knowledge questions (definitions, concepts, how-to) that can be answered from training data
- "search": Query is asking about current events, today's news, recent information, or real-time data

Important: If the query mentions specific names, people, or asks "tell me about [specific thing]", classify as "document" since this likely requires uploaded files.

Also rate your confidence from 0.0 to 1.0.

Respond in this exact format:
Category: [document/knowledge/search]
Confidence: [0.0-1.0]"""

        try:
            response = self.llm.invoke(prompt)
            content = response.content

            # Parse response
            category_line = [l for l in content.split('\n') if 'Category:' in l][0]
            confidence_line = [l for l in content.split('\n') if 'Confidence:' in l][0]

            intent = category_line.split(':')[1].strip().lower()
            confidence = float(confidence_line.split(':')[1].strip())

            state['query_intent'] = intent
            state['confidence'] = confidence

            logger.info(f"Query intent: {intent} (confidence: {confidence:.2f})")
        except Exception as e:
            logger.error(f"Error in query analysis: {str(e)}")
            # Default to document search
            state['query_intent'] = 'document'
            state['confidence'] = 0.5

        return state

    def router_node(self, state: AgentState) -> AgentState:
        """Route query to appropriate tool based on intent."""
        logger.info("Node: Router")

        intent = state.get('query_intent', 'document')
        confidence = state.get('confidence', 0.5)

        # Check if documents exist in vector store
        try:
            from src.vector_store.chroma_manager import ChromaManager
            doc_count = state.get('doc_count', 0)
        except:
            doc_count = 0

        # If documents exist, ALWAYS try RAG first (unless explicitly a search query)
        if doc_count > 0 and intent != 'search':
            state['selected_tool'] = 'rag'
            logger.info(f"Documents exist ({doc_count}), routing to RAG first")
        # Otherwise, use intent-based routing
        elif confidence >= 0.7:
            if intent == 'document':
                state['selected_tool'] = 'rag'
            elif intent == 'knowledge':
                state['selected_tool'] = 'llm'
            elif intent == 'search':
                state['selected_tool'] = 'search'
        else:
            # Low confidence: default to RAG, will fallback if needed
            state['selected_tool'] = 'rag'

        logger.info(f"Routed to: {state['selected_tool']}")

        if 'attempted_tools' not in state:
            state['attempted_tools'] = []
        state['attempted_tools'].append(state['selected_tool'])

        return state

    def rag_node(self, state: AgentState) -> AgentState:
        """Execute RAG tool."""
        logger.info("Node: RAG Tool")

        result = self.tools.rag_tool(state['query'])

        state['retrieved_documents'] = result['documents']
        state['sources'] = result['sources']

        if not result['success']:
            state['needs_fallback'] = True
            logger.warning("RAG tool failed, needs fallback")

        return state

    def llm_node(self, state: AgentState) -> AgentState:
        """Execute LLM knowledge tool."""
        logger.info("Node: LLM Tool")

        result = self.tools.llm_tool(state['query'])

        if result['success']:
            state['response'] = result['response']
            state['sources'] = result['sources']
            state['source_type'] = 'llm'
        else:
            state['needs_fallback'] = True
            logger.warning("LLM tool failed")

        return state

    def search_node(self, state: AgentState) -> AgentState:
        """Execute internet search tool."""
        logger.info("Node: Search Tool")

        result = self.tools.search_tool(state['query'])

        if result['success']:
            state['search_results'] = result['results']
            state['sources'] = result['sources']

            # Generate response from search results
            prompt = f"""Based on the following search results, answer the user's question.

Question: {state['query']}

Search Results:
{result['context']}

Provide a concise, accurate answer:"""

            try:
                response = self.llm.invoke(prompt)
                state['response'] = response.content
                state['source_type'] = 'search'
            except Exception as e:
                logger.error(f"Error generating search response: {str(e)}")
                state['needs_fallback'] = True
        else:
            state['needs_fallback'] = True
            logger.warning("Search tool failed")

        return state

    def relevance_check_node(self, state: AgentState) -> AgentState:
        """Check if retrieved documents are relevant."""
        logger.info("Node: Relevance Check")

        documents = state.get('retrieved_documents', [])

        if not documents:
            state['is_relevant'] = False
            state['relevance_score'] = 0.0
            state['needs_fallback'] = True
            return state

        result = self.relevance_scorer.score_relevance(
            state['query'],
            documents
        )

        state['is_relevant'] = result['is_relevant']
        state['relevance_score'] = result['score']

        if not result['is_relevant']:
            state['needs_fallback'] = True
            logger.warning(f"Documents not relevant: {result['reason']}")

        return state

    def response_synthesis_node(self, state: AgentState) -> AgentState:
        """Synthesize response from retrieved documents."""
        logger.info("Node: Response Synthesis")

        documents = state.get('retrieved_documents', [])

        if not documents:
            state['response'] = "I couldn't find relevant information in the uploaded documents."
            state['source_type'] = 'none'
            return state

        # Combine document content
        context = "\n\n".join([
            f"[{doc.metadata.get('source_file', 'Unknown')} - Page {doc.metadata.get('page', 'N/A')}]\n{doc.page_content}"
            for doc in documents[:3]
        ])

        prompt = f"""Based on the following document excerpts, answer the user's question. Be specific and cite which document you're referencing.

Question: {state['query']}

Documents:
{context}

Provide a clear, accurate answer based ONLY on the information in these documents:"""

        try:
            response = self.llm.invoke(prompt)
            state['response'] = response.content
            state['source_type'] = 'rag'
        except Exception as e:
            logger.error(f"Error in response synthesis: {str(e)}")
            state['response'] = "Error generating response."
            state['error'] = str(e)

        return state

    def hallucination_check_node(self, state: AgentState) -> AgentState:
        """Check if response is grounded in context."""
        logger.info("Node: Hallucination Check")

        # Get context based on source type
        context = None
        if state['source_type'] == 'rag':
            documents = state.get('retrieved_documents', [])
            context = "\n\n".join([doc.page_content for doc in documents[:3]])

        result = self.hallucination_detector.check_grounding(
            state['query'],
            state.get('response', ''),
            context,
            state['source_type']
        )

        state['is_grounded'] = result['is_grounded']
        state['grounding_confidence'] = result['confidence']

        if not result['is_grounded']:
            logger.warning(f"Response may contain hallucinations: {result['reason']}")
            # Add warning to response
            state['response'] += "\n\nNote: This response may contain unverified information."

        return state
