"""
Main Streamlit application.
Entry point for the Agentic RAG system.
"""

import streamlit as st
import logging
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.document_processing.loaders import DocumentLoader
from src.document_processing.chunker import DocumentChunker
from src.document_processing.metadata_extractor import MetadataExtractor
from src.vector_store.chroma_manager import ChromaManager
from src.agents.graph import AgentGraph
from src.ui.upload_section import UploadSection
from src.ui.chat_interface import ChatInterface
from src.ui.settings import SettingsPanel
from src.evaluation.logger import QueryLogger
from src.evaluation.metrics import MetricsCalculator
from config import config

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.logging.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Agentic RAG System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

def initialize_session_state():
    """Initialize session state variables."""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
        st.session_state.chroma_manager = None
        st.session_state.agent_graph = None
        st.session_state.query_logger = None
        st.session_state.show_metrics = False

    ChatInterface.initialize_session()

def initialize_system():
    """Initialize core system components."""
    if not st.session_state.initialized:
        with st.spinner("Initializing system..."):
            try:
                # Initialize ChromaDB
                st.session_state.chroma_manager = ChromaManager()

                # Initialize Agent Graph
                st.session_state.agent_graph = AgentGraph(
                    st.session_state.chroma_manager
                )

                # Initialize Query Logger
                st.session_state.query_logger = QueryLogger()

                st.session_state.initialized = True
                logger.info("System initialized successfully")
            except Exception as e:
                st.error(f"Error initializing system: {str(e)}")
                logger.error(f"Initialization error: {str(e)}")
                st.stop()

def process_uploaded_documents(file_paths):
    """Process and index uploaded documents."""
    if not file_paths:
        return

    with st.spinner("Processing documents..."):
        try:
            # Load documents
            documents = DocumentLoader.load_multiple_documents(file_paths)

            if not documents:
                st.warning("No documents could be loaded")
                return

            # Chunk documents
            chunker = DocumentChunker()
            chunks = chunker.chunk_documents(documents)

            # Enrich metadata
            chunks = MetadataExtractor.enrich_all_metadata(chunks)

            # Add to vector store
            st.session_state.chroma_manager.add_documents(chunks)

            st.success(f"Processed {len(documents)} documents into {len(chunks)} chunks")
            logger.info(f"Indexed {len(chunks)} chunks from {len(documents)} documents")
        except Exception as e:
            st.error(f"Error processing documents: {str(e)}")
            logger.error(f"Document processing error: {str(e)}")

def handle_user_query(query: str):
    """Handle user query through agent graph."""
    try:
        # Execute agent graph
        with st.spinner("Thinking..."):
            result = st.session_state.agent_graph.invoke(query)

        # Check if query was valid
        if not result.get('is_valid', True):
            response = f"Invalid query: {result.get('validation_reason', 'Unknown reason')}"
            sources = []
        else:
            response = result.get('response', 'No response generated')
            sources = result.get('sources', [])

        # Add to chat
        ChatInterface.add_message("assistant", response, sources)

        # Log query
        st.session_state.query_logger.log_query({
            'query': query,
            'response': response,
            'source_type': result.get('source_type', ''),
            'sources': sources,
            'processing_time': result.get('processing_time', 0.0),
            'is_grounded': result.get('is_grounded', True),
            'grounding_confidence': result.get('grounding_confidence', 1.0),
            'relevance_score': result.get('relevance_score', 0.0),
            'error': result.get('error')
        })

        logger.info(f"Query processed: {query[:50]}...")
    except Exception as e:
        st.error(f"Error processing query: {str(e)}")
        logger.error(f"Query processing error: {str(e)}")

def show_metrics_dashboard():
    """Show evaluation metrics dashboard."""
    st.header("Evaluation Metrics")

    try:
        metrics = MetricsCalculator.calculate_metrics(
            './logs/queries.jsonl',
            time_window_hours=24
        )

        if 'error' in metrics:
            st.warning(metrics['error'])
            return

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Queries", metrics['total_queries'])
            st.metric("Avg Response Time", f"{metrics['avg_response_time_seconds']}s")

        with col2:
            st.metric("Avg Relevance Score", metrics['avg_relevance_score'])
            st.metric("Avg Grounding Confidence", metrics['avg_grounding_confidence'])

        with col3:
            st.metric("Error Count", metrics['error_count'])
            st.metric("Error Rate", f"{metrics['error_rate']*100:.1f}%")

        st.subheader("Source Distribution")
        st.bar_chart(metrics['source_distribution'])
    except Exception as e:
        st.error(f"Error loading metrics: {str(e)}")

def main():
    """Main application entry point."""
    st.title("🤖 Agentic RAG System")
    st.caption("Industry-grade Retrieval-Augmented Generation with intelligent routing")

    # Initialize
    initialize_session_state()
    initialize_system()

    # Render settings sidebar
    SettingsPanel.render()

    # Show metrics if requested
    if st.session_state.get('show_metrics', False):
        show_metrics_dashboard()
        if st.button("← Back to Chat"):
            st.session_state.show_metrics = False
            st.rerun()
        return

    # Main layout
    col1, col2 = st.columns([1, 2])

    with col1:
        # Upload section
        uploaded_file_paths = UploadSection.render()

        if uploaded_file_paths:
            if st.button("Process Documents"):
                process_uploaded_documents(uploaded_file_paths)

    with col2:
        # Chat section
        st.header("Chat")

        # Render chat history
        ChatInterface.render_chat_history()

        # Get user input
        user_query = ChatInterface.get_user_input()

        if user_query:
            # Add user message to chat
            ChatInterface.add_message("user", user_query)

            # Process query
            handle_user_query(user_query)

            # Rerun to update chat
            st.rerun()

if __name__ == "__main__":
    main()
