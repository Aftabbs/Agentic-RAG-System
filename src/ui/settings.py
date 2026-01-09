"""
Settings panel UI component.
"""

import streamlit as st
from config import config

class SettingsPanel:
    """Manages settings interface."""

    @staticmethod
    def render():
        """Render settings sidebar."""
        with st.sidebar:
            st.header("Settings")

            # API Status
            st.subheader("API Status")

            groq_status = "✓" if config.api.groq_api_key else "✗"
            serper_status = "✓" if config.api.serper_api_key else "✗"

            st.text(f"{groq_status} Groq API")
            st.text(f"{serper_status} Serper API")

            st.divider()

            # Configuration
            st.subheader("Configuration")

            st.text(f"Model: {config.api.groq_model}")
            st.text(f"Chunk Size: {config.chunking.chunk_size}")
            st.text(f"Top-K Results: {config.retrieval.top_k}")

            st.divider()

            # Vector Store Info
            st.subheader("Vector Store")

            if 'chroma_manager' in st.session_state:
                count = st.session_state.chroma_manager.get_collection_count()
                st.metric("Documents in DB", count)

            st.divider()

            # Actions
            st.subheader("Actions")

            if st.button("Clear Chat History"):
                st.session_state.messages = []
                st.rerun()

            if st.button("View Metrics"):
                st.session_state.show_metrics = True
                st.rerun()
