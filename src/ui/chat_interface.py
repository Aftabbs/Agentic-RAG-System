"""
Chat interface UI component.
"""

import streamlit as st
from typing import Dict, Any, List

class ChatInterface:
    """Manages chat interface."""

    @staticmethod
    def initialize_session():
        """Initialize session state for chat history."""
        if 'messages' not in st.session_state:
            st.session_state.messages = []

    @staticmethod
    def render_chat_history():
        """Render chat message history."""
        for message in st.session_state.messages:
            with st.chat_message(message['role']):
                st.markdown(message['content'])

                # Show sources if available
                if 'sources' in message and message['sources']:
                    with st.expander("Sources"):
                        ChatInterface.render_sources(message['sources'])

    @staticmethod
    def render_sources(sources: List[Dict[str, Any]]):
        """Render source attribution."""
        for i, source in enumerate(sources, 1):
            source_type = source.get('type', 'unknown')

            if source_type == 'document':
                st.markdown(
                    f"**{i}. Document**: {source.get('file', 'Unknown')} "
                    f"(Page {source.get('page', 'N/A')}) - "
                    f"Score: {source.get('score', 0):.2f}"
                )
                # Show preview directly without nested expander
                preview_content = source.get('content', '')
                if preview_content:
                    st.caption(f"Preview: {preview_content}...")

            elif source_type == 'llm_knowledge':
                st.markdown(
                    f"**{i}. LLM Knowledge**: {source.get('model', 'Unknown')}"
                )
                st.info(source.get('note', ''))

            elif source_type == 'internet_search':
                st.markdown(
                    f"**{i}. Internet Search**: [{source.get('title', 'Unknown')}]({source.get('url', '#')})"
                )
                st.caption(source.get('snippet', ''))

    @staticmethod
    def get_user_input() -> str:
        """Get user input from chat."""
        return st.chat_input("Ask a question...")

    @staticmethod
    def add_message(role: str, content: str, sources: List[Dict[str, Any]] = None):
        """Add message to chat history."""
        message = {
            'role': role,
            'content': content
        }

        if sources:
            message['sources'] = sources

        st.session_state.messages.append(message)
