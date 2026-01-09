"""
File upload UI component.
"""

import streamlit as st
from typing import List
from pathlib import Path
import os

class UploadSection:
    """Manages document upload interface."""

    @staticmethod
    def render() -> List[str]:
        """
        Render upload section.

        Returns:
            List of uploaded file paths
        """
        st.header("Document Upload")

        uploaded_files = st.file_uploader(
            "Upload documents (PDF, DOCX, TXT)",
            type=['pdf', 'docx', 'txt'],
            accept_multiple_files=True,
            help="Upload documents to build your knowledge base"
        )

        file_paths = []

        if uploaded_files:
            # Ensure upload directory exists
            upload_dir = Path('./data/uploaded_docs')
            upload_dir.mkdir(parents=True, exist_ok=True)

            for uploaded_file in uploaded_files:
                # Save file
                file_path = upload_dir / uploaded_file.name
                with open(file_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())

                file_paths.append(str(file_path))

            st.success(f"Uploaded {len(uploaded_files)} file(s)")

            # Show uploaded files
            with st.expander("View uploaded files"):
                for file_path in file_paths:
                    st.text(f"{Path(file_path).name}")

        return file_paths
