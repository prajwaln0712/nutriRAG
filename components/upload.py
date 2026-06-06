"""Streamlit document upload component."""

import streamlit as st


def render_upload():
    """Render the document upload interface."""
    st.file_uploader("Upload a document", type=["pdf", "txt", "csv"])
