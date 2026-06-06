"""Streamlit sidebar component."""

import streamlit as st


def render_sidebar():
    """Render the application sidebar."""
    with st.sidebar:
        st.header("NutriRAG")
        st.caption("Your AI Nutrition Assistant")
