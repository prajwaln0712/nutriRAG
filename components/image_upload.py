"""Streamlit image upload component."""

import streamlit as st


def render_image_upload():
    """Render the food image upload interface."""
    return st.file_uploader("Upload a food image", type=["png", "jpg", "jpeg"])
