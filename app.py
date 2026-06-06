"""NutriRAG Streamlit application."""

import streamlit as st

st.set_page_config(page_title="NutriRAG", page_icon="🥗")

st.title("NutriRAG")
st.subheader("Your AI Nutrition Assistant")

chat_tab, identify_tab = st.tabs(["Chat", "Identify Food"])

with chat_tab:
    st.info("Chat coming soon — ask questions about nutrition here.")

with identify_tab:
    st.info("Food identification coming soon — upload a food image here.")
