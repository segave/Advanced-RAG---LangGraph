import streamlit as st
import atexit

from frontend.components.document_uploader import render_document_uploader
from frontend.components.model_selector import render_model_selector
from frontend.components.rag_chat import render_rag_chat
from frontend.styles.apply_styles import apply_custom_styles
from frontend.ui.factory import UIFactory

# Page configuration MUST be the first Streamlit command
st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

def main():
    """Main application entry point."""
    ui = UIFactory.create()

    # Apply custom styling after page config
    apply_custom_styles()

    # Create two columns: main content, right sidebar
    main_content, right_sidebar = st.columns([2, 1])

    # Main content with RAG chat
    with main_content:
        st.header("RAG Assistant")
        # Add model selector above the chat
        render_model_selector(ui)
        render_rag_chat(ui)

    # Right sidebar for document upload
    with right_sidebar:
        render_document_uploader(ui)

if __name__ == "__main__":
    main()