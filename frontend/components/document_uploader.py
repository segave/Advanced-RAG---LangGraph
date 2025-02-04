import streamlit as st
import os
import tempfile
from typing import Optional, Union

from backend.document_processor import (
    DocumentIngester,
    RecursiveTextSplitter,
    ChromaVectorStore,
    FileLoader,
    PDFLoader,
    DirectoryDocumentLoader,
    DocxLoader
)
from frontend.ui.factory import UIFactory
from frontend.ui.interfaces.base import UploadInterface, MessagingInterface
from frontend.ui.interfaces.state import StateInterface
from frontend.ui.interfaces.markup import MarkupInterface
from backend.graph.chains.router import refresh_router
from backend.document_processor.service import document_service
#from backend.document_processor.config import vector_store

def get_document_loader(file_paths: list[str]) -> Union[FileLoader, PDFLoader]:
    """Determines the appropriate loader based on file extensions"""
    # Group files by extension
    pdf_files = [f for f in file_paths if f.lower().endswith('.pdf')]
    text_files = [f for f in file_paths if f.lower().endswith(('.txt'))]
    docx_files = [f for f in file_paths if f.lower().endswith('.docx')]

    if pdf_files and not text_files and not docx_files:
        return PDFLoader(pdf_files)
    elif text_files and not pdf_files and not docx_files:
        return FileLoader(text_files)
    elif docx_files and not pdf_files and not text_files:
        return DocxLoader(docx_files)
    else:
        # If there's a mix of file types, use DirectoryLoader
        directory = os.path.dirname(file_paths[0])
        return DirectoryDocumentLoader(directory)

def render_document_uploader(
    ui: Optional[Union[UploadInterface, MessagingInterface]] = None,
    state: Optional[StateInterface] = None,
    markup: Optional[MarkupInterface] = None
):
    """Render the document uploader section in the right sidebar."""
    ui = ui or UIFactory.create_ui()
    state = state or UIFactory.create_state()
    markup = markup or UIFactory.create_markup()
    
    markup.markdown("### Document Upload")
    
    # File uploader for documents
    uploaded_files = ui.file_uploader(
        "Upload documents",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        help="You can upload one or multiple PDF, DOCX, or TXT files"
    )
    
    # Get service instances
    vector_store = document_service.get_vector_store()
    ingester = document_service.get_ingester()
    
    # Create a temporary directory for storing documents if it doesn't exist
    temp_dir = "temp_docs"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    
    # Document processor configuration
    text_splitter = RecursiveTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    
    # Button to process and ingest the documents
    if ui.button("Ingest Documents"):
        if uploaded_files:
            try:
                # Save uploaded files to temporary directory
                saved_files = []
                for uploaded_file in uploaded_files:
                    # Create temporary file path
                    temp_file_path = os.path.join(temp_dir, uploaded_file.name)
                    saved_files.append(temp_file_path)
                    
                    # Save uploaded file
                    with open(temp_file_path, "wb") as f:
                        f.write(uploaded_file.getvalue())
                
                # Process and ingest the documents
                with ui.spinner("Processing documents..."):
                    # Get appropriate loader based on file types
                    document_loader = get_document_loader(saved_files)
                    # Process the documents
                    ingester.process_documents(document_loader)
                    
                # Refresh router with new documents
                refresh_router()
                
                ui.success(f"{len(saved_files)} document(s) successfully ingested!")
                
                # Clean up
                for file_path in saved_files:
                    if os.path.exists(file_path):
                        os.remove(file_path)
            
            except Exception as e:
                ui.error(f"Error processing documents: {str(e)}")
                # Clean up on error
                for file_path in saved_files:
                    if os.path.exists(file_path):
                        os.remove(file_path)
        else:
            ui.warning("Please upload documents before ingesting.")

    # Add some information about supported formats
    markup.markdown("""
    #### Supported Formats
    - PDF documents (.pdf)
    - Word documents (.docx)
    - Text files (.txt)
    
    #### Instructions
    1. Click 'Browse files' to select documents
    2. You can select multiple files at once
    3. Click 'Ingest Documents' to process and add them to the database
    
    The documents will be processed and stored in a vector database for efficient retrieval.
    """)

    # Add separator before cleanup button
    markup.markdown("---")
    
    # Add cleanup button at the end
    if ui.button("🗑️ Clear Document Database"):
        with ui.spinner("Cleaning up document database..."):
            vector_store.cleanup()
        ui.success("Document database cleared successfully!")
        ui.rerun()  