"""
Document processing module initialization.
Exports main components for document handling.
"""

from .ingestion import (
    WebLoader,
    PDFLoader,
    FileLoader,
    DirectoryDocumentLoader,
    DocxLoader,
    RecursiveTextSplitter,
    ChromaVectorStore,
    DocumentIngester,
    get_document_loader,
    CombinedLoader
)
from .retriever import RetrieverService
from .interfaces import DocumentLoader, TextSplitter, VectorStore

__all__ = [
    'WebLoader',
    'PDFLoader',
    'FileLoader',
    'DirectoryDocumentLoader',
    'RecursiveTextSplitter',
    'ChromaVectorStore',
    'DocumentIngester',
    'RetrieverService',
    'DocumentLoader',
    'TextSplitter',
    'VectorStore',
    'DocxLoader',
    'get_document_loader',
    'CombinedLoader'
] 