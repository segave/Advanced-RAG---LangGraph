from dotenv import load_dotenv
import os
from .ingestion import (
    RecursiveTextSplitter,
    ChromaVectorStore,
    DocumentIngester
)

load_dotenv()

class DocumentService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def __init__(self):
        self._initialize()

    def _initialize(self):
        """Initialize document service components"""
        self.text_splitter = RecursiveTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        self.vector_store = ChromaVectorStore(
            collection_name=os.getenv("INDEX_NAME", "document-store"),
            persist_directory=os.getenv("VECTOR_STORE_PATH", "./.chroma")
        )
        self.ingester = DocumentIngester(self.text_splitter, self.vector_store)

    def get_vector_store(self):
        return self.vector_store

    def get_ingester(self):
        return self.ingester

# Singleton instance
document_service = DocumentService() 