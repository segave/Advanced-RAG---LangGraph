"""
Module for document processing service management.
Provides a centralized service for document processing operations.
"""

from typing import Optional
from .ingestion import (
    DocumentIngester,
    RecursiveTextSplitter,
    ChromaVectorStore
)
from .retriever import RetrieverService

class DocumentService:
    """
    Service for managing document processing operations.
    Handles ingestion, retrieval, and vector store management.
    """

    def __init__(
        self,
        collection_name: str = "rag-chroma",
        persist_directory: str = "./.chroma",
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        search_type: str = "similarity",
        k: int = 4,
        score_threshold: float = 0.5
    ):
        """
        Initialize document service with configuration.
        
        Args:
            collection_name: Name for the vector store collection
            persist_directory: Directory for vector store persistence
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between text chunks
            search_type: Type of search for retrieval
            k: Number of documents to retrieve
            score_threshold: Minimum similarity score for retrieval
        """
        self._vector_store: Optional[ChromaVectorStore] = None
        self._ingester: Optional[DocumentIngester] = None
        self._retriever: Optional[RetrieverService] = None
        
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.search_type = search_type
        self.k = k
        self.score_threshold = score_threshold

    def _initialize_vector_store(self) -> ChromaVectorStore:
        """
        Initialize vector store with current configuration.
        
        Returns:
            Configured vector store instance
        """
        if not self._vector_store:
            self._vector_store = ChromaVectorStore(
                collection_name=self.collection_name,
                persist_directory=self.persist_directory
            )
        return self._vector_store

    def _initialize_ingester(self) -> DocumentIngester:
        """
        Initialize document ingester with current configuration.
        
        Returns:
            Configured document ingester instance
        """
        if not self._ingester:
            text_splitter = RecursiveTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            self._ingester = DocumentIngester(
                text_splitter=text_splitter,
                vector_store=self.get_vector_store()
            )
        return self._ingester

    def _initialize_retriever(self) -> RetrieverService:
        """
        Initialize retriever service with current configuration.
        
        Returns:
            Configured retriever service instance
        """
        if not self._retriever:
            self._retriever = RetrieverService(
                vector_store=self.get_vector_store(),
                search_type=self.search_type,
                k=self.k,
                score_threshold=self.score_threshold
            )
        return self._retriever

    def get_vector_store(self) -> ChromaVectorStore:
        """
        Get vector store instance.
        
        Returns:
            Configured vector store
        """
        return self._initialize_vector_store()

    def get_ingester(self) -> DocumentIngester:
        """
        Get document ingester instance.
        
        Returns:
            Configured document ingester
        """
        return self._initialize_ingester()

    def get_retriever(self) -> RetrieverService:
        """
        Get retriever service instance.
        
        Returns:
            Configured retriever service
        """
        return self._initialize_retriever()

    def update_configuration(
        self,
        collection_name: Optional[str] = None,
        persist_directory: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        search_type: Optional[str] = None,
        k: Optional[int] = None,
        score_threshold: Optional[float] = None
    ) -> None:
        """
        Update service configuration parameters.
        
        Args:
            collection_name: New collection name
            persist_directory: New persistence directory
            chunk_size: New chunk size
            chunk_overlap: New chunk overlap
            search_type: New search type
            k: New number of documents
            score_threshold: New score threshold
        """
        if collection_name:
            self.collection_name = collection_name
        if persist_directory:
            self.persist_directory = persist_directory
        if chunk_size:
            self.chunk_size = chunk_size
        if chunk_overlap:
            self.chunk_overlap = chunk_overlap
        if search_type:
            self.search_type = search_type
        if k:
            self.k = k
        if score_threshold:
            self.score_threshold = score_threshold

        # Reset services to reinitialize with new configuration
        self._vector_store = None
        self._ingester = None
        self._retriever = None

# Create singleton instance
document_service = DocumentService() 