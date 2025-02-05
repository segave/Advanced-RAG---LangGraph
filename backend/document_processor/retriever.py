"""
Module for document retrieval functionality.
Provides services for retrieving relevant documents from vector stores.
"""

from typing import Any, List, Optional
from langchain.schema import Document
from dotenv import load_dotenv
from .interfaces import VectorStore

load_dotenv()

class RetrieverService:
    """
    Service for retrieving documents from vector stores.
    Handles document retrieval with configurable parameters.
    """

    def __init__(
        self, 
        vector_store: VectorStore,
        search_type: str = "similarity",
        k: int = 4,
        score_threshold: Optional[float] = 0.5
    ):
        """
        Initialize retriever service with configuration.
        
        Args:
            vector_store: Vector store to retrieve documents from
            search_type: Type of search to perform ('similarity' or 'mmr')
            k: Number of documents to retrieve
            score_threshold: Minimum similarity score threshold
        """
        self.vector_store = vector_store
        self.search_type = search_type
        self.k = k
        self.score_threshold = score_threshold
        self._retriever = None

    def _initialize_retriever(self) -> None:
        """Initialize the retriever with current configuration."""
        base_retriever = self.vector_store.get_retriever()
        base_retriever.search_type = self.search_type
        base_retriever.search_kwargs = {
            "k": self.k,
            "score_threshold": self.score_threshold
        }
        self._retriever = base_retriever

    def get_retriever(self) -> Any:
        """
        Get configured retriever instance.
        
        Returns:
            Configured retriever for document search
        """
        if not self._retriever:
            self._initialize_retriever()
        return self._retriever

    def retrieve_documents(self, query: str) -> List[Document]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Search query string
            
        Returns:
            List of relevant documents
        """
        if not self._retriever:
            self._initialize_retriever()
        return self._retriever.get_relevant_documents(query)

    def update_search_parameters(
        self,
        search_type: Optional[str] = None,
        k: Optional[int] = None,
        score_threshold: Optional[float] = None
    ) -> None:
        """
        Update search parameters for retriever.
        
        Args:
            search_type: New search type ('similarity' or 'mmr')
            k: New number of documents to retrieve
            score_threshold: New similarity score threshold
        """
        if search_type is not None:
            self.search_type = search_type
        if k is not None:
            self.k = k
        if score_threshold is not None:
            self.score_threshold = score_threshold
        
        # Reinitialize retriever with new parameters
        self._initialize_retriever()