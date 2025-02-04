"""
Module for retrieving relevant documents from the vector store.
Handles document retrieval and search operations.
"""

from typing import Any, Dict, List
from langchain.schema import Document
from backend.document_processor.service import document_service
from backend.graph.state import GraphState

class DocumentRetriever:
    """
    Handles document retrieval operations using vector store.
    """

    def __init__(self, k: int = 4):
        """
        Initialize the retriever with configuration.
        
        Args:
            k: Number of documents to retrieve
        """
        self.k = k
        self._setup_retriever()

    def _setup_retriever(self) -> None:
        """Set up the vector store retriever."""
        self.vector_store = document_service.get_vector_store()
        self.retriever = self.vector_store.get_retriever()
        self.retriever.search_kwargs = {"k": self.k}

    def search_documents(self, query: str) -> List[Document]:
        """
        Search for relevant documents using the query.
        
        Args:
            query: Search query string
            
        Returns:
            List of relevant documents
        """
        print(f"---SEARCHING FOR DOCUMENTS WITH QUERY: {query}---")
        documents = self.retriever.invoke(query)
        print(f"---FOUND {len(documents)} DOCUMENTS---")
        return documents

def retrieve(state: GraphState) -> Dict[str, Any]:
    """
    Retrieve relevant documents for a given question.
    
    Args:
        state: Current graph state containing the question
        
    Returns:
        Updated state with retrieved documents
    """
    print("---RETRIEVE DOCUMENTS---")
    
    question = state["question"]
    retriever = DocumentRetriever()
    documents = retriever.search_documents(question)
    
    return {
        "documents": documents,
        "question": question
    }