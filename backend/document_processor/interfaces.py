from abc import ABC, abstractmethod
from typing import List, Any

class DocumentLoader(ABC):
    """Interface for document loading operations"""
    @abstractmethod
    def load(self) -> List[Any]:
        """Loads documents and returns a list of documents"""
        pass

class TextSplitter(ABC):
    """Interface for document splitting operations"""
    @abstractmethod
    def split_documents(self, documents: List[Any]) -> List[Any]:
        """Splits documents into smaller chunks"""
        pass

class VectorStore(ABC):
    """Interface for vector storage operations"""
    @abstractmethod
    def store_documents(self, documents: List[Any]) -> Any:
        """Stores documents in the vector database"""
        pass
    
    @abstractmethod
    def get_retriever(self) -> Any:
        """Gets the retriever for searching documents"""
        pass 