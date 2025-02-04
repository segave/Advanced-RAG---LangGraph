from dotenv import load_dotenv
from .interfaces import VectorStore 
from typing import Any

load_dotenv()

class RetrieverService:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    def get_retriever(self) -> Any:
        """Gets the retriever from the existing vector database"""
        return self.vector_store.get_retriever() 