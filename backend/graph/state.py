from typing import List, TypedDict, Optional, Dict
from langchain.schema import Document
from datetime import datetime

class ChatMessage(TypedDict):
    """Represents a single message in the chat history."""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: str
    documents_used: Optional[List[str]]  # Lista de fuentes usadas

class GraphState(TypedDict, total=False):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
        generation_attempts: counter for generation attempts
        chat_history: history of all interactions
    """

    question: str
    """The question being asked."""
    
    generation: Optional[str]
    """Generated answer."""
    
    web_search: Optional[bool]
    """Whether web search has been performed."""
    
    documents: Optional[List[Document]]
    """Documents retrieved from vector store."""
    
    generation_attempts: Optional[int]
    """Counter for generation attempts."""
    
    chat_history: List[ChatMessage]
    """History of all interactions."""