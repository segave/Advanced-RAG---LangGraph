"""
Module for generating responses to questions using context and chat history.
Handles response generation and chat history management.
"""

from typing import Any, Dict, List
from datetime import datetime
from backend.graph.chains.generation import generation_chain
from backend.graph.state import GraphState, ChatMessage

class ResponseGenerator:
    """
    Handles the generation of responses and management of chat history.
    """

    @staticmethod
    def _format_context(documents: List[Any]) -> str:
        """
        Format documents into a context string.
        
        Args:
            documents: List of documents to format
            
        Returns:
            Formatted context string
        """
        return "\n\n".join(doc.page_content for doc in documents)

    @staticmethod
    def _add_user_message(chat_history: List[ChatMessage], question: str) -> None:
        """
        Add user question to chat history if not already present.
        
        Args:
            chat_history: Current chat history
            question: User's question to add
        """
        if not chat_history or chat_history[-1]["content"] != question:
            chat_history.append(ChatMessage(
                role="user",
                content=question,
                timestamp=datetime.now().isoformat(),
                documents_used=None
            ))

    @staticmethod
    def _add_assistant_message(
        chat_history: List[ChatMessage],
        response: str,
        documents: List[Any]
    ) -> None:
        """
        Add assistant's response to chat history.
        
        Args:
            chat_history: Current chat history
            response: Generated response to add
            documents: Documents used for the response
        """
        chat_history.append(ChatMessage(
            role="assistant",
            content=response,
            timestamp=datetime.now().isoformat(),
            documents_used=[doc.metadata.get("source", "unknown") for doc in documents]
        ))

def generate(state: GraphState) -> Dict[str, Any]:
    """
    Generate a response to the user's question using available context.
    
    Args:
        state: Current graph state containing question and context
        
    Returns:
        Updated state with generated response and chat history
    """
    print("---GENERATE---")
    
    # Get state variables
    generation_attempts = state.get("generation_attempts", 0) + 1
    print(f"---GENERATION ATTEMPT {generation_attempts}/3---")
    
    question = state["question"]
    documents = state.get("documents", [])
    chat_history = state.get("chat_history", [])
    
    # Update chat history and generate response
    ResponseGenerator._add_user_message(chat_history, question)
    
    context = ResponseGenerator._format_context(documents)
    generation = generation_chain.invoke({
        "question": question,
        "context": context,
        "chat_history": chat_history
    })
    
    ResponseGenerator._add_assistant_message(chat_history, generation, documents)
    
    return {
        "generation": generation,
        "documents": documents,
        "question": question,
        "generation_attempts": generation_attempts,
        "chat_history": chat_history
    }