"""
Module for classifying questions to determine the appropriate entry point.
Decides whether a question needs information search or can be answered directly.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List, Dict, Any

class EntryClassification(BaseModel):
    """
    Represents the classification decision for a question.
    
    Attributes:
        needs_search: Whether the question requires information search
    """
    needs_search: bool = Field(
        description="True if question needs information search, False if it can be answered directly"
    )

class EntryClassifier:
    """
    Classifies questions to determine the appropriate processing path.
    Uses LLM to make classification decisions.
    """
    
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0):
        """
        Initialize the classifier with specific LLM configuration.
        
        Args:
            model_name: Name of the LLM model to use
            temperature: Temperature setting for generation
        """
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self._create_chain()

    def _create_chain(self) -> None:
        """Creates the classification chain with the evaluation prompt."""
        template = """You are deciding whether a question needs information search.

        Question: {question}
        Chat History: {chat_history}

        Classify if this question:
        1. Needs information search (True):
        - Requires specific facts or data
        - References external information
        - Needs verification from sources
        
        2. Can be answered directly (False):
        - Is a clarification or follow-up
        - Can be answered from chat history
        - Is a general knowledge question
        - Is about the conversation itself

        Return True if search is needed, False if it can be answered directly."""

        prompt = ChatPromptTemplate.from_template(template)
        self.chain = prompt | self.llm.with_structured_output(EntryClassification)

    def _format_chat_history(self, chat_history: List[Dict[str, Any]]) -> str:
        """
        Format chat history for the prompt.
        
        Args:
            chat_history: List of chat messages
            
        Returns:
            Formatted chat history string
        """
        if not chat_history:
            return "No previous conversation."
        
        formatted = []
        for msg in chat_history:
            role = "User" if msg["role"] == "user" else "Assistant"
            formatted.append(f"{role}: {msg['content']}")
        return "\n".join(formatted)

    def invoke(self, inputs: Dict[str, Any]) -> EntryClassification:
        """
        Classify a question to determine if it needs information search.
        
        Args:
            inputs: Dictionary containing 'question' and optional 'chat_history'
            
        Returns:
            EntryClassification containing the decision
        """
        formatted_history = self._format_chat_history(inputs.get("chat_history", []))
        return self.chain.invoke({
            "question": inputs["question"],
            "chat_history": formatted_history
        })

# Create singleton instance
entry_classifier = EntryClassifier()