"""
Module for classifying questions to determine the appropriate entry point.
Decides whether a question needs information search or can be answered directly.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import streamlit as st
from ..prompts.templates.entry_classifier_template import CLASSIFICATION_TEMPLATE

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
    
    def __init__(self, temperature: float = 0):
        """
        Initialize the classifier with specific LLM configuration.
        
        Args:
            temperature: Temperature setting for generation
        """
        self.temperature = temperature
        self._create_chain()

    def _create_chain(self) -> None:
        """Creates the classification chain with the evaluation prompt."""
        self.llm = ChatOpenAI(
            model=st.session_state.get("selected_model", "gpt-4o-mini"),
            temperature=self.temperature
        )
        prompt = ChatPromptTemplate.from_template(CLASSIFICATION_TEMPLATE)
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