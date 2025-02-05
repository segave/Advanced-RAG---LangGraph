"""
Module for generating responses to questions using LLM.
Provides functionality to generate contextual and chat-aware responses.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from typing import Dict, Any, Optional
import streamlit as st
from ..prompts.templates.generation_template import RESPONSE_TEMPLATE

class ResponseGenerator:
    """
    Generates responses to questions using context and chat history.
    Uses LLM to create informative and contextual responses.
    """
    
    def __init__(self, temperature: float = 0):
        self.temperature = temperature
        self._create_chain()

    def _create_chain(self) -> None:
        """Creates the generation chain with the response prompt."""
        self.llm = ChatOpenAI(
            model=st.session_state.get("selected_model", "gpt-4o-mini"),
            temperature=self.temperature
        )
        print("MODEL")
        print(st.session_state.get("selected_model", "gpt-4o-mini"))
        prompt = ChatPromptTemplate.from_template(RESPONSE_TEMPLATE)
        self.chain = prompt | self.llm | StrOutputParser()

    def invoke(self, inputs: Dict[str, Any]) -> str:
        """
        Generate a response to a question using context and chat history.
        
        Args:
            inputs: Dictionary containing:
                - question: The question to answer
                - context: Relevant context for the answer
                - chat_history: Optional previous conversation
                
        Returns:
            Generated response as string
        """
        return self.chain.invoke(inputs)

# Create singleton instance
generation_chain = ResponseGenerator()