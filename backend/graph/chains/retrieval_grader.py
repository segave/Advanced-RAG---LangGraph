"""
Module for grading the relevance of retrieved documents to a question.
Evaluates semantic and keyword relevance of documents.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import Dict, Any
import streamlit as st
from ..prompts.templates.retrieval_grader_template import RELEVANCE_TEMPLATE

class DocumentRelevanceGrade(BaseModel):
    """
    Represents the relevance evaluation of a document to a question.
    
    Attributes:
        binary_score: Whether the document is relevant to the question
    """
    binary_score: bool = Field(
        description="True if document is relevant to question, False otherwise"
    )

class RelevanceGrader:
    """
    Evaluates the relevance of retrieved documents to user questions.
    Uses LLM to assess semantic and keyword relevance.
    """
    
    def __init__(self, temperature: float = 0):
        """
        Initialize the grader with specific LLM configuration.
        
        Args:
            temperature: Temperature setting for generation
        """
        self.temperature = temperature
        self._create_chain()

    def _create_chain(self) -> None:
        """Creates the evaluation chain with the grading prompt."""
        self.llm = ChatOpenAI(
            model=st.session_state.get("selected_model", "gpt-4o-mini"),
            temperature=self.temperature
        )
        prompt = ChatPromptTemplate.from_template(RELEVANCE_TEMPLATE)
        self.chain = prompt | self.llm.with_structured_output(DocumentRelevanceGrade)

    def invoke(self, inputs: Dict[str, Any]) -> DocumentRelevanceGrade:
        """
        Evaluate whether a document is relevant to a question.
        
        Args:
            inputs: Dictionary containing:
                - document: Document content to evaluate
                - question: Question to check relevance against
                
        Returns:
            DocumentRelevanceGrade containing the evaluation result
        """
        return self.chain.invoke(inputs)

# Create singleton instance
retrieval_grader = RelevanceGrader()