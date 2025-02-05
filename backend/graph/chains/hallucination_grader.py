"""
Module for grading whether generated responses are grounded in provided facts.
Evaluates potential hallucinations in LLM responses.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from langchain.schema import Document
import streamlit as st
from ..prompts.templates.hallucination_grader_template import HALLUCINATION_TEMPLATE

class HallucinationGrade(BaseModel):
    """
    Represents the evaluation of factual grounding in a generated response.
    
    Attributes:
        binary_score: Whether the response is grounded in provided facts
    """
    binary_score: bool = Field(
        description="True if response is grounded in facts, False if hallucinating"
    )

class HallucinationGrader:
    """
    Evaluates whether generated responses are grounded in provided documents.
    Uses LLM to detect potential hallucinations.
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
        prompt = ChatPromptTemplate.from_template(HALLUCINATION_TEMPLATE)
        self.chain = prompt | self.llm.with_structured_output(HallucinationGrade)

    def _format_documents(self, documents: List[Document]) -> str:
        """
        Format documents into a string for the prompt.
        
        Args:
            documents: List of documents containing facts
            
        Returns:
            Formatted string of facts
        """
        return "\n\n".join(doc.page_content for doc in documents)

    def invoke(self, inputs: Dict[str, Any]) -> HallucinationGrade:
        """
        Evaluate whether a response is grounded in provided documents.
        
        Args:
            inputs: Dictionary containing:
                - documents: List of reference documents
                - generation: Generated response to evaluate
                
        Returns:
            HallucinationGrade containing the evaluation result
        """
        formatted_docs = self._format_documents(inputs["documents"])
        return self.chain.invoke({
            "documents": formatted_docs,
            "generation": inputs["generation"]
        })

# Create singleton instance
hallucination_grader = HallucinationGrader()