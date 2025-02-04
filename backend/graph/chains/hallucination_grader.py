"""
Module for grading whether generated responses are grounded in provided facts.
Evaluates potential hallucinations in LLM responses.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from langchain.schema import Document

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
    
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0):
        """
        Initialize the grader with specific LLM configuration.
        
        Args:
            model_name: Name of the LLM model to use
            temperature: Temperature setting for generation
        """
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self._create_chain()

    def _create_chain(self) -> None:
        """Creates the evaluation chain with the grading prompt."""
        template = """You are an expert fact-checker evaluating if a response is grounded in provided facts.

        Facts:
        {documents}

        Response to evaluate:
        {generation}

        Instructions:
        1. Compare the response against the provided facts
        2. Check if all claims are supported by the facts
        3. Ignore stylistic differences or rephrasing
        4. Focus on factual accuracy

        Return True if the response is fully grounded in facts, False if it contains unsupported claims."""

        prompt = ChatPromptTemplate.from_template(template)
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