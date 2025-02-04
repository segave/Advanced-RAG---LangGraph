"""
Module for grading the relevance of retrieved documents to a question.
Evaluates semantic and keyword relevance of documents.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import Dict, Any

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
        template = """You are an expert evaluating document relevance to questions.

        Document to evaluate:
        {document}

        Question:
        {question}

        Instructions:
        1. Check for keyword matches
        2. Assess semantic relevance
        3. Consider related concepts
        4. Look for contextual connections

        A document is relevant if it:
        - Contains keywords from the question
        - Has semantically related content
        - Provides context for the answer
        - Contains information needed to answer

        Return True if the document is relevant, False otherwise."""

        prompt = ChatPromptTemplate.from_template(template)
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