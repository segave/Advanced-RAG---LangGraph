"""
Module for grading how well an answer addresses a question.
Provides functionality to evaluate answer relevance and quality.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from ..prompts.templates.answer_grader_template import ANSWER_GRADE_TEMPLATE

class AnswerGrade(BaseModel):
    """
    Represents the evaluation of an answer's relevance to a question.
    
    Attributes:
        binary_score: Boolean indicating if answer addresses the question
        reasoning: Detailed explanation for the grade
        confidence: Optional confidence score for the evaluation
    """
    binary_score: bool = Field(
        description="True if the answer addresses the question, False otherwise"
    )

class AnswerGrader:
    """
    Evaluates how well an answer addresses a given question.
    Uses LLM to perform the evaluation.
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

    def _create_chain(self):
        """Creates the evaluation chain."""

        prompt = ChatPromptTemplate.from_template(ANSWER_GRADE_TEMPLATE)
        self.chain = prompt | self.llm.with_structured_output(AnswerGrade)

    def invoke(self, inputs: dict) -> AnswerGrade:
        """
        Evaluate how well an answer addresses a question.
        
        Args:
            inputs: Dictionary containing 'question' and 'generation'
            
        Returns:
            AnswerGrade containing the evaluation results
        """
        return self.chain.invoke(inputs)

# Create singleton instance
answer_grader = AnswerGrader()