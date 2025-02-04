from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

class GradeAnswer(BaseModel):
    """Grade for how well the answer addresses the question."""
    binary_score: bool = Field(description="True if the answer addresses the question, False otherwise")
    reasoning: str = Field(description="Explanation for the grade")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

template = """You are grading how well an answer addresses a question.

Question: {question}
Answer: {generation}

Grade whether the answer addresses the question.
If the answer is relevant and addresses the question, return True.
If the answer is off-topic, irrelevant, or doesn't address the question, return False.

Provide your grade as a boolean (True/False) and explain your reasoning."""

prompt = ChatPromptTemplate.from_template(template)

# Create a chain that outputs a structured object
answer_grader = prompt | llm.with_structured_output(GradeAnswer)