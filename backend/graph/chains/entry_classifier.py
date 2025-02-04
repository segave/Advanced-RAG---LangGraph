from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

class EntryDecision(BaseModel):
    """Decision about whether to search for information or generate directly."""
    needs_search: bool = Field(description="True if the question needs additional information, False if it can be answered directly")
    reasoning: str = Field(description="Explanation for the decision")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

template = """You are deciding whether a question needs additional information to be answered accurately.

Question: {question}
Chat History: {chat_history}

Decide if this question:
1. Needs information search (True): 
   - Requires specific facts, data, or context
   - References documents or external information
   - Needs verification from sources
   
2. Can be answered directly (False):
   - Is a simple clarification or follow-up
   - Can be answered from the chat history
   - Is a general knowledge question
   - Is about the conversation itself

Return True if search is needed, False if it can be answered directly."""

prompt = ChatPromptTemplate.from_template(template)

entry_classifier = prompt | llm.with_structured_output(EntryDecision)