from enum import Enum
from typing import List
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI
from backend.document_processor.service import document_service

class DataSource(str, Enum):
    VECTORSTORE = "vectorstore"
    WEBSEARCH = "websearch"

class RouteQuery(BaseModel):
    """Decision on where to route the query."""
    datasource: DataSource = Field(
        description="Where to route the query: 'vectorstore' for local documents or 'websearch' for internet search"
    )
    reasoning: str = Field(
        description="Explanation of why this source was chosen"
    )

def get_available_topics() -> List[str]:
    """Get a summary of topics available in the vector store"""
    try:
        # Get vector store from service
        vector_store = document_service.get_vector_store()
        # Get a sample of documents to understand available content
        retriever = vector_store.get_retriever()
        sample_docs = retriever.invoke("what topics are available")
        print("---SAMPLE DOCS---")
        print(sample_docs)
        topics = "\n".join([doc.page_content[:200] + "..." for doc in sample_docs])
        return topics
    except Exception as e:
        print(f"Error getting topics: {str(e)}")
        return "No local documents available yet."

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
structured_router = llm.with_structured_output(RouteQuery)

system = """You are a query router that decides whether to search in local documents or the web.
Available topics in local documents:
{available_topics}

Rules for routing:
1. If the question is about topics covered in local documents, use 'vectorstore'
2. If the question is about current events or topics not in local documents, use 'websearch'
3. If unsure, use 'websearch'

Give your routing decision and explain why."""

router_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Question: {question}")
    ]
)

def create_router() -> RunnableSequence:
    """Creates a router with current available topics"""
    topics = get_available_topics()
    return router_prompt.partial(available_topics=topics) | structured_router

# Create initial router
question_router = create_router()

def refresh_router():
    vector_store = document_service.get_vector_store()
    """Refresh the router with current available topics"""
    global question_router
    question_router = create_router()