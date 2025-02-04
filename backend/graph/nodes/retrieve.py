from typing import Any, Dict

from backend.graph.state import GraphState
from backend.document_processor.service import document_service


def retrieve(state: GraphState) -> Dict[str, Any]:
    print("---RETRIEVE---")
    question = state["question"]

    # Get retriever from service
    retriever = document_service.get_vector_store().get_retriever()
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}