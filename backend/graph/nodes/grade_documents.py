"""
Module for grading the relevance of retrieved documents to a question.
"""

from typing import Any, Dict, List
from backend.graph.chains.retrieval_grader import retrieval_grader
from backend.graph.state import GraphState

def grade_documents(state: GraphState) -> Dict[str, Any]:
    """
    Grade the relevance of retrieved documents to the question.
    
    Args:
        state: Current graph state containing documents and question
        
    Returns:
        Updated state with filtered relevant documents
    """
    print("---GRADE DOCUMENTS---")
    
    question = state["question"]
    documents = state["documents"]
    web_search = state.get("web_search", False)
    
    # Filter documents based on relevance
    filtered_docs = []
    for doc in documents:
        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        grade = retrieval_grader.invoke({
            "question": question,
            "document": doc.page_content
        })
        
        if grade.binary_score:
            print("---DOCUMENT IS RELEVANT---")
            filtered_docs.append(doc)
        else:
            print("---DOCUMENT IS NOT RELEVANT---")
    
    print(f"---FOUND {len(filtered_docs)} RELEVANT DOCUMENTS---")
    
    return {
        "documents": filtered_docs,
        "question": question,
        "web_search": web_search
    }