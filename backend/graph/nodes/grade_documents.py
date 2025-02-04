"""
Module for evaluating document relevance to questions.
Filters and grades retrieved documents based on their relevance.
"""

from typing import Any, Dict, List
from langchain.schema import Document
from backend.graph.chains.retrieval_grader import retrieval_grader
from backend.graph.state import GraphState

class DocumentGrader:
    """
    Handles the evaluation and filtering of documents based on relevance.
    """

    @staticmethod
    def _grade_document(question: str, document: Document) -> bool:
        """
        Grade a single document's relevance to a question.
        
        Args:
            question: Question to check relevance against
            document: Document to evaluate
            
        Returns:
            True if document is relevant, False otherwise
        """
        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        grade = retrieval_grader.invoke({
            "question": question,
            "document": document.page_content
        })
        
        is_relevant = grade.binary_score
        print(f"---DOCUMENT IS {'RELEVANT' if is_relevant else 'NOT RELEVANT'}---")
        return is_relevant

    @staticmethod
    def filter_relevant_documents(
        question: str,
        documents: List[Document]
    ) -> List[Document]:
        """
        Filter documents based on their relevance to the question.
        
        Args:
            question: Question to check relevance against
            documents: List of documents to filter
            
        Returns:
            List of relevant documents
        """
        relevant_docs = [
            doc for doc in documents 
            if DocumentGrader._grade_document(question, doc)
        ]
        
        print(f"---FOUND {len(relevant_docs)} RELEVANT DOCUMENTS---")
        return relevant_docs

def grade_documents(state: GraphState) -> Dict[str, Any]:
    """
    Grade and filter documents based on their relevance to the question.
    
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
    filtered_docs = DocumentGrader.filter_relevant_documents(question, documents)
    
    return {
        "documents": filtered_docs,
        "question": question,
        "web_search": web_search
    }