"""
Module containing utility functions for graph state management and decisions.
Handles state transitions and evaluation logic.
"""

from typing import Dict, Any
from backend.graph.state import GraphState
from backend.graph.chains.answer_grader import answer_grader
from backend.graph.chains.hallucination_grader import hallucination_grader
from backend.graph.chains.entry_classifier import entry_classifier
from backend.graph.consts import RETRIEVE, GENERATE, WEBSEARCH

def decide_next_step(state: GraphState) -> str:
    """
    Decide next step based on document relevance assessment.
    
    Args:
        state: Current graph state with documents and search status
        
    Returns:
        Next node to execute in the graph
    """
    print("---ASSESS GRADED DOCUMENTS---")
    
    if not state["documents"] or state["web_search"]:
        print("---DECISION: NO RELEVANT DOCUMENTS FOUND, GO TO WEB SEARCH---")
        return WEBSEARCH
    else:
        print("---DECISION: RELEVANT DOCUMENTS FOUND, GENERATE ANSWER---")
        return GENERATE

def grade_generation_grounded_in_documents_and_question(state: GraphState) -> str:
    """
    Evaluate if generated response is grounded in documents and answers question.
    
    Args:
        state: Current graph state with generation and context
        
    Returns:
        Decision on generation quality: 'useful', 'not useful', or 'not supported'
    """
    print("---CHECK GENERATION---")
    question = state["question"]
    documents = state.get("documents", [])
    generation = state["generation"]

    # Handle direct generation without documents
    if not documents:
        print("---DIRECT GENERATION, CHECKING ONLY ANSWER RELEVANCE---")
        score = answer_grader.invoke({
            "question": question,
            "generation": generation
        })
        if score.binary_score:
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"

    # Handle generation with documents
    if state["generation_attempts"] >= 3:
        state["generation"] = (
            generation + 
            "\n\nNOTE: This response was generated after multiple attempts "
            "and might not be fully grounded in the available documents."
        )
        print("---MAX GENERATION ATTEMPTS REACHED, RETURNING CURRENT RESPONSE---")
        return "useful"

    score = hallucination_grader.invoke({
        "documents": documents,
        "generation": generation
    })

    if score.binary_score:
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({
            "question": question,
            "generation": generation
        })
        if score.binary_score:
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        print(f"---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, "
              f"RE-TRY (Attempt {state['generation_attempts']}/3)---")
        return "not supported"

def decide_entry_point(state: GraphState) -> str:
    """
    Decide whether to search for information or generate directly.
    
    Args:
        state: Current graph state with question and chat history
        
    Returns:
        Initial node to execute in the graph
    """
    print("---DECIDE ENTRY POINT---")
    
    question = state["question"]
    chat_history = state.get("chat_history", [])
    
    decision = entry_classifier.invoke({
        "question": question,
        "chat_history": chat_history
    })
    
    if decision.needs_search:
        print("---DECISION: NEED TO SEARCH FOR INFORMATION---")
        return RETRIEVE
    else:
        print("---DECISION: CAN GENERATE DIRECTLY---")
        return GENERATE 