from dotenv import load_dotenv

from langgraph.graph import END, StateGraph

from backend.graph.chains.answer_grader import answer_grader
from backend.graph.chains.hallucination_grader import hallucination_grader
from backend.graph.consts import RETRIEVE, GRADE_DOCUMENTS, GENERATE, WEBSEARCH
from backend.graph.nodes import generate, grade_documents, retrieve, web_search
from backend.graph.state import GraphState
from backend.graph.chains.entry_classifier import entry_classifier


load_dotenv()


def decide_next_step(state):
    print("---ASSESS GRADED DOCUMENTS---")
    
    if not state["documents"] or state["web_search"]:
        print("---DECISION: NO RELEVANT DOCUMENTS FOUND, GO TO WEB SEARCH---")
        return WEBSEARCH
    else:
        print("---DECISION: RELEVANT DOCUMENTS FOUND, GENERATE ANSWER---")
        return GENERATE


def grade_generation_grounded_in_documents_and_question(state: GraphState) -> str:
    print("---CHECK GENERATION---")
    question = state["question"]
    documents = state.get("documents", [])
    generation = state["generation"]

    # Si no hay documentos, significa que fue una generación directa
    if not documents:
        print("---DIRECT GENERATION, CHECKING ONLY ANSWER RELEVANCE---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        if answer_grade := score.binary_score:
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"

    # Si hay documentos, verificar alucinaciones y relevancia
    if state["generation_attempts"] >= 3:
        state["generation"] = (
            generation + 
            "\n\nNOTE: This response was generated after multiple attempts and might not be fully grounded in the available documents."
        )
        print("---MAX GENERATION ATTEMPTS REACHED, RETURNING CURRENT RESPONSE---")
        return "useful"

    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )

    if hallucination_grade := score.binary_score:
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        if answer_grade := score.binary_score:
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        print(f"---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY (Attempt {state['generation_attempts']}/3)---")
        return "not supported"


def decide_entry_point(state: GraphState) -> str:
    """Decide whether to search for information or generate directly."""
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


workflow = StateGraph(GraphState)

workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(GENERATE, generate)
workflow.add_node(WEBSEARCH, web_search)

workflow.set_conditional_entry_point(
    decide_entry_point,
    {
        GENERATE: GENERATE,
        RETRIEVE: RETRIEVE,
    },
)

workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)

workflow.add_conditional_edges(
    GRADE_DOCUMENTS,
    decide_next_step,
    {
        WEBSEARCH: WEBSEARCH,
        GENERATE: GENERATE,
    },
)

workflow.add_conditional_edges(
    GENERATE,
    grade_generation_grounded_in_documents_and_question,
    {
        "not supported": GENERATE,
        "useful": END,
        "not useful": WEBSEARCH,
    },
)
workflow.add_edge(WEBSEARCH, GENERATE)
workflow.add_edge(GENERATE, END)

app = workflow.compile()

app.get_graph().draw_mermaid_png(output_file_path="graph.png")