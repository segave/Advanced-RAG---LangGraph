"""
Module defining the core graph structure and workflow.
"""

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

from backend.graph.state import GraphState
from backend.graph.nodes import generate, grade_documents, retrieve, web_search
from backend.graph.utils import (
    decide_next_step,
    grade_generation_grounded_in_documents_and_question,
    decide_entry_point
)
from backend.graph.consts import RETRIEVE, GRADE_DOCUMENTS, GENERATE, WEBSEARCH

load_dotenv()

# Create graph
workflow = StateGraph(GraphState)

# Add nodes
workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(GENERATE, generate)
workflow.add_node(WEBSEARCH, web_search)

# Set entry point
workflow.set_conditional_entry_point(
    decide_entry_point,
    {
        GENERATE: GENERATE,
        RETRIEVE: RETRIEVE,
    },
)

# Add edges
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

# Compile graph
app = workflow.compile()

# Generate visualization
app.get_graph().draw_mermaid_png(output_file_path="graph.png")