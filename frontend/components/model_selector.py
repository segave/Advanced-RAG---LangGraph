from typing import Optional

from frontend.ui.factory import UIFactory
from frontend.ui.interfaces.base import SelectionInterface
from frontend.ui.interfaces.state import StateInterface
from frontend.ui.interfaces.markup import MarkupInterface
from backend.graph.chains.generation import generation_chain
from backend.graph.chains.retrieval_grader import retrieval_grader
from backend.graph.chains.hallucination_grader import hallucination_grader
from backend.graph.chains.entry_classifier import entry_classifier

def reset_chains() -> None:
    """Reset all chains to use the newly selected model."""
    generation_chain._create_chain()
    retrieval_grader._create_chain()
    hallucination_grader._create_chain()
    entry_classifier._create_chain()

def render_model_selector(
    ui: Optional[SelectionInterface] = None,
    state: Optional[StateInterface] = None,
    markup: Optional[MarkupInterface] = None
):
    """Render the AI model selector."""
    ui = ui or UIFactory.create_ui()
    state = state or UIFactory.create_state()
    markup = markup or UIFactory.create_markup()
    
    # Crear columnas para centrar y reducir el ancho del selector
    _, col2, _ = markup.columns([3, 2, 3])
    
    # Available models
    models = {
        "GPT-4o": "gpt-4o",
        "GPT-4o-mini": "gpt-4o-mini",
    }
    
    # Initialize session state for model selection
    state.init_default("selected_model", "gpt-4o-mini")
    
    with col2:
        # Create the model selector
        selected_model_name = ui.select_box(
            "Select AI Model",
            options=list(models.keys()),
            index=list(models.keys()).index(
                next(k for k, v in models.items() if v == state.get("selected_model"))
            )
        )
    
    # Update the session state with the selected model
    new_model = models[selected_model_name]
    if new_model != state.get("selected_model"):
        state.set("selected_model", new_model)
        reset_chains() 