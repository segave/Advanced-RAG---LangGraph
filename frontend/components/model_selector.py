from typing import Optional

from frontend.ui.factory import UIFactory
from frontend.ui.interfaces.base import SelectionInterface
from frontend.ui.interfaces.state import StateInterface
from frontend.ui.interfaces.markup import MarkupInterface


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
    state.set("selected_model", models[selected_model_name]) 