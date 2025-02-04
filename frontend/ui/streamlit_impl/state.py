import streamlit as st
from typing import Any

from ..interfaces.state import StateInterface

class StreamlitState(StateInterface):
    """Streamlit implementation of state management."""
    
    def get(self, key: str, default: Any = None) -> Any:
        return st.session_state.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        st.session_state[key] = value
    
    def init_default(self, key: str, default: Any) -> None:
        if key not in st.session_state:
            st.session_state[key] = default 