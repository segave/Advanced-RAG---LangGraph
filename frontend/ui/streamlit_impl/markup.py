import streamlit as st

from ..interfaces.markup import MarkupInterface

class StreamlitMarkup(MarkupInterface):
    """Streamlit implementation of markup rendering."""
    
    def header(self, text: str) -> None:
        st.header(text)
    
    def markdown(self, text: str, unsafe_allow_html: bool = False) -> None:
        st.markdown(text, unsafe_allow_html=unsafe_allow_html)
    
    def columns(self, widths: list[int]) -> list:
        return st.columns(widths) 