from abc import ABC, abstractmethod

class MarkupInterface(ABC):
    """Interface for markup rendering."""
    
    @abstractmethod
    def header(self, text: str) -> None:
        """Render a header."""
        pass
    
    @abstractmethod
    def markdown(self, text: str, unsafe_allow_html: bool = False) -> None:
        """Render markdown text."""
        pass
    
    @abstractmethod
    def columns(self, widths: list[int]) -> list:
        """Create columns with specified widths."""
        pass 