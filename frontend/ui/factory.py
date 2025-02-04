from typing import Dict, Type, Union

from .interfaces.base import (
    InputInterface,
    SelectionInterface,
    MessagingInterface,
    ChatInterface,
    UploadInterface
)

from .interfaces.state import StateInterface
from .interfaces.markup import MarkupInterface
from .streamlit_impl.streamlit_renderer import StreamlitRenderer
from .streamlit_impl.state import StreamlitState
from .streamlit_impl.markup import StreamlitMarkup

UIType = Union[
    InputInterface,
    SelectionInterface,
    MessagingInterface,
    ChatInterface,
    UploadInterface
]

class UIFactory:
    """Factory for creating UI, state, and markup components.
    
    This factory provides methods to create different UI implementations,
    managing the instantiation of renderers, state handlers, and markup generators.
    
    Attributes:
        _renderers (Dict[str, Type[UIType]]): Mapping of renderer names to their classes
    """
    
    _renderers: Dict[str, Type[UIType]] = {
        "streamlit": StreamlitRenderer
    }
    
    @classmethod
    def register_renderer(cls, name: str, renderer_class: Type[UIType]) -> None:
        """Register a new UI renderer implementation.
        
        Args:
            name (str): Unique identifier for the renderer
            renderer_class (Type[UIType]): The renderer class to register
        """
        cls._renderers[name] = renderer_class
    
    @classmethod
    def create(cls, renderer_type: str = "streamlit") -> UIType:
        """Create a UI renderer instance of the specified type.
        
        Args:
            renderer_type (str, optional): Type of renderer to create. Defaults to "streamlit".
        
        Returns:
            UIType: An instance of the requested renderer type
        
        Raises:
            ValueError: If the renderer type is not registered
        """
        if renderer_type not in cls._renderers:
            raise ValueError(f"Unknown renderer type: {renderer_type}")
        
        return cls._renderers[renderer_type]()
    
    @classmethod
    def create_ui(cls) -> StreamlitRenderer:
        """Create a new UI renderer instance.
        
        Returns:
            StreamlitRenderer: A configured UI renderer implementation
        """
        return StreamlitRenderer()
    
    @classmethod
    def create_state(cls) -> StateInterface:
        """Create a new state management instance.
        
        Returns:
            StateInterface: A configured state management implementation
        """
        return StreamlitState()
    
    @classmethod
    def create_markup(cls) -> MarkupInterface:
        """Create a new markup renderer instance.
        
        Returns:
            MarkupInterface: A configured markup rendering implementation
        """
        return StreamlitMarkup()