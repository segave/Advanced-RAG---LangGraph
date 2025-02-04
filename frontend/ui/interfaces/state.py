from abc import ABC, abstractmethod
from typing import Any

class StateInterface(ABC):
    """Interface defining application state management.
    
    This interface defines the contract for managing application state,
    including getting, setting, and initializing state values.
    """
    
    @abstractmethod
    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a value from the application state.
        
        Args:
            key (str): The state key to retrieve
            default (Any, optional): Default value if key not found. Defaults to None.
        
        Returns:
            Any: The value stored in state, or the default value
        """
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any) -> None:
        """Store a value in the application state.
        
        Args:
            key (str): The state key to set
            value (Any): The value to store
        """
        pass
    
    @abstractmethod
    def init_default(self, key: str, default: Any) -> None:
        """Initialize a key with default value if not exists."""
        pass 