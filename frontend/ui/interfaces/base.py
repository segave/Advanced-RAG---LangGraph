from abc import ABC, abstractmethod
from typing import Any, Optional, ContextManager

class InputInterface(ABC):
    """Interface defining basic input UI components.
    
    This interface defines the contract for rendering basic input elements
    like text inputs and text areas.
    """
    
    @abstractmethod
    def text_input(self, label: str, value: str = "", **kwargs) -> str:
        """Render a text input field.
        
        Args:
            label (str): Label text for the input
            value (str, optional): Initial value. Defaults to "".
            **kwargs: Additional styling or behavior parameters
        
        Returns:
            str: The current value of the input
        """
        pass
    
    @abstractmethod
    def text_area(self, label: str, value: str = "", **kwargs) -> str:
        """Render a multi-line text input area.
        
        Args:
            label (str): Label text for the text area
            value (str, optional): Initial value. Defaults to "".
            **kwargs: Additional styling or behavior parameters
        
        Returns:
            str: The current value of the text area
        """
        pass

    @abstractmethod
    def expander(self, label: str, **kwargs) -> Any:
        """Create an expandable container.
        
        Args:
            label (str): Label for the expander
            **kwargs: Additional styling or behavior parameters
        
        Returns:
            Any: Context manager for the expander
        """
        pass

    @abstractmethod
    def button(self, label: str, **kwargs) -> bool:
        """Create a clickable button.
        
        Args:
            label (str): Text to display on the button
            **kwargs: Additional styling or behavior parameters
        
        Returns:
            bool: True if button was clicked, False otherwise
        """
        pass

class SelectionInterface(ABC):
    """Interface defining selection UI components.
    
    This interface defines the contract for rendering selection elements
    like dropdown menus and option lists.
    """
    
    @abstractmethod
    def select_box(self, label: str, options: list, **kwargs) -> Any:
        """Render a dropdown selection box.
        
        Args:
            label (str): Label text for the select box
            options (list): List of available options
            **kwargs: Additional styling or behavior parameters
        
        Returns:
            Any: The currently selected value
        """
        pass

class MessagingInterface(ABC):
    """Interface defining message display components.
    
    This interface defines the contract for rendering different types of
    messages like success notifications and error alerts.
    """
    
    @abstractmethod
    def success(self, message: str) -> None:
        """Display a success message.
        
        Args:
            message (str): The success message to display
        """
        pass
    
    @abstractmethod
    def error(self, message: str) -> None:
        """Display an error message.
        
        Args:
            message (str): The error message to display
        """
        pass

    @abstractmethod
    def chat_message(self, role: str) -> ContextManager:
        """Creates a chat message container"""
        pass

    @abstractmethod
    def chat_input(self, placeholder: str) -> Optional[str]:
        """Creates a chat input box"""
        pass

    @abstractmethod
    def spinner(self, text: str) -> ContextManager:
        """Creates a loading spinner with text"""
        pass

    @abstractmethod
    def rerun(self) -> None:
        """Reruns the app"""
        pass

class ChatInterface(ABC):
    """Interface defining chat UI components.
    
    This interface defines the contract for rendering chat-related elements
    like message inputs and chat bubbles.
    """
    
    @abstractmethod
    def chat_input(self, placeholder: str, **kwargs) -> str:
        """Render a chat input field.
        
        Args:
            placeholder (str): Placeholder text for the input
            **kwargs: Additional styling or behavior parameters
        
        Returns:
            str: The current value of the chat input
        """
        pass
    
    @abstractmethod
    def chat_message(self, role: str, **kwargs):
        """Render a chat message bubble.
        
        Args:
            role (str): The role of the message sender (e.g., "user", "assistant")
            **kwargs: Additional styling or content parameters
        """
        pass

class UploadInterface(ABC):
    """Interface defining file upload components.
    
    This interface defines the contract for rendering file upload elements
    that allow users to upload documents and other files.
    """
    
    @abstractmethod
    def file_uploader(self, label: str, type: list, **kwargs) -> Any:
        """Render a file upload component.
        
        Args:
            label (str): Label text for the uploader
            type (list): List of allowed file extensions
            **kwargs: Additional styling or behavior parameters
        
        Returns:
            Any: The uploaded file object or None if no file is uploaded
        """
        pass 