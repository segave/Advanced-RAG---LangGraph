"""
Module for generating responses to questions using LLM.
Provides functionality to generate contextual and chat-aware responses.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from typing import Dict, Any, Optional

class ResponseGenerator:
    """
    Generates responses to questions using context and chat history.
    Uses LLM to create informative and contextual responses.
    """
    
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0,
        max_tokens: Optional[int] = None
    ):
        """
        Initialize the generator with specific LLM configuration.
        
        Args:
            model_name: Name of the LLM model to use
            temperature: Temperature setting for generation
            max_tokens: Maximum tokens in response (None for model default)
        """
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )
        self._create_chain()

    def _create_chain(self) -> None:
        """Creates the generation chain with the response prompt."""
        template = """You are a helpful AI assistant. Answer the question based on the provided context and chat history.

        Context: {context}
        Chat History: {chat_history}
        Question: {question}

        Instructions:
        1. Use the context to ground your response
        2. Consider chat history for continuity
        3. Be clear and concise
        4. If you can't find relevant information, say so
        5. Don't make up information

        Your response:"""

        prompt = ChatPromptTemplate.from_template(template)
        self.chain = prompt | self.llm | StrOutputParser()

    def invoke(self, inputs: Dict[str, Any]) -> str:
        """
        Generate a response to a question using context and chat history.
        
        Args:
            inputs: Dictionary containing:
                - question: The question to answer
                - context: Relevant context for the answer
                - chat_history: Optional previous conversation
                
        Returns:
            Generated response as string
        """
        return self.chain.invoke(inputs)

# Create singleton instance
generation_chain = ResponseGenerator()