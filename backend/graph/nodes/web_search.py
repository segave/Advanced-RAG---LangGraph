"""
Module for performing web searches and processing results.
Handles web queries and document conversion using Tavily Search API.
"""

from typing import Any, Dict, List
from langchain.schema import Document
from langchain_community.tools.tavily_search import TavilySearchResults
from backend.graph.state import GraphState
from dotenv import load_dotenv

class WebSearcher:
    """
    Handles web search operations using Tavily Search API.
    """

    def __init__(self, max_results: int = 3):
        """
        Initialize the web searcher with configuration.
        
        Args:
            max_results: Maximum number of search results to retrieve
        """
        load_dotenv()
        self.max_results = max_results
        self._setup_search_tool()

    def _setup_search_tool(self) -> None:
        """Set up the Tavily search tool."""
        self.search_tool = TavilySearchResults(max_results=self.max_results)

    def _process_results(self, results: List[Dict[str, str]]) -> Document:
        """
        Process and combine search results into a single document.
        
        Args:
            results: List of search results from Tavily
            
        Returns:
            Document containing combined search results
        """
        combined_content = "\n".join(
            [result["content"] for result in results]
        )
        return Document(
            page_content=combined_content,
            metadata={"source": "web_search"}
        )

    def search(self, query: str) -> Document:
        """
        Perform web search and process results.
        
        Args:
            query: Search query string
            
        Returns:
            Document containing processed search results
        """
        print(f"---SEARCHING WEB FOR: {query}---")
        results = self.search_tool.invoke({"query": query})
        return self._process_results(results)

def web_search(state: GraphState) -> Dict[str, Any]:
    """
    Perform web search for a given question and update state.
    
    Args:
        state: Current graph state containing question and context
        
    Returns:
        Updated state with web search results
    """
    print("---WEB SEARCH---")
    
    # Get state variables
    question = state["question"]
    documents = state.get("documents", [])
    generation_attempts = state.get("generation_attempts", 0)

    # Perform web search
    searcher = WebSearcher()
    web_results = searcher.search(question)
    documents.append(web_results)
    
    return {
        "documents": documents,
        "question": question,
        "web_search": True,
        "generation_attempts": generation_attempts,
    }


if __name__ == "__main__":
    web_search(state={"question": "agent memory", "documents": None})