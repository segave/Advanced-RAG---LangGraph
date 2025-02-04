from typing import Any, Dict

from langchain.schema import Document
from langchain_community.tools.tavily_search import TavilySearchResults

from backend.graph.state import GraphState
from dotenv import load_dotenv

load_dotenv()
web_search_tool = TavilySearchResults(max_results=3)


def web_search(state: GraphState) -> Dict[str, Any]:
    print("---WEB SEARCH---")
    
    generation_attempts = state.get("generation_attempts", 0)
    chat_history = state.get("chat_history", [])
    
    question = state["question"]
    documents = state.get("documents", [])

    tavily_results = web_search_tool.invoke({"query": question})
    joined_tavily_result = "\n".join(
        [tavily_result["content"] for tavily_result in tavily_results]
    )
    web_results = Document(
        page_content=joined_tavily_result,
        metadata={"source": "web_search"}
    )
    documents.append(web_results)
    
    return {
        "documents": documents,
        "question": question,
        "web_search": True,
        "generation_attempts": generation_attempts,
        "chat_history": chat_history
    }


if __name__ == "__main__":
    web_search(state={"question": "agent memory", "documents": None})