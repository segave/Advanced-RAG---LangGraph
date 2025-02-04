from typing import Any, Dict
from datetime import datetime
from backend.graph.chains.generation import generation_chain
from backend.graph.state import GraphState, ChatMessage


def generate(state: GraphState) -> Dict[str, Any]:
    print("---GENERATE---")
    
    # Get and increment generation attempts from previous state
    generation_attempts = state.get("generation_attempts", 0) + 1
    print(f"---GENERATION ATTEMPT {generation_attempts}/3---")
    
    question = state["question"]
    documents = state.get("documents", [])
    chat_history = state.get("chat_history", [])

    print(f"---CHAT HISTORY---")
    for message in chat_history:
        print(f"Role: {message['role']}, Content: {message['content']}, Timestamp: {message['timestamp']}, Documents Used: {message['documents_used']}")
    
    # Añadir pregunta del usuario al historial
    if not chat_history or chat_history[-1]["content"] != question:
        chat_history.append(ChatMessage(
            role="user",
            content=question,
            timestamp=datetime.now().isoformat(),
            documents_used=None
        ))
    
    # Generar respuesta considerando el historial
    context = "\n\n".join([doc.page_content for doc in documents])
    generation = generation_chain.invoke({
        "question": question,
        "context": context,
        "chat_history": chat_history
    })
    
    # Añadir respuesta al historial
    chat_history.append(ChatMessage(
        role="assistant",
        content=generation,
        timestamp=datetime.now().isoformat(),
        documents_used=[doc.metadata.get("source", "unknown") for doc in documents]
    ))
    
    return {
        "generation": generation, 
        "documents": documents, 
        "question": question,
        "generation_attempts": generation_attempts,
        "chat_history": chat_history
    }