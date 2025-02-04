from typing import Optional
from datetime import datetime

from backend.graph.graph import app
from frontend.ui.factory import UIFactory
from frontend.ui.interfaces.base import MessagingInterface
from frontend.ui.interfaces.state import StateInterface
from frontend.ui.interfaces.markup import MarkupInterface

def format_response(response: dict) -> str:
    """Format the graph response into a readable string"""
    if not isinstance(response, dict):
        return str(response)
    
    # Extract the generation (answer) from the response
    generation = response.get('generation', '')
    
    # Format source documents if present
    documents = response.get('documents', [])
    sources_text = ""
    if documents:
        sources_text = "\n\n**Sources:**\n"
        for i, doc in enumerate(documents, 1):
            content = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            sources_text += f"\n{i}. {content}\n"
    
    # Combine the answer with sources
    formatted_response = f"{generation}{sources_text}"
    return formatted_response

def render_rag_chat(
    ui: Optional[MessagingInterface] = None,
    state: Optional[StateInterface] = None,
    markup: Optional[MarkupInterface] = None
):
    """Renders the RAG chat interface that uses the graph for responses"""
    ui = ui or UIFactory.create_ui()
    state = state or UIFactory.create_state()
    markup = markup or UIFactory.create_markup()

    # Initialize chat history in session state
    state.init_default("messages", [])
    messages = state.get("messages")

    markup.markdown("### Chat")

    # Display chat history
    for message in messages:
        with ui.chat_message(message["role"]):
            markup.markdown(message["content"])

    # Chat input
    if prompt := ui.chat_input("What would you like to know?"):
        # Display user message
        with ui.chat_message("user"):
            markup.markdown(prompt)
        
        # Add user message to chat history
        messages.append({"role": "user", "content": prompt})
        state.set("messages", messages)

        # Get response from graph
        with ui.chat_message("assistant"):
            with ui.spinner("Thinking..."):
                try:
                    # Convert messages to ChatMessage format
                    chat_history = [
                        {
                            "role": msg["role"],
                            "content": msg["content"],
                            "timestamp": datetime.now().isoformat(),
                            "documents_used": None
                        }
                        for msg in messages[:-1]  # Exclude the last message as it will be added by generate
                    ]
                    
                    # Pass chat history to the graph
                    response = app.invoke(input={
                        "question": prompt,
                        "chat_history": chat_history
                    })
                    
                    formatted_response = format_response(response)
                    markup.markdown(formatted_response)
                    # Add assistant response to chat history
                    messages.append(
                        {"role": "assistant", "content": formatted_response}
                    )
                    state.set("messages", messages)
                except Exception as e:
                    error_message = f"Error getting response: {str(e)}"
                    ui.error(error_message)
                    messages.append(
                        {"role": "assistant", "content": error_message}
                    )
                    state.set("messages", messages)

    # Add a button to clear chat history
    if ui.button("Clear Chat History"):
        state.set("messages", [])
        ui.rerun() 