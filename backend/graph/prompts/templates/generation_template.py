"""Templates for response generation prompts."""

RESPONSE_TEMPLATE = """You are a helpful AI assistant. Answer the question based on the provided context and chat history.

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