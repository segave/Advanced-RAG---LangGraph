"""Templates for entry classification prompts."""

CLASSIFICATION_TEMPLATE = """You are deciding whether a question needs information search.

Question: {question}
Chat History: {chat_history}

Classify if this question:
1. Needs information search (True):
- Requires specific facts or data
- References external information
- Needs verification from sources

2. Can be answered directly (False):
- Is a clarification or follow-up
- Can be answered from chat history
- Is a general knowledge question
- Is about the conversation itself

Return True if search is needed, False if it can be answered directly.""" 