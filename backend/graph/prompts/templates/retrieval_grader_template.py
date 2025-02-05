"""Templates for document retrieval grading prompts."""

RELEVANCE_TEMPLATE = """You are an expert evaluating document relevance to questions.

Document to evaluate:
{document}

Question:
{question}

Instructions:
1. Check for keyword matches
2. Assess semantic relevance
3. Consider related concepts
4. Look for contextual connections

A document is relevant if it:
- Contains keywords from the question
- Has semantically related content
- Provides context for the answer
- Contains information needed to answer

Return True if the document is relevant, False otherwise.""" 