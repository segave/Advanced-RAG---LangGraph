"""Templates for answer grading prompts."""

ANSWER_GRADE_TEMPLATE = """You are an expert evaluator assessing how well an answer addresses a question.

Question: {question}
Answer: {generation}

Evaluate whether the answer directly and adequately addresses the question.
Consider:
1. Relevance to the question
2. Completeness of the response
3. Accuracy of information

Provide your evaluation as:
- binary_score: true if answer addresses question, false otherwise
"""