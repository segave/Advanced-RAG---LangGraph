"""Templates for hallucination grading prompts."""

HALLUCINATION_TEMPLATE = """You are an expert fact-checker evaluating if a response is grounded in provided facts.

Facts:
{documents}

Response to evaluate:
{generation}

Instructions:
1. Compare the response against the provided facts
2. Check if all claims are supported by the facts
3. Ignore stylistic differences or rephrasing
4. Focus on factual accuracy

Return True if the response is fully grounded in facts, False if it contains unsupported claims.""" 