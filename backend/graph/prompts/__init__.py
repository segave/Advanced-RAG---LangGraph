"""
Package for prompt templates used in various chains.
"""

from .templates.generation_template import RESPONSE_TEMPLATE
from .templates.entry_classifier_template import CLASSIFICATION_TEMPLATE
from .templates.hallucination_grader_template import HALLUCINATION_TEMPLATE
from .templates.retrieval_grader_template import RELEVANCE_TEMPLATE

__all__ = [
    'RESPONSE_TEMPLATE',
    'CLASSIFICATION_TEMPLATE',
    'HALLUCINATION_TEMPLATE',
    'RELEVANCE_TEMPLATE'
] 