"""Prompt generation and manipulation."""

from .conflict_prompts import (
    build_priority_prompt,
    build_temporal_prompt,
    generate_priority_paraphrases,
    first_line,
    LABEL_MAP,
)

__all__ = [
    "build_priority_prompt",
    "build_temporal_prompt",
    "generate_priority_paraphrases",
    "first_line",
    "LABEL_MAP",
]
