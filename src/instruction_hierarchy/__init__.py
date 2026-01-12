"""
Instruction Hierarchy: Mechanistic Interpretability of Conflicting Instructions

This package provides tools for analyzing and steering language model behavior
when presented with conflicting instructions of different priorities.
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from . import models
from . import steering
from . import prompts
from . import evaluation
from . import utils

__all__ = ["models", "steering", "prompts", "evaluation", "utils"]
