"""Model loading and initialization utilities."""

from typing import Optional, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(
    model_name: str,
    device_map: str = "auto",
    torch_dtype: Optional[torch.dtype] = None,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load a pretrained model and tokenizer.

    Args:
        model_name: HuggingFace model identifier (e.g., 'google/gemma-3-1b-it')
        device_map: Device mapping strategy for model loading
        torch_dtype: Optional torch dtype for model weights

    Returns:
        Tuple of (model, tokenizer)
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        torch_dtype=torch_dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Disable gradient computation by default for inference
    model.eval()

    return model, tokenizer


def get_num_layers(model: AutoModelForCausalLM) -> int:
    """
    Robustly infer number of transformer layers for common HF architectures.

    Args:
        model: HuggingFace model

    Returns:
        Number of transformer layers

    Raises:
        ValueError: If layer count cannot be inferred
    """
    # Most decoder-only HF models expose blocks at model.model.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return len(model.model.layers)

    # Fallbacks for other HF model wrappers (e.g., GPT-2)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return len(model.transformer.h)

    raise ValueError(
        "Could not infer layer count: inspect model attributes to locate the block list."
    )


def format_chat_prompt(user_prompt: str, model_type: str = "gemma") -> str:
    """
    Format a user prompt according to model-specific chat templates.

    Args:
        user_prompt: The user's input text
        model_type: Type of model ('gemma', 'llama', etc.)

    Returns:
        Formatted prompt string
    """
    if model_type == "gemma":
        return f"""<start_of_turn>user
{user_prompt}<end_of_turn>
<start_of_turn>model
"""
    else:
        # Default format for other models
        return user_prompt
