"""Activation extraction and manipulation utilities."""

from typing import Dict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def gather_residual_activations(
    model: AutoModelForCausalLM,
    layer_idx: int,
    inputs: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """
    Extract residual stream activations at a specified layer.

    Args:
        model: HuggingFace model
        layer_idx: Index of the transformer layer (0-indexed)
        inputs: Tokenized inputs dict with 'input_ids', 'attention_mask', etc.

    Returns:
        Residual activations tensor of shape (batch, seq_len, d_model)
    """
    activations = {}

    def hook_fn(module, inp, out):
        # Handle both tuple and tensor outputs
        if isinstance(out, tuple):
            activations["resid"] = out[0]
        else:
            activations["resid"] = out

    handle = model.model.layers[layer_idx].register_forward_hook(hook_fn)

    with torch.no_grad():
        model(**inputs)

    handle.remove()

    return activations["resid"]


def get_last_token_residual(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    layer: int,
) -> torch.Tensor:
    """
    Extract the residual stream activation for the last prompt token.

    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        prompt: Input text prompt
        layer: Layer index to extract activations from

    Returns:
        Residual activation vector for last token, shape (d_model,)
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    resid = gather_residual_activations(model, layer, inputs)

    # Extract last token, remove batch dimension
    if resid.ndim == 3:
        resid = resid[0]  # (seq_len, d_model)

    return resid[-1].detach().cpu()
