"""Activation steering utilities for controlling model behavior."""

from contextlib import contextmanager
from typing import Callable
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def compute_steering_direction(
    activations_high: torch.Tensor,
    activations_low: torch.Tensor,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Compute steering direction from contrasting activation sets.

    Args:
        activations_high: Activations for high-priority outputs, shape (n_samples, d_model)
        activations_low: Activations for low-priority outputs, shape (n_samples, d_model)
        normalize: Whether to normalize the direction vector

    Returns:
        Steering direction vector, shape (d_model,)
    """
    mu_high = activations_high.mean(dim=0)
    mu_low = activations_low.mean(dim=0)

    direction = mu_high - mu_low

    if normalize:
        direction = direction / (direction.norm() + 1e-8)

    return direction


def _create_steering_hook(
    direction: torch.Tensor,
    scale: float,
    token_pos: int = -1,
) -> Callable:
    """
    Create a forward hook function for activation steering.

    Args:
        direction: Steering direction vector
        scale: Magnitude of steering intervention
        token_pos: Position in sequence to apply steering (-1 for last token)

    Returns:
        Hook function compatible with PyTorch's register_forward_hook
    """
    direction = direction.to(torch.float32)

    def hook_fn(module, inp, out):
        # Handle both tuple and tensor outputs
        if isinstance(out, tuple):
            hidden_states = out[0]
            rest = out[1:]
        else:
            hidden_states = out
            rest = None

        # Clone to avoid in-place modification issues
        hidden_states = hidden_states.clone()

        # Apply steering to specified token position
        hidden_states[0, token_pos, :] += scale * direction.to(hidden_states.device)

        if rest is None:
            return hidden_states
        return (hidden_states,) + rest

    return hook_fn


@contextmanager
def steer_model(
    model: AutoModelForCausalLM,
    layer: int,
    direction: torch.Tensor,
    scale: float,
    token_pos: int = -1,
):
    """
    Context manager for temporarily steering model activations.

    Args:
        model: HuggingFace model to steer
        layer: Layer index to apply steering
        direction: Steering direction vector
        scale: Magnitude of steering intervention
        token_pos: Position in sequence to apply steering

    Yields:
        None (model is steered within context)

    Example:
        >>> with steer_model(model, layer=17, direction=dir_vec, scale=10.0):
        ...     outputs = model.generate(**inputs)
    """
    hook = _create_steering_hook(direction, scale, token_pos)
    handle = model.model.layers[layer].register_forward_hook(hook)

    try:
        yield
    finally:
        handle.remove()


def generate_with_steering(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    layer: int,
    direction: torch.Tensor,
    scale: float,
    max_new_tokens: int = 3,
    **generation_kwargs,
) -> str:
    """
    Generate text with activation steering applied.

    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        prompt: Input prompt text
        layer: Layer to apply steering
        direction: Steering direction vector
        scale: Steering magnitude
        max_new_tokens: Maximum tokens to generate
        **generation_kwargs: Additional arguments for model.generate()

    Returns:
        Generated text (suffix only, excluding prompt)
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[-1]

    generation_kwargs.setdefault("do_sample", False)
    generation_kwargs.setdefault("pad_token_id", tokenizer.eos_token_id)

    if scale == 0:
        # No steering needed
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            **generation_kwargs,
        )
    else:
        with steer_model(model, layer, direction, scale):
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                **generation_kwargs,
            )

    # Extract only generated tokens
    generated_ids = outputs[0][prompt_len:]
    return tokenizer.decode(generated_ids, skip_special_tokens=False)
