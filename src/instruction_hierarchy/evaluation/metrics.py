"""Evaluation metrics and analysis utilities."""

from collections import Counter
from typing import Dict, List, Tuple
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def compute_accuracy(
    expected: List[str],
    predicted: List[str],
) -> float:
    """
    Compute accuracy between expected and predicted outputs.

    Args:
        expected: List of expected outputs
        predicted: List of predicted outputs

    Returns:
        Accuracy as fraction of correct predictions
    """
    if len(expected) != len(predicted):
        raise ValueError("Expected and predicted lists must have same length")

    if len(expected) == 0:
        return 0.0

    correct = sum(1 for e, p in zip(expected, predicted) if e == p)
    return correct / len(expected)


def categorize_output(output: str, label_map: Dict[str, str]) -> str:
    """
    Categorize raw model output into standard labels.

    Args:
        output: Raw model output string
        label_map: Mapping from raw labels to canonical labels

    Returns:
        Categorized label: 'PRIMARY', 'SECONDARY', 'EMPTY', or 'OTHER'
    """
    if output is None:
        return "EMPTY"

    cleaned = output.strip()

    if cleaned == "":
        return "EMPTY"

    # Check if output matches known labels
    if cleaned in label_map:
        return label_map[cleaned]

    # Check if output is already a canonical label
    if cleaned in ("PRIMARY", "SECONDARY"):
        return cleaned

    return "OTHER"


def summarize_outputs(
    outputs: List[Tuple[str, str]],
    label_map: Dict[str, str],
) -> Dict:
    """
    Compute summary statistics for model outputs.

    Args:
        outputs: List of (expected, got) tuples
        label_map: Mapping from raw to canonical labels

    Returns:
        Dictionary with accuracy and output distribution statistics
    """
    n = len(outputs)
    if n == 0:
        return {
            "n": 0,
            "accuracy": 0.0,
            "p_primary": 0.0,
            "p_secondary": 0.0,
            "p_empty": 0.0,
            "p_other": 0.0,
            "counts": Counter(),
        }

    # Categorize outputs
    categorized = [
        (label_map.get(e, e), categorize_output(g, label_map))
        for e, g in outputs
    ]

    # Compute accuracy
    correct = sum(1 for e, g in categorized if e == g)
    accuracy = correct / n

    # Count output types
    got_list = [g for _, g in categorized]
    counts = Counter(got_list)

    return {
        "n": n,
        "accuracy": accuracy,
        "p_primary": counts.get("PRIMARY", 0) / n,
        "p_secondary": counts.get("SECONDARY", 0) / n,
        "p_empty": counts.get("EMPTY", 0) / n,
        "p_other": counts.get("OTHER", 0) / n,
        "counts": counts,
    }


@torch.no_grad()
def compute_continuation_logprob(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    continuation_ids: List[int],
) -> float:
    """
    Compute log probability of a continuation given a prompt.

    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        prompt: Input prompt text
        continuation_ids: Token IDs for continuation

    Returns:
        Sum of log probabilities across continuation tokens
    """
    device = next(model.parameters()).device
    prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    total_logprob = 0.0
    current_ids = prompt_ids

    for token_id in continuation_ids:
        # Get next-token logits
        outputs = model(input_ids=current_ids)
        logits = outputs.logits[:, -1, :]

        # Compute log probabilities
        logprobs = F.log_softmax(logits, dim=-1)
        total_logprob += logprobs[0, token_id].item()

        # Append token for next iteration
        current_ids = torch.cat(
            [current_ids, torch.tensor([[token_id]], device=device)],
            dim=1,
        )

    return total_logprob


def compute_logprob_margin(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    option_a: str,
    option_b: str,
) -> Tuple[float, float, float]:
    """
    Compute log probability margin between two continuation options.

    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        prompt: Input prompt text
        option_a: First continuation option
        option_b: Second continuation option

    Returns:
        Tuple of (margin, logprob_a, logprob_b) where margin = logprob_a - logprob_b
    """
    ids_a = tokenizer.encode(option_a, add_special_tokens=False)
    ids_b = tokenizer.encode(option_b, add_special_tokens=False)

    logprob_a = compute_continuation_logprob(model, tokenizer, prompt, ids_a)
    logprob_b = compute_continuation_logprob(model, tokenizer, prompt, ids_b)

    margin = logprob_a - logprob_b

    return margin, logprob_a, logprob_b
