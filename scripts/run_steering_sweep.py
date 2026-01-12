#!/usr/bin/env python
"""
Run steering experiments across multiple layers and scales.

This script performs a systematic sweep of activation steering parameters,
evaluating model behavior under different intervention strengths and locations.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict
import random

import torch
import numpy as np
from tqdm import tqdm

from instruction_hierarchy.models import (
    load_model_and_tokenizer,
    get_num_layers,
    get_last_token_residual,
)
from instruction_hierarchy.prompts import (
    build_priority_prompt,
    generate_priority_paraphrases,
    first_line,
    LABEL_MAP,
)
from instruction_hierarchy.steering import (
    compute_steering_direction,
    generate_with_steering,
)
from instruction_hierarchy.evaluation import summarize_outputs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_dataset(num_samples_per_class: int = 10) -> List[tuple]:
    """Build dataset of prompts with expected outputs."""
    dataset = []

    for token in ["ALPHA", "BETA"]:
        prompts = generate_priority_paraphrases(token)
        # Sample if we have more than needed
        if len(prompts) > num_samples_per_class:
            prompts = random.sample(prompts, num_samples_per_class)

        for prompt in prompts:
            dataset.append((token, prompt))

    return dataset


def compute_direction_from_dataset(
    model,
    tokenizer,
    dataset: List[tuple],
    layer: int,
) -> torch.Tensor:
    """Compute steering direction from a dataset of prompts."""
    activations_alpha = []
    activations_beta = []

    logger.info(f"Extracting activations at layer {layer}...")

    for expected, prompt in tqdm(dataset, desc="Processing prompts"):
        resid = get_last_token_residual(model, tokenizer, prompt, layer)

        if expected == "ALPHA":
            activations_alpha.append(resid)
        else:
            activations_beta.append(resid)

    # Stack and compute direction
    X_alpha = torch.stack(activations_alpha)
    X_beta = torch.stack(activations_beta)

    direction = compute_steering_direction(X_alpha, X_beta, normalize=True)

    logger.info(
        f"Computed direction from {len(X_alpha)} ALPHA and {len(X_beta)} BETA samples"
    )

    return direction


def evaluate_steering(
    model,
    tokenizer,
    test_dataset: List[tuple],
    layer: int,
    direction: torch.Tensor,
    scales: List[float],
) -> Dict:
    """Evaluate steering across multiple scales."""
    results = {}

    for scale in tqdm(scales, desc=f"Layer {layer}"):
        outputs = []

        for expected, prompt in test_dataset:
            generated = generate_with_steering(
                model,
                tokenizer,
                prompt,
                layer=layer,
                direction=direction,
                scale=scale,
                max_new_tokens=3,
            )
            got = first_line(generated)
            outputs.append((expected, got))

        # Compute metrics
        summary = summarize_outputs(outputs, LABEL_MAP)

        results[scale] = {
            "accuracy": summary["accuracy"],
            "p_primary": summary["p_primary"],
            "p_secondary": summary["p_secondary"],
            "p_empty": summary["p_empty"],
            "p_other": summary["p_other"],
            "n_samples": summary["n"],
        }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run steering experiments across layers and scales"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-3-1b-it",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[1, 10, 17, 25, 35],
        help="Layers to evaluate",
    )
    parser.add_argument(
        "--sweep-all-layers",
        action="store_true",
        help="Sweep all layers in the model",
    )
    parser.add_argument(
        "--scales",
        type=float,
        nargs="+",
        default=[-50, -30, -20, -10, -5, 0, 5, 10, 20, 30, 50],
        help="Steering scales to evaluate",
    )
    parser.add_argument(
        "--num-train-samples",
        type=int,
        default=10,
        help="Number of training samples per class",
    )
    parser.add_argument(
        "--num-test-samples",
        type=int,
        default=20,
        help="Number of test samples per class",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/steering_sweep.json",
        help="Output file for results",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load model
    logger.info(f"Loading model: {args.model}")
    torch.set_grad_enabled(False)
    model, tokenizer = load_model_and_tokenizer(args.model)

    # Determine layers to evaluate
    if args.sweep_all_layers:
        n_layers = get_num_layers(model)
        layers = list(range(1, n_layers))
        logger.info(f"Sweeping all {len(layers)} layers")
    else:
        layers = args.layers
        logger.info(f"Evaluating layers: {layers}")

    # Build datasets
    logger.info("Building datasets...")
    train_dataset = build_dataset(args.num_train_samples)
    test_dataset = build_dataset(args.num_test_samples)

    logger.info(
        f"Train set: {len(train_dataset)} samples, Test set: {len(test_dataset)} samples"
    )

    # Run experiments
    all_results = {}

    for layer in layers:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing layer {layer}")
        logger.info(f"{'='*60}")

        # Compute steering direction for this layer
        direction = compute_direction_from_dataset(
            model, tokenizer, train_dataset, layer
        )

        # Evaluate across scales
        layer_results = evaluate_steering(
            model, tokenizer, test_dataset, layer, direction, args.scales
        )

        all_results[layer] = layer_results

        # Log summary
        best_scale = max(
            layer_results.keys(), key=lambda s: layer_results[s]["accuracy"]
        )
        best_acc = layer_results[best_scale]["accuracy"]
        logger.info(f"Best accuracy: {best_acc:.3f} at scale {best_scale}")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results_dict = {
        "model": args.model,
        "layers": layers,
        "scales": args.scales,
        "num_train_samples": args.num_train_samples,
        "num_test_samples": args.num_test_samples,
        "results": {int(k): v for k, v in all_results.items()},
    }

    with open(output_path, "w") as f:
        json.dump(results_dict, f, indent=2)

    logger.info(f"\nResults saved to {output_path}")

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    for layer in layers:
        best_scale = max(
            all_results[layer].keys(),
            key=lambda s: all_results[layer][s]["accuracy"],
        )
        best_acc = all_results[layer][best_scale]["accuracy"]
        logger.info(f"Layer {layer:2d}: Best accuracy = {best_acc:.3f} @ scale {best_scale:+4.0f}")


if __name__ == "__main__":
    main()
