#!/usr/bin/env python
"""
Analyze and visualize steering experiment results.

This script loads results from steering sweep experiments and generates
comprehensive visualizations and summary statistics.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from instruction_hierarchy.utils import (
    plot_steering_heatmap,
    plot_dose_response,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_results(path: str) -> dict:
    """Load results from JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def create_accuracy_matrix(results: dict) -> tuple:
    """
    Create accuracy matrix from results.

    Returns:
        (accuracy_matrix, layers, scales)
    """
    layers = sorted([int(k) for k in results["results"].keys()])
    scales = sorted(results["scales"])

    acc_matrix = np.zeros((len(layers), len(scales)))

    for i, layer in enumerate(layers):
        layer_results = results["results"][str(layer)]
        for j, scale in enumerate(scales):
            acc_matrix[i, j] = layer_results[str(float(scale))]["accuracy"]

    return acc_matrix, layers, scales


def find_best_configurations(results: dict, top_k: int = 5) -> list:
    """Find top-k best (layer, scale) configurations."""
    configs = []

    for layer, layer_results in results["results"].items():
        for scale, metrics in layer_results.items():
            configs.append({
                "layer": int(layer),
                "scale": float(scale),
                "accuracy": metrics["accuracy"],
                "p_primary": metrics["p_primary"],
                "p_secondary": metrics["p_secondary"],
            })

    # Sort by accuracy
    configs.sort(key=lambda x: x["accuracy"], reverse=True)

    return configs[:top_k]


def analyze_layer_effectiveness(results: dict) -> dict:
    """Analyze which layers are most effective for steering."""
    layer_stats = {}

    for layer, layer_results in results["results"].items():
        accuracies = [metrics["accuracy"] for metrics in layer_results.values()]

        layer_stats[int(layer)] = {
            "max_accuracy": max(accuracies),
            "mean_accuracy": np.mean(accuracies),
            "std_accuracy": np.std(accuracies),
            "baseline_accuracy": layer_results["0.0"]["accuracy"],
        }

    return layer_stats


def plot_layer_effectiveness(layer_stats: dict, save_path: str = None):
    """Plot layer effectiveness metrics."""
    layers = sorted(layer_stats.keys())
    max_accs = [layer_stats[l]["max_accuracy"] for l in layers]
    mean_accs = [layer_stats[l]["mean_accuracy"] for l in layers]
    baseline_accs = [layer_stats[l]["baseline_accuracy"] for l in layers]

    plt.figure(figsize=(12, 6))

    plt.plot(layers, max_accs, marker="o", label="Max accuracy", linewidth=2)
    plt.plot(layers, mean_accs, marker="s", label="Mean accuracy", linewidth=2)
    plt.plot(
        layers,
        baseline_accs,
        marker="^",
        label="Baseline (no steering)",
        linewidth=2,
        linestyle="--",
    )

    plt.xlabel("Layer")
    plt.ylabel("Accuracy")
    plt.title("Layer Effectiveness for Steering")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze steering experiment results"
    )
    parser.add_argument(
        "input",
        type=str,
        help="Input JSON file with results",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/figures",
        help="Output directory for figures",
    )
    parser.add_argument(
        "--best-layer",
        type=int,
        default=None,
        help="Layer to plot dose-response curve for",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of best configurations to report",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    logger.info(f"Loading results from {args.input}")
    results = load_results(args.input)

    logger.info(f"Model: {results['model']}")
    logger.info(f"Layers evaluated: {len(results['results'])}")
    logger.info(f"Scales evaluated: {len(results['scales'])}")

    # Create accuracy matrix and heatmap
    logger.info("\nGenerating accuracy heatmap...")
    acc_matrix, layers, scales = create_accuracy_matrix(results)

    plot_steering_heatmap(
        acc_matrix,
        layers,
        scales,
        title=f"Steering Accuracy: {results['model']}",
        save_path=str(output_dir / "accuracy_heatmap.png"),
    )

    # Find best configurations
    logger.info(f"\nTop {args.top_k} configurations:")
    best_configs = find_best_configurations(results, args.top_k)

    for i, config in enumerate(best_configs, 1):
        logger.info(
            f"{i}. Layer {config['layer']:2d}, Scale {config['scale']:+5.0f}: "
            f"Accuracy = {config['accuracy']:.3f}"
        )

    # Analyze layer effectiveness
    logger.info("\nAnalyzing layer effectiveness...")
    layer_stats = analyze_layer_effectiveness(results)

    plot_layer_effectiveness(
        layer_stats,
        save_path=str(output_dir / "layer_effectiveness.png"),
    )

    # Plot dose-response for best layer
    if args.best_layer is not None:
        best_layer = args.best_layer
    else:
        # Auto-select best layer
        best_layer = max(
            layer_stats.keys(),
            key=lambda l: layer_stats[l]["max_accuracy"],
        )

    logger.info(f"\nPlotting dose-response curve for layer {best_layer}...")

    layer_results = results["results"][str(best_layer)]
    scales = sorted([float(s) for s in layer_results.keys()])

    metrics = {
        "accuracy": [layer_results[str(s)]["accuracy"] for s in scales],
        "p_primary": [layer_results[str(s)]["p_primary"] for s in scales],
        "p_secondary": [layer_results[str(s)]["p_secondary"] for s in scales],
        "p_empty": [layer_results[str(s)]["p_empty"] for s in scales],
        "p_other": [layer_results[str(s)]["p_other"] for s in scales],
    }

    plot_dose_response(
        scales,
        metrics,
        best_layer,
        title=f"Dose-Response Curve at Layer {best_layer}",
        save_path=str(output_dir / f"dose_response_layer_{best_layer}.png"),
    )

    # Save summary statistics
    summary_path = output_dir / "summary.txt"
    logger.info(f"\nSaving summary to {summary_path}")

    with open(summary_path, "w") as f:
        f.write(f"Model: {results['model']}\n")
        f.write(f"Layers evaluated: {len(results['results'])}\n")
        f.write(f"Scales evaluated: {len(results['scales'])}\n")
        f.write(f"\nTop {args.top_k} configurations:\n")

        for i, config in enumerate(best_configs, 1):
            f.write(
                f"{i}. Layer {config['layer']:2d}, Scale {config['scale']:+5.0f}: "
                f"Accuracy = {config['accuracy']:.3f}\n"
            )

        f.write("\nLayer effectiveness (max accuracy):\n")
        for layer in sorted(layer_stats.keys()):
            f.write(
                f"Layer {layer:2d}: {layer_stats[layer]['max_accuracy']:.3f} "
                f"(baseline: {layer_stats[layer]['baseline_accuracy']:.3f})\n"
            )

    logger.info(f"\nAnalysis complete. Figures saved to {output_dir}")


if __name__ == "__main__":
    main()
