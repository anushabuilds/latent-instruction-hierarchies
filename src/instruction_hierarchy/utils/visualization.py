"""Visualization utilities for analysis results."""

from typing import List, Dict
import numpy as np
import matplotlib.pyplot as plt


def plot_steering_heatmap(
    accuracy_matrix: np.ndarray,
    layers: List[int],
    scales: List[float],
    title: str = "Steering Accuracy Heatmap",
    save_path: str = None,
):
    """
    Plot heatmap of steering accuracy across layers and scales.

    Args:
        accuracy_matrix: 2D array of accuracies, shape (n_layers, n_scales)
        layers: List of layer indices
        scales: List of steering scales
        title: Plot title
        save_path: Optional path to save figure
    """
    plt.figure(figsize=(12, 8))
    plt.imshow(accuracy_matrix, aspect="auto", origin="lower", cmap="viridis")

    plt.xticks(range(len(scales)), scales, rotation=45)
    plt.yticks(range(len(layers)), layers)

    plt.xlabel("Steering Scale")
    plt.ylabel("Layer")
    plt.title(title)
    plt.colorbar(label="Accuracy")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_dose_response(
    scales: List[float],
    metrics: Dict[str, List[float]],
    layer: int,
    title: str = None,
    save_path: str = None,
):
    """
    Plot dose-response curve showing output distribution vs steering scale.

    Args:
        scales: List of steering scales
        metrics: Dictionary mapping metric names to lists of values
        layer: Layer index being analyzed
        title: Optional plot title
        save_path: Optional path to save figure
    """
    plt.figure(figsize=(10, 6))

    # Plot each metric
    if "p_primary" in metrics:
        plt.plot(scales, metrics["p_primary"], marker="o", label="P(PRIMARY)")
    if "p_secondary" in metrics:
        plt.plot(scales, metrics["p_secondary"], marker="o", label="P(SECONDARY)")
    if "p_empty" in metrics:
        plt.plot(scales, metrics["p_empty"], marker="o", label="P(EMPTY)")
    if "p_other" in metrics:
        plt.plot(scales, metrics["p_other"], marker="o", label="P(OTHER)")
    if "accuracy" in metrics:
        plt.plot(
            scales,
            metrics["accuracy"],
            marker="o",
            linestyle="--",
            linewidth=2,
            label="Accuracy",
        )

    plt.xlabel("Steering Scale")
    plt.ylabel("Fraction / Accuracy")

    if title is None:
        title = f"Dose-Response Curve at Layer {layer}"
    plt.title(title)

    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_logprob_margins(
    scales: List[float],
    margins: List[float],
    layer: int,
    title: str = None,
    save_path: str = None,
):
    """
    Plot log probability margins vs steering scale.

    Args:
        scales: List of steering scales
        margins: List of average log probability margins (ALPHA - BETA)
        layer: Layer index being analyzed
        title: Optional plot title
        save_path: Optional path to save figure
    """
    plt.figure(figsize=(10, 6))

    plt.plot(scales, margins, marker="o", linewidth=2)
    plt.axhline(y=0, color="k", linestyle="--", alpha=0.3)

    plt.xlabel("Steering Scale")
    plt.ylabel("Avg Log Prob Margin (ALPHA - BETA)")

    if title is None:
        title = f"Log Probability Margins at Layer {layer}"
    plt.title(title)

    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
