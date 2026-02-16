"""
Evaluation Module
=================
Provides metrics computation and visualisation utilities for the credit risk
classification pipeline, supporting the 8-way model comparison described in:

    "Quantum-Enhanced Smart Computing Framework for Sustainable Credit Risk
     Decision Communication" (Jain, Singh, et al., 2026)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


# ═══════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute a comprehensive metrics dictionary.

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted labels.
    y_proba : np.ndarray, optional
        Predicted probabilities for the positive class (used for AUC-ROC).

    Returns
    -------
    dict
        Keys: ``accuracy``, ``precision``, ``recall``, ``f1``, ``auc_roc``.
    """
    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    if y_proba is not None:
        try:
            # Handle both 1-D (positive class) and 2-D (all classes) arrays
            if y_proba.ndim == 2:
                y_score = y_proba[:, 1]
            else:
                y_score = y_proba
            metrics["auc_roc"] = float(roc_auc_score(y_true, y_score))
        except ValueError:
            metrics["auc_roc"] = float("nan")
    else:
        metrics["auc_roc"] = float("nan")

    return metrics


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
) -> None:
    """Pretty-print a scikit-learn classification report.

    Parameters
    ----------
    y_true, y_pred : np.ndarray
        True and predicted labels.
    model_name : str
        Display name.
    """
    print(f"\n{'─' * 50}")
    print(f" Classification Report: {model_name}")
    print(f"{'─' * 50}")
    print(classification_report(y_true, y_pred, zero_division=0))


# ═══════════════════════════════════════════════════════════════════════════
# Visualisations
# ═══════════════════════════════════════════════════════════════════════════

# Set publication-quality defaults
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[str | Path] = None,
    model_name: str = "Model",
) -> None:
    """Plot and optionally save a confusion matrix heatmap.

    Parameters
    ----------
    y_true, y_pred : np.ndarray
        True and predicted labels.
    save_path : str or Path, optional
        If provided, the figure is saved to this path.
    model_name : str
        Title prefix.
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Low Risk", "High Risk"],
        yticklabels=["Low Risk", "High Risk"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{model_name} — Confusion Matrix")
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        print(f"[eval] Saved confusion matrix → {save_path}")
    plt.close(fig)


def plot_roc_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    save_path: Optional[str | Path] = None,
    model_name: str = "Model",
) -> None:
    """Plot and optionally save a ROC curve.

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_scores : np.ndarray
        Probability scores for the positive class (1-D) or all classes (2-D).
    save_path : str or Path, optional
        Save path.
    model_name : str
        Title prefix.
    """
    if y_scores.ndim == 2:
        y_scores = y_scores[:, 1]

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"{model_name} (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{model_name} — ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        print(f"[eval] Saved ROC curve → {save_path}")
    plt.close(fig)


def plot_model_comparison(
    results_dict: Dict[str, Dict[str, float]],
    save_path: Optional[str | Path] = None,
) -> None:
    """Bar chart comparing all models across accuracy, precision, recall, F1.

    Parameters
    ----------
    results_dict : dict
        ``{model_name: {metric_name: value, …}, …}``.
    save_path : str or Path, optional
        Save path for the figure.
    """
    metrics_to_plot = ["accuracy", "precision", "recall", "f1"]
    model_names = list(results_dict.keys())
    n_models = len(model_names)
    n_metrics = len(metrics_to_plot)

    x = np.arange(n_metrics)
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.cm.get_cmap("tab10", n_models)

    for idx, name in enumerate(model_names):
        vals = [results_dict[name].get(m, 0) for m in metrics_to_plot]
        offset = (idx - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=name, color=cmap(idx), edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in metrics_to_plot])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — Credit Risk Classification")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        print(f"[eval] Saved comparison chart → {save_path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Persistence
# ═══════════════════════════════════════════════════════════════════════════

def save_results(
    results_dict: Dict[str, Dict[str, Any]],
    path: str | Path,
) -> None:
    """Save evaluation results to a JSON file.

    Parameters
    ----------
    results_dict : dict
        ``{model_name: {metric_name: value, …}, …}``.
    path : str or Path
        Output JSON path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types for JSON serialisation
    def _convert(obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    serialisable = {
        model: {k: _convert(v) for k, v in metrics.items()}
        for model, metrics in results_dict.items()
    }

    with open(path, "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"[eval] Saved results → {path}")


# ---------------------------------------------------------------------------
# Module self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Evaluation Module — Self-Test")
    print("=" * 60)

    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, 100)
    y_pred = rng.integers(0, 2, 100)
    y_proba = rng.random((100, 2))
    y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)

    m = compute_metrics(y_true, y_pred, y_proba)
    print(f"\nMetrics: {m}")

    print_classification_report(y_true, y_pred, "Self-Test Model")

    plot_confusion_matrix(y_true, y_pred, model_name="Self-Test")
    plot_roc_curve(y_true, y_proba, model_name="Self-Test")

    fake_results = {
        "ModelA": {"accuracy": 0.90, "precision": 0.88, "recall": 0.91, "f1": 0.89},
        "ModelB": {"accuracy": 0.85, "precision": 0.83, "recall": 0.87, "f1": 0.85},
    }
    plot_model_comparison(fake_results)

    print("\n✓ Evaluation self-test passed.")
