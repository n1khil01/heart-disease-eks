from __future__ import annotations

from pathlib import Path
from typing import cast


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from matplotlib.container import BarContainer

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.dpi"] = 100


def _ensure_dir(path: str | Path) -> Path:
    """Make sure the parent directory exists before saving a figure."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def plot_target_distribution(df: pd.DataFrame, out_path: str | Path) -> None:
    """Side-by-side bar charts: raw 5-class target vs binarized 2-class target.

    Expects `df` to already have a 'target' column (created by clean()).
    """
    out_path = _ensure_dir(out_path)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    df["num"].value_counts().sort_index().plot(
        kind="bar", ax=axes[0], color=sns.color_palette("deep")
    )
    axes[0].set_title("Original target (num) — 5 classes")
    axes[0].set_xlabel("Disease severity (0 = none)")
    axes[0].set_ylabel("Patients")
    axes[0].tick_params(axis="x", rotation=0)

    df["target"].value_counts().sort_index().plot(
        kind="bar", ax=axes[1], color=["#2ecc71", "#e74c3c"]
    )
    axes[1].set_title("Binarized target — 2 classes")
    axes[1].set_xlabel("0 = No disease,  1 = Disease")
    axes[1].set_xticklabels(["No disease", "Disease"], rotation=0)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_feature_distributions(
    df: pd.DataFrame,
    numeric_cols: list[str],
    out_path: str | Path,
) -> None:
    """Grid of histograms for numeric features, one subplot per column."""
    out_path = _ensure_dir(out_path)

    n = len(numeric_cols)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 3.5 * nrows))
    for ax, col in zip(axes.flat, numeric_cols):
        sns.histplot(x=df[col].dropna(), kde=True, ax=ax, color="#3498db")
        ax.set_title(col)
    for ax in axes.flat[n:]:
        ax.set_visible(False)

    plt.suptitle("Numeric feature distributions", y=1.02, fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_correlation_heatmap(df: pd.DataFrame, out_path: str | Path) -> None:
    """Correlation heatmap for numeric columns only.

    Drops `id` (not a feature) and `num` (raw target) before computing.
    """
    out_path = _ensure_dir(out_path)

    corr_df = df.select_dtypes(include=[np.number]).drop(
        columns=[c for c in ["id", "num"] if c in df.columns]
    )

    plt.figure(figsize=(9, 7))
    sns.heatmap(
        corr_df.corr(),
        annot=True, fmt=".2f", cmap="coolwarm", center=0,
        square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
    )
    plt.title("Correlation heatmap (numeric features)")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_confusion_matrices(results: dict, y_test, out_path: str | Path) -> None:
    """Side-by-side confusion matrices for all trained models.

    `results` is the dict returned by model_training.train_and_evaluate().
    """
    out_path = _ensure_dir(out_path)

    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4.7 * n, 4.2))
    if n == 1:
        axes = [axes]

    for ax, (name, r) in zip(axes, results.items()):
        cm = confusion_matrix(y_test, r.y_pred)
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False,
            xticklabels=["No disease", "Disease"],
            yticklabels=["No disease", "Disease"],
        )
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.suptitle("Confusion Matrices", y=1.03, fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curves(results: dict, y_test, out_path: str | Path) -> None:
    """Overlaid ROC curves for all trained models."""
    out_path = _ensure_dir(out_path)

    plt.figure(figsize=(7, 6))
    for name, r in results.items():
        fpr, tpr, _ = roc_curve(y_test, r.y_proba)
        auc = roc_auc_score(y_test, r.y_proba)
        plt.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC = {auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_feature_importance(
    model,
    feature_names: list[str],
    out_path: str | Path,
    top_n: int = 15,
) -> None:
    out_path = _ensure_dir(out_path)

    if not hasattr(model, "feature_importances_"):
        raise ValueError(
            f"{type(model).__name__} has no feature_importances_. "
            "Pass a tree-based model like RandomForest or GradientBoosting."
        )

    importances = pd.Series(model.feature_importances_, index=feature_names)
    top = importances.sort_values(ascending=True).tail(top_n)

    plt.figure(figsize=(8, 6))
    top.plot(kind="barh", color="#16a085")
    plt.title(f"Top {top_n} Feature Importances")
    plt.xlabel("Importance (mean decrease in impurity)")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_model_comparison(results_df: pd.DataFrame, out_path: str | Path) -> None:
    """Grouped bar chart comparing all metrics across all models.

    Expects the DataFrame returned by model_training.results_to_dataframe().
    """
    out_path = _ensure_dir(out_path)

    ax = results_df.plot(
        kind="bar", figsize=(11, 5.5),
        colormap="viridis", edgecolor="black", width=0.8,
    )
    ax.set_title("Model Performance Comparison")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_xticklabels(results_df.index, rotation=0)
    ax.legend(loc="lower right", ncol=len(results_df.columns))
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)

    for container in ax.containers:
        ax.bar_label(cast(BarContainer, container), fmt="%.2f", fontsize=8, padding=2)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()