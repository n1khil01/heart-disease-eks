from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .data_preprocessing import Preprocessor, prepare_data


def build_models() -> dict:
    """Return a fresh dict of untrained models.

    Defined as a function (not a module-level constant) so each call gets
    fresh, unfitted instances — avoids accidentally reusing a trained model
    across calls.
    """
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, random_state=42),
    }


@dataclass
class ModelResult:
    """Everything we want to know about a trained model in one place."""
    name: str
    model: object  
    y_pred: np.ndarray
    y_proba: np.ndarray
    metrics: dict[str, float]


def evaluate(y_true, y_pred, y_proba) -> dict[str, float]:
    return {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall":    float(recall_score(y_true, y_pred)),
        "f1":        float(f1_score(y_true, y_pred)),
        "roc_auc":   float(roc_auc_score(y_true, y_proba)),
    }


def train_and_evaluate(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, ModelResult]:
    """Train all models and return a dict of results keyed by model name."""
    results: dict[str, ModelResult] = {}
    for name, model in build_models().items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics = evaluate(y_test, y_pred, y_proba)
        results[name] = ModelResult(name, model, y_pred, y_proba, metrics)
        print(f"✓ {name:22s}  acc={metrics['accuracy']:.3f}  "
              f"recall={metrics['recall']:.3f}  auc={metrics['roc_auc']:.3f}")
    return results


def results_to_dataframe(results: dict[str, ModelResult]) -> pd.DataFrame:
    """Flatten the results dict into a tidy comparison table."""
    rows = []
    for name, r in results.items():
        row = {"Model": name, **r.metrics}
        rows.append(row)
    return pd.DataFrame(rows).set_index("Model").round(4)


def select_best(results: dict[str, ModelResult], metric: str = "roc_auc") -> ModelResult:
    """Pick the best model by a chosen metric.

    Default is ROC-AUC because it's threshold-independent — a better
    measure of ranking quality than accuracy. Change to 'recall' if you
    want to prioritize catching sick patients over everything else.
    """
    return max(results.values(), key=lambda r: r.metrics[metric])


def save_bundle(
    model: object,
    preprocessor: Preprocessor,
    path: str | Path,
) -> None:
    """Pickle the model + preprocessor together as a single artifact.

    At inference time (FastAPI endpoint), you load this one file and you
    have everything you need. Keeping them together prevents the classic
    bug where the preprocessor version doesn't match the model version.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"model": model, "preprocessor": preprocessor}, f)
    print(f"Saved bundle to {path}")


def load_bundle(path: str | Path) -> tuple[object, Preprocessor]:
    """Inverse of save_bundle. Used by the inference code."""
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    return bundle["model"], bundle["preprocessor"]


def main(csv_path: str = "data/heart_disease_uci.csv",
         model_out: str = "outputs/model_bundle.pkl") -> None:
    """CLI entry point: prepare data, train, evaluate, save best model."""
    print("Preparing data...")
    X_train, X_test, y_train, y_test, preprocessor = prepare_data(csv_path)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    print("\nTraining models...")
    results = train_and_evaluate(X_train, y_train, X_test, y_test)

    print("\nResults table:")
    print(results_to_dataframe(results))

    best = select_best(results)
    print(f"\n Best model: {best.name} (roc_auc={best.metrics['roc_auc']:.4f})")

    save_bundle(best.model, preprocessor, model_out)


if __name__ == "__main__":
    main()