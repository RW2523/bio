"""Classification metrics using scikit-learn."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: list[int]) -> dict[str, Any]:
    y_true_i = y_true.astype(np.int64, copy=False)
    y_pred_i = y_pred.astype(np.int64, copy=False)

    macro_f1 = float(f1_score(y_true_i, y_pred_i, average="macro", labels=labels, zero_division=0))
    weighted_f1 = float(f1_score(y_true_i, y_pred_i, average="weighted", labels=labels, zero_division=0))
    acc = float(accuracy_score(y_true_i, y_pred_i))
    kappa = float(cohen_kappa_score(y_true_i, y_pred_i))
    cm = confusion_matrix(y_true_i, y_pred_i, labels=labels)
    report = classification_report(
        y_true_i,
        y_pred_i,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    return {
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "accuracy": acc,
        "cohen_kappa": kappa,
        "confusion_matrix": cm.tolist(),
        "per_class_report": report,
    }
