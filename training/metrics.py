from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import (
        average_precision_score,
        confusion_matrix,
        precision_recall_curve,
        precision_recall_fscore_support,
        roc_auc_score,
    )
except ImportError as exc:  # pragma: no cover - dependency checked on training host.
    raise ImportError("scikit-learn is required for evaluation.") from exc


def _finite_metric(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _sample_pr_curve(
    labels: np.ndarray,
    scores: np.ndarray,
    max_points: int = 1001,
) -> dict[str, list[float]]:
    if np.unique(labels).size < 2:
        return {"precision": [], "recall": [], "thresholds": []}
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    if len(precision) > max_points:
        selected = np.unique(
            np.linspace(0, len(precision) - 1, max_points, dtype=np.int64)
        )
        precision = precision[selected]
        recall = recall[selected]
        threshold_indices = selected[selected < len(thresholds)]
        sampled_thresholds = thresholds[threshold_indices]
    else:
        sampled_thresholds = thresholds
    return {
        "precision": precision.astype(float).tolist(),
        "recall": recall.astype(float).tolist(),
        "thresholds": sampled_thresholds.astype(float).tolist(),
    }


def window_metrics(
    labels: np.ndarray | pd.Series,
    scores: np.ndarray | pd.Series,
    threshold: float,
) -> dict[str, Any]:
    y_true = np.asarray(labels, dtype=np.int64)
    y_score = np.asarray(scores, dtype=np.float64)
    if y_true.shape != y_score.shape:
        raise ValueError("labels and scores must have the same shape.")
    if y_true.size == 0:
        return {
            "count": 0,
            "positive_count": 0,
            "negative_count": 0,
            "pr_auc": None,
            "roc_auc": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "false_positive_rate": None,
            "confusion_matrix": [[0, 0], [0, 0]],
            "pr_curve": {"precision": [], "recall": [], "thresholds": []},
        }

    predictions = (y_score >= float(threshold)).astype(np.int64)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        predictions,
        average="binary",
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    false_positive_rate = fp / (fp + tn) if fp + tn else np.nan
    if np.unique(y_true).size == 2:
        pr_auc = average_precision_score(y_true, y_score)
        roc_auc = roc_auc_score(y_true, y_score)
    else:
        pr_auc = np.nan
        roc_auc = np.nan
    return {
        "count": int(y_true.size),
        "positive_count": int(y_true.sum()),
        "negative_count": int((y_true == 0).sum()),
        "threshold": float(threshold),
        "pr_auc": _finite_metric(pr_auc),
        "roc_auc": _finite_metric(roc_auc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_positive_rate": _finite_metric(false_positive_rate),
        "confusion_matrix": [
            [int(tn), int(fp)],
            [int(fn), int(tp)],
        ],
        "pr_curve": _sample_pr_curve(y_true, y_score),
    }


def grouped_window_metrics(
    predictions: pd.DataFrame,
    group_column: str,
    threshold: float,
) -> dict[str, dict[str, Any]]:
    if group_column not in predictions:
        raise ValueError(f"Missing group column: {group_column}")
    result: dict[str, dict[str, Any]] = {}
    for group, frame in predictions.groupby(group_column, sort=True):
        result[str(group)] = window_metrics(
            frame["label"],
            frame["score"],
            threshold,
        )
    return result


def full_window_report(
    predictions: pd.DataFrame,
    threshold: float,
) -> dict[str, Any]:
    frame = predictions.copy()
    frame["year"] = pd.to_datetime(frame["center_time"]).dt.year
    return {
        "overall": window_metrics(frame["label"], frame["score"], threshold),
        "by_station": grouped_window_metrics(frame, "station", threshold),
        "by_year": grouped_window_metrics(frame, "year", threshold),
    }


def summarize_fold_metrics(
    fold_reports: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    keys = ("pr_auc", "roc_auc", "precision", "recall", "f1", "false_positive_rate")
    summary: dict[str, Any] = {"folds": fold_reports}
    for key in keys:
        values = [
            report["window"]["overall"].get(key)
            for report in fold_reports.values()
        ]
        finite = np.asarray([value for value in values if value is not None], dtype=float)
        summary[key] = {
            "mean": float(finite.mean()) if finite.size else None,
            "std": float(finite.std(ddof=0)) if finite.size else None,
        }
    event_values = [
        report["event"].get("event_f1")
        for report in fold_reports.values()
        if report["event"].get("event_f1") is not None
    ]
    summary["event_f1"] = {
        "mean": float(np.mean(event_values)) if event_values else None,
        "std": float(np.std(event_values)) if event_values else None,
    }
    return summary
