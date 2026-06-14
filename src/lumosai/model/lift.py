from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from lumosai.exceptions import LumosValidationError
from lumosai.model.scores import ClassificationScores, safe_label


def _decile_table(events: np.ndarray, probabilities: np.ndarray) -> list[dict[str, Any]]:
    order = np.argsort(-probabilities, kind="mergesort")
    sorted_events = events[order]
    baseline = float(events.mean())
    rows: list[dict[str, Any]] = []

    for decile, indices in enumerate(np.array_split(np.arange(len(sorted_events)), 10), start=1):
        decile_events = sorted_events[indices]
        if len(decile_events) == 0:
            event_count = 0
            event_rate = None
            lift = None
        else:
            event_count = int(decile_events.sum())
            event_rate = float(decile_events.mean())
            lift = float(event_rate / baseline) if baseline > 0 else float("nan")
        rows.append(
            {
                "decile": decile,
                "rows": int(len(decile_events)),
                "event_count": event_count,
                "event_rate": event_rate,
                "baseline_event_rate": baseline,
                "lift": lift,
            }
        )

    return rows


def _class_key(label: Any, scores: ClassificationScores) -> str:
    return "positive" if scores.positive_label is not None else safe_label(label)


def _validate_class_keys(labels: list[Any], scores: ClassificationScores) -> None:
    seen: dict[str, Any] = {}
    for label in labels:
        class_key = _class_key(label, scores)
        if class_key in seen:
            msg = (
                "safe_label metric key collision for class labels "
                f"{seen[class_key]!r} and {label!r}: both resolve to {class_key!r}"
            )
            raise LumosValidationError(msg)
        seen[class_key] = label


def lift_metrics(
    y_true: pd.Series,
    scores: ClassificationScores,
) -> tuple[dict[str, float], dict[str, Any]]:
    if len(y_true) != scores.values.shape[0]:
        msg = (
            "y_true row count must match scores.values row count for lift metrics: "
            f"y_true={len(y_true)}, scores={scores.values.shape[0]}"
        )
        raise LumosValidationError(msg)

    metrics: dict[str, float] = {}
    summary: dict[str, Any] = {"classes": {}, "warnings": []}

    labels_to_score = (
        [scores.positive_label] if scores.positive_label is not None else scores.labels
    )
    _validate_class_keys(labels_to_score, scores)
    for label in labels_to_score:
        class_index = scores.label_index(label)
        events = (y_true == label).to_numpy(dtype=int)
        if events.sum() == 0:
            summary["warnings"].append(
                f"Skipping lift for class {label!r} because it has no positives."
            )
            continue

        class_key = _class_key(label, scores)
        table = _decile_table(events, scores.values[:, class_index])
        summary["classes"][class_key] = table
        for row in table:
            if row["lift"] is not None:
                metrics[f"lift/{class_key}/decile_{row['decile']}"] = float(row["lift"])
        top_decile = next(row for row in table if row["rows"] > 0)
        metrics[f"lift/{class_key}/top_decile"] = float(top_decile["lift"])

    return metrics, summary
