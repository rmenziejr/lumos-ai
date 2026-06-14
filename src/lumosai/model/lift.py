from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from lumosai.model.scores import ClassificationScores, safe_label


def _decile_table(events: np.ndarray, probabilities: np.ndarray) -> list[dict[str, Any]]:
    order = np.argsort(-probabilities, kind="mergesort")
    sorted_events = events[order]
    baseline = float(events.mean())
    rows: list[dict[str, Any]] = []

    for decile, indices in enumerate(np.array_split(np.arange(len(sorted_events)), 10), start=1):
        decile_events = sorted_events[indices]
        event_count = int(decile_events.sum())
        event_rate = float(decile_events.mean()) if len(decile_events) else 0.0
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


def lift_metrics(
    y_true: pd.Series,
    scores: ClassificationScores,
) -> tuple[dict[str, float], dict[str, Any]]:
    metrics: dict[str, float] = {}
    summary: dict[str, Any] = {"classes": {}, "warnings": []}

    labels_to_score = (
        [scores.positive_label] if scores.positive_label is not None else scores.labels
    )
    for label in labels_to_score:
        class_index = scores.label_index(label)
        events = (y_true == label).to_numpy(dtype=int)
        if events.sum() == 0:
            summary["warnings"].append(
                f"Skipping lift for class {label!r} because it has no positives."
            )
            continue

        class_key = "positive" if scores.positive_label is not None else safe_label(label)
        table = _decile_table(events, scores.values[:, class_index])
        summary["classes"][class_key] = table
        for row in table:
            metrics[f"lift/{class_key}/decile_{row['decile']}"] = float(row["lift"])
        metrics[f"lift/{class_key}/top_decile"] = float(table[0]["lift"])

    return metrics, summary
