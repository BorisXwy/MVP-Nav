"""Standard Habitat navigation metric helpers."""

from typing import Any, Dict, Iterable


METRIC_KEYS = ("success", "spl", "soft_spl", "distance_to_goal")


def coerce_nav_metrics(info: Dict[str, Any]) -> Dict[str, float]:
    metrics = {}
    for key in METRIC_KEYS:
        value = info.get(key, 0.0)
        try:
            metrics[key] = float(value)
        except (TypeError, ValueError):
            metrics[key] = 0.0
    return metrics


def zero_nav_metrics() -> Dict[str, float]:
    return {key: 0.0 for key in METRIC_KEYS}


def summarize_nav_metrics(rows: Iterable[Dict[str, float]]) -> Dict[str, float]:
    rows = list(rows)
    if not rows:
        return zero_nav_metrics()
    total = float(len(rows))
    return {
        "sr": sum(row.get("success", 0.0) for row in rows) / total,
        "spl": sum(row.get("spl", 0.0) for row in rows) / total,
        "soft_spl": sum(row.get("soft_spl", 0.0) for row in rows) / total,
        "distance_to_goal": sum(row.get("distance_to_goal", 0.0) for row in rows) / total,
    }
