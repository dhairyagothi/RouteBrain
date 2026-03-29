from __future__ import annotations

from app.utils import clamp


def compute_score(state: dict) -> dict:
    completed = max(1, int(state["metrics"]["completed_orders"]))
    on_time = int(state["metrics"]["on_time_deliveries"])

    success_rate = on_time / completed
    delay_norm = clamp(
        float(state["metrics"]["total_delay"]) / (float(state["max_delay_per_delivery"]) * completed),
        0.0,
        1.0,
    )
    cost_norm = clamp(
        float(state["metrics"]["total_cost"]) / (float(state["max_cost_per_delivery"]) * completed),
        0.0,
        1.0,
    )

    score = (0.5 * success_rate) + (0.3 * (1.0 - delay_norm)) + (0.2 * (1.0 - cost_norm))
    score = clamp(score, 0.0, 1.0)

    return {
        "task_id": state["task_id"],
        "score": score,
        "components": {
            "success_rate": success_rate,
            "delay_norm": delay_norm,
            "cost_norm": cost_norm,
        },
        "metrics": state["metrics"],
    }
