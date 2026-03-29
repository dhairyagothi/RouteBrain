from __future__ import annotations

from app.utils import clamp


def compute_step_reward(
    *,
    eta_minutes: float,
    deadline_minutes: float,
    delay_minutes: float,
    cost_value: float,
    max_cost_per_delivery: float,
    invalid_action: bool,
) -> tuple[float, dict[str, float]]:
    speed_component = clamp(1.0 - (eta_minutes / max(1.0, deadline_minutes)), -1.0, 1.0)
    delay_penalty = delay_minutes / max(1.0, deadline_minutes)
    cost_penalty = cost_value / max(1.0, max_cost_per_delivery)
    on_time_bonus = 0.4 if delay_minutes <= 0.0 else 0.0
    invalid_penalty = 0.5 if invalid_action else 0.0

    reward = (0.6 * speed_component) - (0.3 * delay_penalty) - (0.1 * cost_penalty) + on_time_bonus - invalid_penalty

    parts = {
        "speed_component": speed_component,
        "delay_penalty": delay_penalty,
        "cost_penalty": cost_penalty,
        "on_time_bonus": on_time_bonus,
        "invalid_penalty": invalid_penalty,
    }
    return reward, parts
