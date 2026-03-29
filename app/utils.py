from __future__ import annotations

from typing import Any


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def manhattan_distance(a: list[int], b: list[int]) -> float:
    return float(abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1])))


def _route_sensitivity(route_id: int) -> tuple[float, float]:
    bucket = int(route_id) % 3
    if bucket == 0:
        return 1.00, 0.15
    if bucket == 1:
        return 0.55, 0.45
    return 0.75, 0.25


def infer_external_factors(order: dict[str, Any], time_step: int) -> dict[str, Any]:
    traffic_index = clamp(float(order.get("traffic_level", 0.0)), 0.0, 2.0)
    weather_index = clamp(float(order.get("weather_penalty", 0.0)) / 5.0, 0.0, 2.0)
    rush_hour = (int(time_step) % 6) in (2, 3)
    return {
        "traffic_index": traffic_index,
        "weather_index": weather_index,
        "rush_hour": rush_hour,
    }


def route_eta_minutes(
    distance_km: float,
    traffic_factor: float,
    restaurant_wait: float,
    weather_penalty: float,
    traffic_level: float = 0.0,
    route_id: int = 0,
    time_step: int = 0,
) -> float:
    # Calibrated travel conversion keeps tasks solvable while preserving difficulty tiers.
    base_travel = distance_km * 3.0
    traffic_weight, weather_weight = _route_sensitivity(route_id)
    rush_multiplier = 1.15 if (int(time_step) % 6) in (2, 3) else 1.0
    weather_index = clamp(weather_penalty / 5.0, 0.0, 2.0)
    external_multiplier = 1.0 + (traffic_level * traffic_weight * rush_multiplier) + (weather_index * weather_weight * 0.6)
    return (base_travel * traffic_factor * external_multiplier) + restaurant_wait + weather_penalty


def select_best_route(order: dict[str, Any], vehicle_location: list[int], time_step: int = 0) -> tuple[int, float]:
    best_id = -1
    best_eta = float("inf")
    best_score = float("inf")
    deadline = float(order.get("deadline_minutes", 0.0))
    traffic_level = float(order.get("traffic_level", 0.0))
    pickup = list(order.get("pickup", [0, 0]))
    pickup_eta = manhattan_distance(vehicle_location, pickup) * 1.2

    for route in order["routes"]:
        travel_eta = route_eta_minutes(
            distance_km=float(route["distance_km"]),
            traffic_factor=float(route["traffic_factor"]),
            restaurant_wait=float(order["restaurant_wait"]),
            weather_penalty=float(order["weather_penalty"]),
            traffic_level=traffic_level,
            route_id=int(route["route_id"]),
            time_step=time_step,
        )
        eta = pickup_eta + travel_eta
        delay = max(0.0, eta - deadline)
        route_cost = float(route["distance_km"]) * (1.0 + traffic_level)
        # Missing deadlines is expensive; among similar delays prefer lower ETA and cost.
        score = (delay * 8.0) + eta + (0.2 * route_cost)

        if score < best_score:
            best_score = score
            best_eta = eta
            best_id = int(route["route_id"])
    return best_id, best_eta
