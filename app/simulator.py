from __future__ import annotations

from typing import Any

from app.utils import manhattan_distance, route_eta_minutes, select_best_route


def evaluate_route_choice(
    order: dict[str, Any],
    route_choice: int,
    cost_per_km: float,
    vehicle_location: list[int],
    time_step: int,
) -> dict[str, Any]:
    route_map = {int(r["route_id"]): r for r in order["routes"]}
    invalid_action = route_choice not in route_map

    if invalid_action:
        selected_route = max(order["routes"], key=lambda r: float(r["distance_km"]))
        selected_id = int(selected_route["route_id"])
    else:
        selected_route = route_map[route_choice]
        selected_id = route_choice

    traffic_level = float(order.get("traffic_level", 0.0))
    pickup_eta = manhattan_distance(vehicle_location, list(order["pickup"])) * 1.2
    travel_eta = route_eta_minutes(
        distance_km=float(selected_route["distance_km"]),
        traffic_factor=float(selected_route["traffic_factor"]),
        restaurant_wait=float(order["restaurant_wait"]),
        weather_penalty=float(order["weather_penalty"]),
        traffic_level=traffic_level,
        route_id=selected_id,
        time_step=time_step,
    )
    eta_minutes = pickup_eta + travel_eta

    deadline = float(order["deadline_minutes"])
    delay = max(0.0, eta_minutes - deadline)
    on_time = delay <= 0.0

    weather_index = min(2.0, max(0.0, float(order.get("weather_penalty", 0.0)) / 5.0))
    cost = float(selected_route["distance_km"]) * cost_per_km * (1.0 + traffic_level + (0.2 * weather_index))

    best_route_id, best_eta = select_best_route(order, vehicle_location=vehicle_location, time_step=time_step)

    return {
        "selected_route_id": selected_id,
        "invalid_action": invalid_action,
        "pickup_eta_minutes": pickup_eta,
        "travel_eta_minutes": travel_eta,
        "eta_minutes": eta_minutes,
        "delay_minutes": delay,
        "on_time": on_time,
        "cost": cost,
        "best_route_id": best_route_id,
        "best_eta_minutes": best_eta,
    }
