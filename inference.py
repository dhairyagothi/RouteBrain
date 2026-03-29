from __future__ import annotations

import json
import os
import re
from typing import Any

import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

ROUTEBRAIN_URL = os.getenv("ROUTEBRAIN_URL", "http://localhost:7860")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
MAX_STEPS = int(os.getenv("MAX_STEPS", "20"))


def extract_json_object(text: str) -> dict[str, Any] | None:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def estimate_eta(route: dict[str, Any], order: dict[str, Any]) -> float:
    traffic_level = float(order.get("traffic_level", 0.0))
    weather_index = max(0.0, min(2.0, float(order.get("weather_penalty", 0.0)) / 5.0))
    rid = int(route.get("route_id", 0)) % 3
    if rid == 0:
        traffic_weight, weather_weight = 1.00, 0.15
    elif rid == 1:
        traffic_weight, weather_weight = 0.55, 0.45
    else:
        traffic_weight, weather_weight = 0.75, 0.25

    rush_multiplier = 1.15 if (int(order.get("order_id", 0)) % 6) in (2, 3) else 1.0
    external_multiplier = 1.0 + (traffic_level * traffic_weight * rush_multiplier) + (weather_index * weather_weight * 0.6)
    return (
        float(route["distance_km"]) * 3.0 * float(route["traffic_factor"]) * external_multiplier
        + float(order["restaurant_wait"])
        + float(order["weather_penalty"])
    )


def _pickup_eta(vehicle_location: list[int], pickup: list[int]) -> float:
    return (abs(int(vehicle_location[0]) - int(pickup[0])) + abs(int(vehicle_location[1]) - int(pickup[1]))) * 1.2


def fallback_action(observation: dict[str, Any]) -> dict[str, int]:
    current = observation.get("current_order")
    if not current:
        return {"route_choice": 0, "vehicle_id": 0}
    routes = current.get("routes", [])
    if not routes:
        return {"route_choice": 0, "vehicle_id": 0}

    deadline = float(current.get("deadline_minutes", 0.0))
    traffic_level = float(current.get("traffic_level", 0.0))
    pickup = list(current.get("pickup", [0, 0]))
    fleet = observation.get("fleet", [])

    if not fleet:
        best_route = min(routes, key=lambda route: estimate_eta(route, current))
        return {"route_choice": int(best_route["route_id"]), "vehicle_id": 0}

    best_vehicle_id = 0
    best_route_id = int(routes[0]["route_id"])
    best_score = float("inf")

    for vehicle in fleet:
        vehicle_id = int(vehicle.get("vehicle_id", 0))
        vehicle_location = list(vehicle.get("location", [0, 0]))
        available_in = float(vehicle.get("available_in_minutes", 0.0))
        pickup_eta = _pickup_eta(vehicle_location, pickup)

        for route in routes:
            travel_eta = estimate_eta(route, current)
            eta = available_in + pickup_eta + travel_eta
            delay = max(0.0, eta - deadline)
            cost = float(route["distance_km"]) * (1.0 + traffic_level)
            score = (delay * 8.0) + eta + (0.2 * cost)

            if score < best_score:
                best_score = score
                best_vehicle_id = vehicle_id
                best_route_id = int(route["route_id"])

    return {"route_choice": best_route_id, "vehicle_id": best_vehicle_id}


def llm_action(client: OpenAI, observation: dict[str, Any]) -> dict[str, int]:
    system_prompt = (
        "You are a dispatch planner. Return JSON only with keys route_choice and vehicle_id. "
        "Choose the pair that minimizes ETA and delays considering traffic, weather, and driver availability."
    )
    user_prompt = json.dumps(observation)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    content = response.choices[0].message.content or "{}"
    parsed = extract_json_object(content)
    if not parsed or "route_choice" not in parsed:
        raise ValueError("Model did not return route_choice")
    vehicle_id = int(parsed.get("vehicle_id", 0))
    return {"route_choice": int(parsed["route_choice"]), "vehicle_id": vehicle_id}


def call_reset(task_id: str) -> dict[str, Any]:
    response = requests.post(
        f"{ROUTEBRAIN_URL}/reset",
        json={"task_id": task_id},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def call_step(session_id: str, action: dict[str, int]) -> dict[str, Any]:
    response = requests.post(
        f"{ROUTEBRAIN_URL}/step",
        json={"session_id": session_id, "action": action},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def call_grader(session_id: str) -> dict[str, Any]:
    response = requests.get(
        f"{ROUTEBRAIN_URL}/grader",
        params={"session_id": session_id},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def run_task(task_id: str, client: OpenAI | None) -> float:
    reset_payload = call_reset(task_id)
    session_id = reset_payload["session_id"]
    observation = reset_payload["observation"]

    for _ in range(MAX_STEPS):
        if observation.get("current_order") is None:
            break

        action_payload: dict[str, int]
        if client is None:
            action_payload = fallback_action(observation)
        else:
            try:
                action_payload = llm_action(client, observation)
            except Exception:
                action_payload = fallback_action(observation)

        step_payload = call_step(session_id, action_payload)
        observation = step_payload["observation"]
        if bool(step_payload["done"]):
            break

    grade = call_grader(session_id)
    score = float(grade["score"])
    print(f"task={task_id} score={score:.4f}")
    return score


def main() -> None:
    client: OpenAI | None = None
    if GROQ_API_KEY:
        client = OpenAI(api_key=GROQ_API_KEY, base_url=API_BASE_URL)

    tasks = ["easy", "medium", "hard"]
    scores: list[float] = []
    for task_id in tasks:
        scores.append(run_task(task_id, client))

    avg = sum(scores) / max(1, len(scores))
    print(f"average_score={avg:.4f}")


if __name__ == "__main__":
    main()
