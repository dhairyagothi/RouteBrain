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
HF_TOKEN = os.getenv("HF_TOKEN", "")
API_KEY = HF_TOKEN or os.getenv("API_KEY", "")
BENCHMARK = os.getenv("ROUTEBRAIN_BENCHMARK", "routebrain")
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
    phase = (int(order.get("order_id", 0)) + int(route.get("route_id", 0))) % 5
    temporal_multiplier = 1.0
    if rid == 0 and phase in (1, 2):
        temporal_multiplier = 1.10 + (0.08 * traffic_level)
    elif rid == 1 and phase in (0, 4):
        temporal_multiplier = 1.05 + (0.10 * weather_index)
    elif rid == 2 and phase == 3:
        temporal_multiplier = 1.08 + (0.05 * (traffic_level + weather_index))

    external_multiplier = 1.0 + (traffic_level * traffic_weight * rush_multiplier) + (weather_index * weather_weight * 0.6)
    external_multiplier *= temporal_multiplier
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
    print(f"[START] task={task_id} env={BENCHMARK} model={MODEL_NAME}")

    steps = 0
    rewards: list[float] = []
    score = 0.0
    success = False
    episode_error: str | None = None

    try:
        reset_payload = call_reset(task_id)
        session_id = reset_payload["session_id"]
        observation = reset_payload["observation"]

        for step_number in range(1, MAX_STEPS + 1):
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
            reward = float(step_payload["reward"])
            done = bool(step_payload["done"])
            info = step_payload.get("info", {})
            last_action_error = info.get("last_action_error")
            error_str = "null" if last_action_error is None else str(last_action_error)
            action_str = json.dumps(action_payload, separators=(",", ":"))

            print(
                f"[STEP] step={step_number} action={action_str} reward={reward:.2f} "
                f"done={'true' if done else 'false'} error={error_str}"
            )

            rewards.append(reward)
            steps = step_number
            observation = step_payload["observation"]
            if done:
                break

        grade = call_grader(session_id)
        score = float(grade["score"])
        success = 0.0 <= score <= 1.0
        return score
    except Exception as exc:
        episode_error = str(exc)
        success = False
        return score
    finally:
        rewards_str = ",".join(f"{r:.2f}" for r in rewards)
        print(
            f"[END] success={'true' if success else 'false'} steps={steps} "
            f"score={score:.2f} rewards={rewards_str}"
        )


def main() -> None:
    client: OpenAI | None = None
    if API_KEY:
        client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

    tasks = ["easy", "medium", "hard"]
    for task_id in tasks:
        run_task(task_id, client)


if __name__ == "__main__":
    main()
