from __future__ import annotations

from copy import deepcopy
from typing import Any

from app.grader import compute_score
from app.reward import compute_step_reward
from app.schemas import Action, Metrics, Observation
from app.simulator import evaluate_route_choice
from app.tasks import task_store
from app.utils import infer_external_factors, manhattan_distance


class RouteEnv:
    def __init__(self) -> None:
        self._state: dict[str, Any] | None = None

    def reset(self, task_id: str) -> tuple[Observation, dict[str, Any]]:
        task = deepcopy(task_store.get_task(task_id))
        orders = list(task["orders"])
        fleet = self._build_fleet(task)

        self._state = {
            "task_id": task["task_id"],
            "time_step": 0,
            "max_steps": int(task["max_steps"]),
            "orders": orders,
            "cost_per_km": float(task["cost_per_km"]),
            "max_delay_per_delivery": float(task["max_delay_per_delivery"]),
            "max_cost_per_delivery": float(task["max_cost_per_delivery"]),
            "driver_location": list(fleet[0]["location"]) if fleet else list(orders[0]["driver_location"]),
            "fleet": fleet,
            "sim_clock_minutes": 0.0,
            "done": False,
            "metrics": {
                "completed_orders": 0,
                "on_time_deliveries": 0,
                "total_delay": 0.0,
                "total_cost": 0.0,
            },
            "history": [],
        }

        return self._build_observation(), {"message": "Environment reset.", "task_id": task_id}

    def step(self, action: Action) -> tuple[Observation, float, bool, dict[str, Any]]:
        if self._state is None:
            raise RuntimeError("Environment not initialized. Call reset first.")
        if self._state["done"]:
            raise RuntimeError("Episode already finished. Call reset for a new run.")

        idx = int(self._state["time_step"])
        order = self._state["orders"][idx]
        sim_clock = float(self._state["sim_clock_minutes"])

        selected_vehicle = self._resolve_vehicle(order=order, action=action)
        selected_vehicle_id = int(selected_vehicle["vehicle_id"])
        queue_wait = max(0.0, float(selected_vehicle["available_at_minutes"]) - sim_clock)

        result = evaluate_route_choice(
            order=order,
            route_choice=int(action.route_choice),
            cost_per_km=float(self._state["cost_per_km"]),
            vehicle_location=list(selected_vehicle["location"]),
            time_step=idx,
        )

        effective_eta = float(result["eta_minutes"]) + queue_wait
        effective_delay = max(0.0, effective_eta - float(order["deadline_minutes"]))
        on_time = effective_delay <= 0.0

        reward, reward_parts = compute_step_reward(
            eta_minutes=effective_eta,
            deadline_minutes=float(order["deadline_minutes"]),
            delay_minutes=effective_delay,
            cost_value=float(result["cost"]),
            max_cost_per_delivery=float(self._state["max_cost_per_delivery"]),
            invalid_action=bool(result["invalid_action"]),
        )

        self._state["metrics"]["completed_orders"] += 1
        self._state["metrics"]["total_delay"] += float(effective_delay)
        self._state["metrics"]["total_cost"] += float(result["cost"])
        if on_time:
            self._state["metrics"]["on_time_deliveries"] += 1

        dispatch_start = max(sim_clock, float(selected_vehicle["available_at_minutes"]))
        selected_vehicle["available_at_minutes"] = dispatch_start + float(result["eta_minutes"])
        selected_vehicle["location"] = list(order["drop"])
        selected_vehicle["deliveries_completed"] = int(selected_vehicle["deliveries_completed"]) + 1

        self._state["sim_clock_minutes"] = sim_clock + 4.0
        self._state["driver_location"] = list(selected_vehicle["location"])
        self._state["time_step"] += 1
        self._state["done"] = self._state["time_step"] >= self._state["max_steps"]

        step_info = {
            "order_id": int(order["order_id"]),
            "vehicle_id": selected_vehicle_id,
            "selected_route_id": int(result["selected_route_id"]),
            "best_route_id": int(result["best_route_id"]),
            "pickup_eta_minutes": float(result["pickup_eta_minutes"]),
            "travel_eta_minutes": float(result["travel_eta_minutes"]),
            "queue_wait_minutes": float(queue_wait),
            "eta_minutes": float(effective_eta),
            "best_eta_minutes": float(result["best_eta_minutes"]),
            "delay_minutes": float(effective_delay),
            "invalid_action": bool(result["invalid_action"]),
            "reward_parts": reward_parts,
            "ignored_fields": {
                "assignments": len(action.assignments),
                "route_updates": len(action.route_updates),
            },
        }
        self._state["history"].append(step_info)

        return self._build_observation(), float(reward), bool(self._state["done"]), step_info

    def state(self) -> dict[str, Any]:
        if self._state is None:
            raise RuntimeError("Environment not initialized. Call reset first.")
        return deepcopy(self._state)

    def grade(self) -> dict[str, Any]:
        if self._state is None:
            raise RuntimeError("Environment not initialized. Call reset first.")
        return compute_score(self._state)

    def _build_observation(self) -> Observation:
        assert self._state is not None

        time_step = int(self._state["time_step"])
        current_order: dict[str, Any] | None
        if time_step >= int(self._state["max_steps"]):
            current_order = None
        else:
            current_order = self._state["orders"][time_step]

        metrics = Metrics(**self._state["metrics"])
        external = {"traffic_index": 0.0, "weather_index": 0.0, "rush_hour": False}
        if current_order is not None:
            external = infer_external_factors(current_order, time_step)

        sim_clock = float(self._state.get("sim_clock_minutes", 0.0))
        fleet_obs = []
        for vehicle in self._state.get("fleet", []):
            fleet_obs.append(
                {
                    "vehicle_id": int(vehicle["vehicle_id"]),
                    "location": list(vehicle["location"]),
                    "available_in_minutes": max(0.0, float(vehicle["available_at_minutes"]) - sim_clock),
                    "deliveries_completed": int(vehicle["deliveries_completed"]),
                }
            )

        return Observation(
            task_id=str(self._state["task_id"]),
            time_step=time_step,
            max_steps=int(self._state["max_steps"]),
            driver_location=list(self._state["driver_location"]),
            fleet=fleet_obs,
            external_factors=external,
            current_order=current_order,
            metrics=metrics,
        )

    def _resolve_vehicle(self, order: dict[str, Any], action: Action) -> dict[str, Any]:
        assert self._state is not None
        fleet = self._state["fleet"]
        if not fleet:
            raise RuntimeError("No vehicles available in environment.")

        requested_vehicle_id = action.vehicle_id
        if requested_vehicle_id is None and action.assignments:
            requested_vehicle_id = int(action.assignments[0].vehicle_id)

        if requested_vehicle_id is not None:
            for vehicle in fleet:
                if int(vehicle["vehicle_id"]) == int(requested_vehicle_id):
                    return vehicle

        pickup = list(order["pickup"])
        return min(
            fleet,
            key=lambda v: (
                manhattan_distance(list(v["location"]), pickup),
                float(v["available_at_minutes"]),
            ),
        )

    def _build_fleet(self, task: dict[str, Any]) -> list[dict[str, Any]]:
        if "vehicles" in task and isinstance(task["vehicles"], list) and task["vehicles"]:
            fleet = []
            for vehicle in task["vehicles"]:
                fleet.append(
                    {
                        "vehicle_id": int(vehicle["vehicle_id"]),
                        "location": list(vehicle["location"]),
                        "available_at_minutes": float(vehicle.get("available_at_minutes", 0.0)),
                        "deliveries_completed": int(vehicle.get("deliveries_completed", 0)),
                    }
                )
            return fleet

        task_id = str(task.get("task_id", "medium"))
        count = 2 if task_id == "easy" else (3 if task_id == "medium" else 4)
        anchor = list(task["orders"][0].get("driver_location", [0, 0]))
        fleet = []
        for idx in range(count):
            fleet.append(
                {
                    "vehicle_id": idx,
                    "location": [int(anchor[0]) + idx, int(anchor[1]) - idx],
                    "available_at_minutes": 0.0,
                    "deliveries_completed": 0,
                }
            )
        return fleet
