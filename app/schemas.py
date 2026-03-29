from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Assignment(BaseModel):
    order_id: int
    vehicle_id: int
    priority: int = 0


class RouteUpdate(BaseModel):
    vehicle_id: int
    route: list[int]


class Action(BaseModel):
    route_choice: int = Field(ge=0)
    vehicle_id: int | None = Field(default=None, ge=0)
    assignments: list[Assignment] = Field(default_factory=list)
    route_updates: list[RouteUpdate] = Field(default_factory=list)


class RouteOption(BaseModel):
    route_id: int
    distance_km: float
    traffic_factor: float


class CurrentOrder(BaseModel):
    order_id: int
    pickup: list[int]
    drop: list[int]
    deadline_minutes: float
    restaurant_wait: float
    weather_penalty: float
    traffic_level: float
    routes: list[RouteOption]


class Metrics(BaseModel):
    completed_orders: int
    on_time_deliveries: int
    total_delay: float
    total_cost: float


class VehicleState(BaseModel):
    vehicle_id: int
    location: list[int]
    available_in_minutes: float
    deliveries_completed: int


class ExternalFactors(BaseModel):
    traffic_index: float
    weather_index: float
    rush_hour: bool


class Observation(BaseModel):
    task_id: str
    time_step: int
    max_steps: int
    driver_location: list[int]
    fleet: list[VehicleState] = Field(default_factory=list)
    external_factors: ExternalFactors
    current_order: CurrentOrder | None
    metrics: Metrics


class ResetRequest(BaseModel):
    task_id: str = "easy"
    session_id: str | None = None


class StepRequest(BaseModel):
    session_id: str
    action: Action


class ResetResponse(BaseModel):
    session_id: str
    observation: Observation
    info: dict[str, Any]


class StepResponse(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: dict[str, Any]


class StateResponse(BaseModel):
    session_id: str
    state: dict[str, Any]


class GraderResponse(BaseModel):
    task_id: str
    score: float
    components: dict[str, float]
    metrics: dict[str, Any]


class BaselineTaskScore(BaseModel):
    task_id: str
    score: float


class BaselineResponse(BaseModel):
    results: list[BaselineTaskScore]
