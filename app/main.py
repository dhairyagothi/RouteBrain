from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.responses import FileResponse

from app.env import RouteEnv
from app.schemas import (
    Action,
    BaselineResponse,
    BaselineTaskScore,
    GraderResponse,
    ResetRequest,
    ResetResponse,
    Observation,
    StateResponse,
    StepRequest,
    StepResponse,
)
from app.tasks import task_store
from app.utils import select_best_route
from configs.settings import DEFAULT_TASK_ID, SESSION_TTL_SECONDS

app = FastAPI(title="RouteBrain", version="1.0.0")
WEB_UI_PATH = Path(__file__).resolve().parent / "web" / "index.html"


@dataclass
class SessionEntry:
    env: RouteEnv
    last_access: float


SESSIONS: dict[str, SessionEntry] = {}


def _cleanup_sessions() -> None:
    now = time.time()
    expired = [sid for sid, entry in SESSIONS.items() if now - entry.last_access > SESSION_TTL_SECONDS]
    for sid in expired:
        del SESSIONS[sid]


def _get_session_env(session_id: str) -> RouteEnv:
    _cleanup_sessions()
    entry = SESSIONS.get(session_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="session_id not found")
    entry.last_access = time.time()
    return entry.env


@app.get("/")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "routebrain"}


@app.get("/health")
def openenv_health() -> dict[str, str]:
    return {"status": "healthy"}


@app.get("/metadata")
def openenv_metadata() -> dict[str, str]:
    return {
        "name": "routebrain-env",
        "description": "Deterministic last-mile route decision simulation for OpenEnv validation.",
    }


@app.get("/schema")
def openenv_schema() -> dict[str, Any]:
    return {
        "action": Action.model_json_schema(),
        "observation": Observation.model_json_schema(),
        "state": {"type": "object"},
    }


@app.post("/mcp")
def openenv_mcp(payload: dict[str, Any]) -> dict[str, Any]:
    req_id = payload.get("id")
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "result": {"status": "ok"},
    }


@app.get("/simulator")
def simulator_ui() -> FileResponse:
    return FileResponse(WEB_UI_PATH)


@app.get("/favicon.ico")
def favicon() -> Response:
    return Response(status_code=204)


@app.post("/reset", response_model=ResetResponse)
def reset_environment(request: ResetRequest | None = None) -> ResetResponse:
    if request is None:
        request = ResetRequest()

    _cleanup_sessions()
    task_id = request.task_id or DEFAULT_TASK_ID
    if task_id not in task_store.list_tasks():
        raise HTTPException(status_code=400, detail="Unknown task_id")

    session_id = request.session_id or str(uuid.uuid4())
    if session_id not in SESSIONS:
        SESSIONS[session_id] = SessionEntry(env=RouteEnv(), last_access=time.time())

    env = SESSIONS[session_id].env
    observation, info = env.reset(task_id=task_id)
    SESSIONS[session_id].last_access = time.time()

    return ResetResponse(session_id=session_id, observation=observation, info=info)


@app.get("/reset", response_model=ResetResponse)
def reset_environment_get(
    task_id: str = Query(DEFAULT_TASK_ID),
    session_id: str | None = Query(default=None),
) -> ResetResponse:
    return reset_environment(ResetRequest(task_id=task_id, session_id=session_id))


@app.post("/step", response_model=StepResponse)
def step_environment(request: StepRequest) -> StepResponse:
    env = _get_session_env(request.session_id)
    try:
        observation, reward, done, info = env.step(request.action)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return StepResponse(observation=observation, reward=reward, done=done, info=info)


@app.get("/state", response_model=StateResponse)
def get_state(session_id: str = Query(...)) -> StateResponse:
    env = _get_session_env(session_id)
    try:
        state = env.state()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return StateResponse(session_id=session_id, state=state)


@app.get("/tasks")
def list_tasks() -> dict[str, list[str]]:
    return {"tasks": task_store.list_tasks()}


@app.get("/grader", response_model=GraderResponse)
def grade(session_id: str = Query(...)) -> GraderResponse:
    env = _get_session_env(session_id)
    try:
        result = env.grade()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return GraderResponse(**result)


@app.post("/baseline", response_model=BaselineResponse)
def run_baseline() -> BaselineResponse:
    scores: list[BaselineTaskScore] = []

    for task_id in task_store.list_tasks():
        env = RouteEnv()
        obs, _ = env.reset(task_id)
        done = False

        while not done:
            if obs.current_order is None:
                break
            order_dict = obs.current_order.model_dump()
            fleet = obs.fleet
            vehicle_id = int(fleet[0].vehicle_id) if fleet else 0
            vehicle_location = list(fleet[0].location) if fleet else list(obs.driver_location)
            best_route_id, _ = select_best_route(order_dict, vehicle_location=vehicle_location, time_step=obs.time_step)
            obs, _, done, _ = env.step(Action(route_choice=best_route_id, vehicle_id=vehicle_id))

        grade_result = env.grade()
        scores.append(BaselineTaskScore(task_id=task_id, score=float(grade_result["score"])))

    return BaselineResponse(results=scores)
