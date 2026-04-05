---
title: RouteBrain
emoji: "🚚"
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# RouteBrain

RouteBrain is a deterministic, OpenEnv-compliant environment for evaluating AI decision making in last-mile delivery.

## Hugging Face Space Deploy

This repo is ready for a Docker Space.

1. Create Space: `Dhairyagothi/RouteBrain` with SDK set to `Docker`.
2. Add Space secrets/variables:
	- `API_BASE_URL`
	- `MODEL_NAME`
	- `HF_TOKEN`
	- `ROUTEBRAIN_URL` (optional)
	- `MAX_STEPS` (optional)
3. Push this repository contents to the Space repo.

The container listens on port `7860` as required by Hugging Face Spaces.

The environment simulates route selection under operational constraints such as traffic, weather impact, wait time, delivery deadlines, and route cost. An agent receives structured observations and returns structured actions. The environment advances with a strict reset/step/state loop and produces both step rewards and a final grader score.

This repository is built with a validation-first mindset for hackathon judging:
- FastAPI service on port 7860
- Required endpoints: /reset, /step, /state, /tasks, /grader, /baseline
- Deterministic scenario fixtures for reproducibility
- Score in [0, 1] that changes with agent performance
- Runnable baseline inference flow

## What The Project Solves

RouteBrain is not a delivery app. It is a decision simulation system used to test and compare routing behavior.

For each step, the environment presents a delivery order, a fleet snapshot (multiple drivers), and multiple route options. The agent chooses both route and vehicle. The simulator computes ETA, delay, and cost using deterministic formulas that include driver-to-pickup travel, traffic, weather, and rush-hour effects. Rewards are returned immediately, and a final grader score is computed at episode end.

This makes RouteBrain suitable for:
- Agent benchmarking
- Reproducible evaluation across tasks
- Rapid experimentation with decision policies
- Judge-friendly deterministic replay

## System Architecture

The system is organized as a clean layered pipeline:

```text
[Agent / Inference Client]
					|
					v
[FastAPI Interface Layer]
	/reset /step /state /tasks /grader /baseline
					|
					v
[RouteEnv State Machine]
	reset() -> step() -> state() -> grade()
					|
					v
[Simulation + Reward + Grader]
	simulator.py / reward.py / grader.py
					|
					v
[Deterministic Task Data]
	task_easy.json / task_medium.json / task_hard.json
```

### Component Responsibilities

- app/main.py
	API contracts, session lifecycle, endpoint routing.

- app/env.py
	Core environment state machine. Enforces reset/step/state semantics.

- app/simulator.py
	Deterministic route evaluation and route quality calculation.

- app/reward.py
	Dense per-step reward shaping from ETA, delay, cost, and action validity.

- app/grader.py
	Final episode score in [0, 1] based on success, delay, and cost normalization.

- app/tasks.py + data/*.json
	Scenario loading and deterministic difficulty progression.

- inference.py
	End-to-end baseline runner: reset -> step loop -> grader -> score output.

## Environment Lifecycle

### 1) Reset

`POST /reset`

Initializes a session for a chosen task (`easy`, `medium`, `hard`) and returns initial observation.

### 2) Step

`POST /step`

Takes action input (`route_choice`) and executes:
- vehicle selection (explicit `vehicle_id` or nearest-driver fallback)
- action validation
- deterministic simulation update
- metric accumulation
- reward computation
- termination check

Returns:
- observation
- reward
- done
- info

### 3) State

`GET /state`

Returns full internal environment state for inspection/debugging.

### 4) Grader

`GET /grader`

Returns final score and score components.

## Scoring and Reward Design

Step reward is dense and non-sparse, combining:
- speed component
- delay penalty
- cost penalty
- on-time bonus
- invalid action penalty

Final grader score is deterministic and bounded:

```text
score = 0.5 * success_rate + 0.3 * (1 - delay_norm) + 0.2 * (1 - cost_norm)
```

All outputs are clamped to valid ranges to prevent drift.

## Determinism Guarantees

RouteBrain is intentionally deterministic:
- no live traffic or maps API calls
- no uncontrolled randomness in the step loop
- fixed task fixtures drive scenario behavior
- external factors (traffic/weather/rush-hour) are scenario-derived, not random

Given the same task and the same action sequence, outcomes remain reproducible.

## Project Structure

- app/main.py: API endpoints and tokenized session manager
- app/env.py: environment state machine (reset, step, state, grade)
- app/schemas.py: strict Pydantic models
- app/simulator.py: deterministic multi-route + multi-driver evaluation
- app/reward.py: dense step reward
- app/grader.py: final score computation
- app/tasks.py: task loading from data files
- data/task_easy.json, data/task_medium.json, data/task_hard.json: deterministic scenarios
- openenv.yaml: OpenEnv metadata
- inference.py: baseline script with Groq-compatible OpenAI SDK flow
- Dockerfile: deployment entry

## Local Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Start API:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 7860
```

3. Smoke check:

```bash
curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{"task_id":"easy"}'
```

4. Open simulator UI in browser:

```text
http://localhost:7860/simulator
```

The simulator page helps you understand what is happening at each step:
- reset task and start a new session
- view current order details, fleet status, and external factors
- click a route to run one step manually
- use auto play to run best-route heuristic quickly
- inspect timeline of reward, ETA, delay, and validity
- call grader and view final score components

## Docker Run

```bash
docker build -t routebrain .
docker run -p 7860:7860 routebrain
```

## Inference Script

The required flow is implemented in inference.py:
1. call /reset
2. loop model action -> /step
3. call /grader
4. print score

Set environment variables (see .env.example), then run:

```bash
python inference.py
```

If GROQ_API_KEY is not set, the script falls back to a deterministic heuristic and still completes.

### Groq Baseline Configuration

Use these variables for inference:

```env
API_BASE_URL=https://api.groq.com/openai/v1
MODEL_NAME=llama-3.1-8b-instant
GROQ_API_KEY=your_key
```

## API Contracts

### POST /reset

Request:

```json
{
	"task_id": "easy",
	"session_id": "optional-session-id"
}
```

### POST /step

Request:

```json
{
	"session_id": "<session-id>",
	"action": {
		"route_choice": 0,
		"vehicle_id": 1,
		"assignments": [],
		"route_updates": []
	}
}
```

Response shape:

```json
{
	"observation": {},
	"reward": 0.0,
	"done": false,
	"info": {}
}
```

### GET /grader?session_id=<id>

Returns deterministic final score in [0, 1] using:

```text
score = 0.5 * success + 0.3 * (1 - delay_norm) + 0.2 * (1 - cost_norm)
```

## Runtime Constraints

- Baseline loop capped by MAX_STEPS (default 20)
- Lightweight per-step computations for 2 vCPU / 8GB environments