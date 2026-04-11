import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="Gnan AI Tutor", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

current_state = {}
current_task_id = "easy"

TASK_CONFIGS = {
    "easy":   {"steps": 10, "energy": 1.0},
    "medium": {"steps": 8,  "energy": 0.7},
    "hard":   {"steps": 6,  "energy": 0.4},
}

class ResetRequest(BaseModel):
    task_id: Optional[str] = "easy"

class ActionPayload(BaseModel):
    action: str
    intensity: float = 0.5

# ----------------------------
# HEALTH
# ----------------------------
@app.get("/health")
def health():
    return {"status": "healthy", "env": "gnan-tutor"}

# ----------------------------
# RESET
# ----------------------------
@app.post("/reset")
def reset(request: ResetRequest):
    global current_state, current_task_id
    task_id = request.task_id or "easy"
    if task_id not in TASK_CONFIGS:
        task_id = "easy"
    current_task_id = task_id
    config = TASK_CONFIGS[task_id]
    current_state = {
        "mastery": 0.0,
        "energy": config["energy"],
        "steps_left": config["steps"],
        "last_mastery_gain": 0.0,
        "done": False,
        "reward": 0.0,
        "metadata": {"task_id": task_id, "difficulty": task_id}
    }
    return {"observation": current_state}

# ----------------------------
# STEP
# ----------------------------
@app.post("/step")
def step(payload: ActionPayload):
    global current_state
    if not current_state:
        reset(ResetRequest(task_id="easy"))

    action = payload.action
    intensity = max(0.1, min(1.0, payload.intensity))
    reward = 0.0
    current_state["steps_left"] -= 1
    prev_mastery = current_state["mastery"]

    if action == "study":
        energy_cost = 0.15 * intensity
        current_state["energy"] -= energy_cost
        current_state["energy"] = max(0.0, current_state["energy"])
        base_gain = 0.15 * intensity
        if current_state["energy"] < 0.3:
            base_gain *= 0.5
        current_state["mastery"] = min(1.0, current_state["mastery"] + base_gain)
        current_state["last_mastery_gain"] = round(
            current_state["mastery"] - prev_mastery, 4
        )
        reward = round(base_gain * 0.8, 4)
        if current_state["energy"] <= 0.0:
            current_state["done"] = True
            reward = -1.0

    elif action == "rest":
        old_energy = current_state["energy"]
        current_state["energy"] = min(
            1.0, current_state["energy"] + 0.3 * intensity
        )
        current_state["last_mastery_gain"] = 0.0
        reward = 0.05 if old_energy < 0.5 else 0.01

    elif action == "test":
        current_state["energy"] = max(
            0.0, current_state["energy"] - 0.05
        )
        current_state["last_mastery_gain"] = 0.0
        if current_state["mastery"] >= 0.8:
            reward = 1.0
            current_state["done"] = True
        else:
            reward = round(-0.1 * (0.8 - current_state["mastery"]), 4)

    # Clamp all values
    current_state["mastery"] = round(min(1.0, max(0.0, current_state["mastery"])), 4)
    current_state["energy"] = round(min(1.0, max(0.0, current_state["energy"])), 4)

    if current_state["steps_left"] <= 0:
        current_state["done"] = True

    current_state["reward"] = reward

    return {
        "observation": current_state,
        "reward": reward,
        "done": current_state["done"],
        "info": {}
    }

# ----------------------------
# STATE
# ----------------------------
@app.get("/state")
def state():
    return {
        "task_id": current_task_id,
        "done": current_state.get("done", False),
        "steps_left": current_state.get("steps_left", 0)
    }

# ----------------------------
# GRADER — clamped to avoid 0.0 and 1.0
# ----------------------------
@app.get("/grader")
def grader():
    if current_state.get("energy", 1.0) <= 0.0:
        return {"score": 0.001}
    raw_score = current_state.get("mastery", 0.0)
    clamped_score = max(0.001, min(0.999, raw_score))
    return {"score": clamped_score}

# ----------------------------
# TASKS
# ----------------------------
@app.get("/tasks")
def tasks():
    return {
        "tasks": [
            {
                "id": "easy",
                "description": "10 steps, full energy (1.0). Balance study and rest.",
                "difficulty": "easy",
                "max_steps": 10,
                "starting_energy": 1.0
            },
            {
                "id": "medium",
                "description": "8 steps, reduced energy (0.7). Tighter constraints.",
                "difficulty": "medium",
                "max_steps": 8,
                "starting_energy": 0.7
            },
            {
                "id": "hard",
                "description": "6 steps, low energy (0.4). Near-burnout from start.",
                "difficulty": "hard",
                "max_steps": 6,
                "starting_energy": 0.4
            }
        ],
        "action_schema": {
            "action": {
                "type": "string",
                "values": ["study", "rest", "test"]
            },
            "intensity": {
                "type": "float",
                "range": "0.1 to 1.0"
            }
        }
    }

# ----------------------------
# BASELINE
# ----------------------------
@app.post("/baseline")
def baseline():
    results = {}
    for task_id in ["easy", "medium", "hard"]:
        reset(ResetRequest(task_id=task_id))
        done = False
        total_reward = 0.0
        steps = 0
        while not done and steps < 25:
            obs = current_state
            if obs["energy"] < 0.3:
                a, i = "rest", 0.8
            elif obs["mastery"] >= 0.8:
                a, i = "test", 0.9
            else:
                a, i = "study", 0.6
            r = step(ActionPayload(action=a, intensity=i))
            total_reward += r["reward"]
            done = r["done"]
            steps += 1
        results[task_id] = {
            "total_reward": round(total_reward, 4),
            "final_score": current_state.get("mastery", 0.0),
            "steps": steps
        }
    return results

def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()