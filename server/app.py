import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Global state to remember the environment between steps
current_state = {}

class ActionPayload(BaseModel):
    action: str
    intensity: float = 0.5

# ----------------------------
# 1. RESET ENDPOINT
# ----------------------------
@app.post("/reset")
def reset(task_id: str = "easy"):
    global current_state
    steps_map = {"easy": 10, "medium": 15, "hard": 20}
    energy_map = {"easy": 1.0, "medium": 1.0, "hard": 0.8}
    if not task_id or task_id not in steps_map:
        task_id = "easy"
    current_state = {
        "mastery": 0.0,
        "energy": energy_map[task_id],
        "steps_left": steps_map[task_id],
        "last_mastery_gain": 0.0,
        "done": False,
        "reward": 0.0,
        "metadata": {
            "task_id": task_id,
            "difficulty": task_id
        }
    }
    return {"observation": current_state}

# ----------------------------
# 2. STEP ENDPOINT (THE MISSING PIECE)
# ----------------------------
@app.post("/step")
def step(payload: ActionPayload):
    global current_state
    # Safety catch if step is called before reset
    if not current_state:
        reset("easy")
    action = payload.action
    intensity = max(0.0, min(1.0, payload.intensity))
    
    reward = 0.0
    current_state["steps_left"] -= 1

    # Core Environment Logic
    if action == "study":
        if current_state["energy"] > 0:
            gain = 0.2 * intensity * current_state["energy"]
            current_state["mastery"] += gain
            current_state["energy"] -= (0.1 * intensity)
            reward = gain  # Reward for learning
        else:
            reward = -0.2  # Penalty for studying while exhausted
            
    elif action == "rest":
        current_state["energy"] += (0.3 * intensity)
        reward = 0.05  # Small reward for managing health safely
        
    elif action == "test":
        if current_state["mastery"] >= 0.8:
            reward = 1.0  # Massive reward for passing
            current_state["done"] = True
        else:
            reward = -0.5  # Heavy penalty for failing early

    # Clamp values between 0.0 and 1.0 so we don't break the rules
    current_state["mastery"] = min(1.0, max(0.0, current_state["mastery"]))
    current_state["energy"] = min(1.0, max(0.0, current_state["energy"]))

    # End the episode if we run out of time
    if current_state["steps_left"] <= 0:
        current_state["done"] = True

    return {
        "observation": current_state,
        "reward": reward,
        "done": current_state["done"],
        "info": {}
    }

# ----------------------------
# 3. MAIN RUNNER (THE FIX FOR SCALER)
# ----------------------------
def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()