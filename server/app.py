from fastapi import FastAPI
app = FastAPI()

@app.post("/reset")
def reset(task_id: str = "easy"):
    steps_map = {"easy": 10, "medium": 15, "hard": 20}
    energy_map = {"easy": 1.0, "medium": 1.0, "hard": 0.8}

    if not task_id or task_id not in steps_map:
        task_id = "easy"

    obs = {
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

    return {"observation": obs}