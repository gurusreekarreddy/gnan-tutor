@app.post("/reset")
async def reset(task_id: str = "easy"):
    # Define the 3 tasks for the hackathon
    steps_map = {"easy": 10, "medium": 15, "hard": 20}
    energy_map = {"easy": 1.0, "medium": 1.0, "hard": 0.8}  # Hard starts with less energy

    # 🔥 Validation (handles invalid inputs cleanly)
    if task_id not in steps_map:
        print(f"[WARN] Invalid task_id '{task_id}', defaulting to 'easy'")
        task_id = "easy"

    # Initialize the state based on the task
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