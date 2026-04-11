import os
import json
import requests
import time
from openai import OpenAI

# =========================
# 1. ENV VARIABLES
# =========================
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")
ENV_NAME = "gnan-tutor"

# =========================
# 2. SMART FALLBACK
# =========================
def get_fallback(obs):
    energy = obs.get("energy", 1.0)
    mastery = obs.get("mastery", 0.0)
    if energy < 0.2:
        return {"action": "rest", "intensity": 1.0}
    if mastery >= 0.8:
        return {"action": "test", "intensity": 1.0}
    return {"action": "study", "intensity": 0.8}

# =========================
# 3. LLM ENGINE
# =========================
def get_action(obs):
    prompt = f"""You are an AI teacher. Manage a student's learning session.

Student state:
- Mastery: {obs.get('mastery', 0.0):.2f} (goal: reach 0.8)
- Energy: {obs.get('energy', 1.0):.2f} (don't let it hit 0.0)
- Steps left: {obs.get('steps_left', 0)}

Rules:
- study: increases mastery, costs energy
- rest: recovers energy, no mastery gain
- test: wins if mastery >= 0.8, penalized if not
- If energy < 0.2, must rest
- If energy hits 0.0, episode fails with -1.0

Respond ONLY with JSON, no markdown:
{{"action": "study", "intensity": 0.8}}"""

    try:
        res = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=40
        )
        content = res.choices[0].message.content.strip()
        content = content.replace("```json", "").replace("```", "").strip()
        data = json.loads(content)
        if data.get("action") not in ["study", "rest", "test"]:
            data["action"] = "study"
        data["intensity"] = max(0.1, min(1.0, float(data.get("intensity", 0.8))))
        return data
    except Exception:
        return get_fallback(obs)

# =========================
# 4. WAIT FOR SERVER
# =========================
def wait_for_server():
    for _ in range(30):
        try:
            r = requests.get(f"{ENV_BASE_URL}/health", timeout=2)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False

# =========================
# 5. MAIN EVALUATION LOOP
# =========================
def run():
    wait_for_server()

    for task_id in ["easy", "medium", "hard"]:
        # EXACT FORMAT: [START]
        print(f"[START] task={task_id} env={ENV_NAME} model={MODEL_NAME}", flush=True)

        obs = {}
        done = False
        step = 0
        rewards = []

        try:
            # Reset with JSON body — critical fix
            r = requests.post(
                f"{ENV_BASE_URL}/reset",
                json={"task_id": task_id},
                timeout=10
            )
            obs = r.json().get("observation", {})
        except Exception as e:
            # Guaranteed [END] even if reset fails
            print(f"[END] success=false steps=0 rewards=", flush=True)
            continue

        while not done and step < 20:
            step += 1
            action = get_action(obs)
            action_name = action.get("action", "study")
            intensity = float(action.get("intensity", 0.8))
            action_str = f"{action_name}({intensity})"

            reward = 0.0
            error_str = "null"

            try:
                s = requests.post(
                    f"{ENV_BASE_URL}/step",
                    json=action,
                    timeout=10
                )
                step_data = s.json()
                obs = step_data.get("observation", obs)
                reward = float(step_data.get("reward", 0.0))
                done = bool(step_data.get("done", False))
            except Exception as e:
                error_str = str(e).replace("\n", " ")[:50]
                done = True

            rewards.append(f"{reward:.2f}")

            # EXACT FORMAT: [STEP]
            print(
                f"[STEP] step={step} action={action_str} "
                f"reward={reward:.2f} done={str(done).lower()} "
                f"error={error_str}",
                flush=True
            )

            time.sleep(0.1)

        # EXACT FORMAT: [END]
        final_mastery = float(obs.get("mastery", 0.0))
        success = final_mastery >= 0.8
        rewards_str = ",".join(rewards) if rewards else "0.00"
        print(
            f"[END] success={str(success).lower()} "
            f"steps={step} "
            f"rewards={rewards_str}",
            flush=True
        )

if __name__ == "__main__":
    run()