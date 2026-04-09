import os
import json
import requests
import time
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN is required for hackathon evaluation.")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")
MAX_STEPS = 15

# ✅ STABLE FALLBACK (IMPORTANT)
def smart_fallback(observation):
    energy = observation.get("energy", 1.0)
    mastery = observation.get("mastery", 0.0)

    if mastery >= 0.8:
        return {"action": "test", "intensity": 1.0}

    if energy < 0.3:
        return {"action": "rest", "intensity": 0.8}

    if energy > 0.6:
        return {"action": "study", "intensity": 0.6}
    else:
        return {"action": "study", "intensity": 0.4}

# ✅ SAFE LLM CALL
def get_llm_action(observation):
    prompt = f"""
    State: {observation}
    Goal: Maximize mastery safely.
    Avoid energy dropping below 0.3.
    Respond only JSON:
    {{"action": "study", "intensity": 0.5}}
    """

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=50
        )
        content = response.choices[0].message.content.strip()

        if content.startswith("```"):
            content = content.replace("```json", "").replace("```", "").strip()

        data = json.loads(content)

        if data.get("action") not in ["study", "rest", "test"]:
            data["action"] = "study"

        intensity = float(data.get("intensity", 0.5))
        data["intensity"] = max(0.1, min(1.0, intensity))

        return data

    except Exception:
        return smart_fallback(observation)

# ✅ STRICT LOOP (DO NOT TOUCH FORMAT)
def run_evaluation():
    for task_id in ["easy", "medium", "hard"]:
        
        print(f"[START] task={task_id} env=gnan-tutor model={MODEL_NAME}", flush=True)

        try:
            response = requests.post(
                f"{ENV_BASE_URL}/reset",
                params={"task_id": task_id},
                timeout=30
            ).json()
        except Exception:
            response = requests.post(
                f"{ENV_BASE_URL}/reset?task_id={task_id}",
                timeout=30
            ).json()

        obs = response.get("observation", {})
        done = False
        step_count = 0
        rewards = []

        while not done and step_count < MAX_STEPS:
            step_count += 1

            action_payload = get_llm_action(obs)
            act_str = f"{action_payload.get('action')}({action_payload.get('intensity')})"

            try:
                step_response = requests.post(
                    f"{ENV_BASE_URL}/step",
                    json=action_payload,
                    timeout=30
                ).json()
            except Exception:
                break

            obs = step_response.get("observation", obs)
            reward = float(step_response.get("reward", 0.0))
            done = step_response.get("done", False)

            rewards.append(reward)

            print(f"[STEP] step={step_count} action={act_str} reward={reward:.2f} done={str(done).lower()} error=null", flush=True)

            time.sleep(0.2)

        score = float(obs.get("mastery", 0.0))
        success = str(score >= 0.8).lower()
        rewards_csv = ",".join([f"{r:.2f}" for r in rewards])

        print(f"[END] success={success} steps={step_count} score={score:.3f} rewards={rewards_csv}", flush=True)

if __name__ == "__main__":
    run_evaluation()