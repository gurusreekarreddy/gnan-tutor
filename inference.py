import os
import json
import requests
import time
from openai import OpenAI

# =========================
# 1. ENV VARIABLES
# =========================
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN is required for hackathon evaluation.")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
ENV_BASE_URL = "https://guru2408-gnan-tutor.hf.space"
MAX_STEPS = 15

# =========================
# 2. SMART FALLBACK
# =========================
def smart_fallback(observation):
    energy = observation.get("energy", 1.0)
    mastery = observation.get("mastery", 0.0)
    if energy < 0.3:
        return {"action": "rest", "intensity": 0.8}
    elif mastery >= 0.8:
        return {"action": "test", "intensity": 0.9}
    else:
        return {"action": "study", "intensity": 0.5 if energy > 0.6 else 0.3}

# =========================
# 3. LLM ENGINE
# =========================
def get_llm_action(observation):
    prompt = f"""
    Current state: {observation}
    Goal: Maximize mastery while avoiding energy depletion.
    Actions: study, rest, test. Intensity: 0.1 to 1.0.
    Respond ONLY in JSON: {{"action": "study", "intensity": 0.5}}
    """
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        data = json.loads(response.choices[0].message.content)
        if "action" not in data: data["action"] = "study"
        if "intensity" not in data: data["intensity"] = 0.5
        return data
    except Exception:
        return smart_fallback(observation)

# =========================
# 4. STRICT EVALUATION LOOP
# =========================
def run_evaluation():
    # Loop exactly through the 3 tasks
    for task_id in ["easy", "medium", "hard"]:
        
        # STRICT LOG: [START]
        print(f"[START] task={task_id} env=gnan-tutor model={MODEL_NAME}", flush=True)

        response = requests.post(f"{ENV_BASE_URL}/reset", params={"task_id": task_id}).json()
        obs = response.get("observation", {})
        done = False
        step_count = 0
        rewards_list = []

        while not done and step_count < MAX_STEPS:
            step_count += 1
            action_payload = get_llm_action(obs)
            
            # Format action for log (e.g., "study(0.5)")
            act_str = f"{action_payload.get('action')}({action_payload.get('intensity')})"

            step_response = requests.post(f"{ENV_BASE_URL}/step", json=action_payload).json()
            obs = step_response.get("observation", {})
            reward = float(step_response.get("reward", 0.0))
            done = step_response.get("done", False)
            
            rewards_list.append(reward)
            done_str = str(done).lower()

            # STRICT LOG: [STEP]
            print(f"[STEP] step={step_count} action={act_str} reward={reward:.2f} done={done_str} error=null", flush=True)

        # STRICT LOG: [END]
        score = float(obs.get("mastery", 0.0))
        success_str = str(score >= 0.8).lower()
        rewards_csv = ",".join([f"{r:.2f}" for r in rewards_list])
        
        print(f"[END] success={success_str} steps={step_count} score={score:.3f} rewards={rewards_csv}", flush=True)

if __name__ == "__main__":
    run_evaluation()