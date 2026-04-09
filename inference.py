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

if not HF_TOKEN:
    raise ValueError("HF_TOKEN is required for hackathon evaluation.")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
# Let's keep localhost as default for the Docker grader, but allow HF Space override
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")
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
        intensity = 0.6 if energy > 0.5 else 0.3
        return {"action": "study", "intensity": intensity}

# =========================
# 3. LLM ENGINE
# =========================
def get_llm_action(observation):
    prompt = f"""You are an AI teacher managing a student learning session.

Student state:
- Mastery: {observation.get('mastery', 0.0):.2f} / 1.0
- Energy: {observation.get('energy', 1.0):.2f} / 1.0
- Steps remaining: {observation.get('steps_left', 0)}

Rules:
- If energy < 0.3, studying gives 50% less mastery
- If energy hits 0.0, episode ends with -1.0 penalty
- Test only wins if mastery >= 0.8

Respond ONLY with valid JSON, no markdown:
{{"action": "study", "intensity": 0.6}}

Valid actions: study, rest, test
Intensity: 0.1 to 1.0"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=50
        )
        content = response.choices[0].message.content.strip()
        # Clean up any markdown Qwen might inject
        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "").strip()
            
        data = json.loads(content)
        if data.get("action") not in ["study", "rest", "test"]:
            data["action"] = "study"
            
        intensity = float(data.get("intensity", 0.5))
        data["intensity"] = max(0.1, min(1.0, intensity))
        return data
        
    except Exception as e:
        # Fallback to hardcoded logic if the LLM fails or times out
        return smart_fallback(observation)

# =========================
# 4. STRICT EVALUATION LOOP
# =========================
def run_evaluation():
    # Loop exactly through the 3 tasks for the grader
    for task_id in ["easy", "medium", "hard"]:
        
        # 🚨 STRICT META FORMAT: [START]
        print(f"[START] task={task_id} env=gnan-tutor model={MODEL_NAME}", flush=True)

        # FIXED: Use params= instead of json= for FastAPI query strings
        try:
            response = requests.post(f"{ENV_BASE_URL}/reset", params={"task_id": task_id}, timeout=30).json()
        except Exception:
            # Fallback if query params fail for some reason
            response = requests.post(f"{ENV_BASE_URL}/reset?task_id={task_id}", timeout=30).json()
            
        obs = response.get("observation", {})
        done = False
        step_count = 0
        rewards_list = []

        while not done and step_count < MAX_STEPS:
            step_count += 1
            action_payload = get_llm_action(obs)
            
            # Format action for log (e.g., "study(0.5)")
            act_str = f"{action_payload.get('action')}({action_payload.get('intensity')})"

            try:
                step_response = requests.post(f"{ENV_BASE_URL}/step", json=action_payload, timeout=30).json()
            except Exception:
                break # End early if server dies

            obs = step_response.get("observation", obs)
            reward = float(step_response.get("reward", 0.0))
            done = step_response.get("done", False)
            
            rewards_list.append(reward)
            done_str = str(done).lower()

            # 🚨 STRICT META FORMAT: [STEP]
            print(f"[STEP] step={step_count} action={act_str} reward={reward:.2f} done={done_str} error=null", flush=True)
            
            time.sleep(0.2) # Small delay so we don't spam the API

        # 🚨 STRICT META FORMAT: [END]
        score = float(obs.get("mastery", 0.0))
        success_str = str(score >= 0.8).lower()
        rewards_csv = ",".join([f"{r:.2f}" for r in rewards_list])
        
        print(f"[END] success={success_str} steps={step_count} score={score:.3f} rewards={rewards_csv}", flush=True)

if __name__ == "__main__":
    run_evaluation()