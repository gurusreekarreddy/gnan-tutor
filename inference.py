import os
import json
import requests
import time
from openai import OpenAI

# =========================
# 1. ENV VARIABLES (MANDATORY)
# =========================
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN is required for hackathon evaluation.")

# =========================
# 2. OPENAI CLIENT INIT
# =========================
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)

ENV_BASE_URL = "https://guru2408-gnan-tutor.hf.space"
MAX_STEPS = 15


# =========================
# 3. SMART FALLBACK POLICY
# =========================
def smart_fallback(observation):
    energy = observation.get("energy", 1.0)
    mastery = observation.get("mastery", 0.0)

    if energy < 0.3:
        return {"action": "rest", "intensity": 0.8}
    elif mastery >= 0.8:
        return {"action": "test", "intensity": 0.9}
    else:
        return {
            "action": "study",
            "intensity": 0.5 if energy > 0.6 else 0.3
        }


# =========================
# 4. LLM DECISION ENGINE
# =========================
def get_llm_action(observation):
    prompt = f"""
    You are an AI student agent.

    Current state:
    {observation}

    Goal:
    Maximize mastery while avoiding energy depletion.

    Rules:
    - Actions: study, rest, test
    - Intensity: 0.0 to 1.0

    Respond ONLY in JSON:
    {{"action": "study", "intensity": 0.5}}
    """

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1
        )

        data = json.loads(response.choices[0].message.content)

        # Safety guards
        if "action" not in data:
            data["action"] = "study"
        if "intensity" not in data:
            data["intensity"] = 0.5

        return data

    except Exception as e:
        print(f"STEP: LLM ERROR -> {e} | Using fallback policy")
        return smart_fallback(observation)


# =========================
# 5. MAIN EVALUATION LOOP
# =========================
def run_evaluation():

    print("START: Evaluating AROHAN Lite / Gnan Tutor Environment")

    for task_id in ["easy", "medium", "hard"]:

        print(f"STEP: Beginning Task: {task_id}")

        # Reset environment with task_id
        response = requests.post(
            f"{ENV_BASE_URL}/reset",
            params={"task_id": task_id}
        ).json()

        obs = response.get("observation", {})
        done = False
        step_count = 0
        total_reward = 0.0

        print(f"STEP: Initial State -> {obs}")

        while not done and step_count < MAX_STEPS:
            step_count += 1

            # Agent decides action
            action_payload = get_llm_action(obs)

            print(
                f"STEP: Action -> {action_payload.get('action')} "
                f"| Intensity -> {action_payload.get('intensity')}"
            )

            # Environment step
            step_response = requests.post(
                f"{ENV_BASE_URL}/step",
                json=action_payload
            ).json()

            obs = step_response.get("observation", {})
            reward = step_response.get("reward", 0.0)
            done = step_response.get("done", False)

            total_reward += reward

            print(
                f"STEP: Reward -> {reward:.2f} | "
                f"Mastery -> {obs.get('mastery', 0):.2f} | "
                f"Energy -> {obs.get('energy', 0):.2f}"
            )

            time.sleep(0.5)

        print(
            f"STEP: Task {task_id} Completed | "
            f"Final Mastery -> {obs.get('mastery', 0):.2f} | "
            f"Total Reward -> {total_reward:.2f}"
        )

    print("END: Evaluation completed successfully")


# =========================
# 6. ENTRY POINT
# =========================
if __name__ == "__main__":
    run_evaluation()