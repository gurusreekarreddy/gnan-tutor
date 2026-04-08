import os
import json
from openai import OpenAI

# Import your environment
from server.gnan_tutor_environment import GnanTutorEnv
from server.models import TutorAction


# 🔐 Environment check
HF_TOKEN = os.environ.get("HF_TOKEN")
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN is not set in environment variables.")

# 🧠 Initialize OpenAI client (HF router)
client = OpenAI(
    api_key=HF_TOKEN,
    base_url=API_BASE_URL,
)

MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"


def get_action_from_llm(observation):
    """
    Calls LLM to decide next action.
    Includes JSON safety fallback.
    """
    prompt = f"""
You are an intelligent tutor agent.

Given the student state:
{observation}

Choose the best action:
- study
- rest
- test

Also provide intensity (0.1 to 1.0).

Respond ONLY in JSON format:
{{"action": "study", "intensity": 0.5}}
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )

        content = response.choices[0].message.content.strip()

        # 🔥 Try parsing JSON
        parsed = json.loads(content)

        action = parsed.get("action", "rest")
        intensity = float(parsed.get("intensity", 0.5))

        # Clamp intensity
        intensity = max(0.1, min(1.0, intensity))

        return action, intensity

    except Exception as e:
        # 🚨 Fallback safety (VERY IMPORTANT)
        print(f"[WARN] LLM parsing failed, fallback to rest. Error: {e}")
        return "rest", 0.5


def run_task(task_id):
    env = GnanTutorEnv(task_id=task_id)

    obs = env.reset()

    print(f"[START] Task: {task_id} | Initial State: {obs}")

    done = False

    while not done:
        action, intensity = get_action_from_llm(obs)

        action_obj = TutorAction(action=action, intensity=intensity)
        obs, reward, done, info = env.step(action_obj)

        print(f"[STEP] Action: {action}, Intensity: {intensity:.2f} | Reward: {reward:.3f} | State: {obs}")

        if done:
            break

    final_state = env.state()
    print(f"[END] Task: {task_id} | Final State: {final_state}\n")


def main():
    for task in ["easy", "medium", "hard"]:
        run_task(task)


if __name__ == "__main__":
    main()