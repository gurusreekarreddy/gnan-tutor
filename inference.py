import os
import json
import logging
from openai import OpenAI
from server.gnan_tutor_environment import GnanTutorEnv
from server.models import TutorAction

def main():
    client = OpenAI(
        base_url="https://api.endpoints.huggingface.cloud/v1",
        api_key=os.environ.get("HF_TOKEN", "mock_token")
    )
    
    env = GnanTutorEnv()
    tasks = ["easy", "medium", "hard"]
    
    for task in tasks:
        obs = env.reset(task_id=task)
        done = False
        print(f"[START] Task {task} | Initial Observation: {obs.model_dump()}")
        
        while not done:
            prompt = (
                f"You are an AI Tutor. Student observation: {obs.model_dump()}. "
                "Choose an action: 'study', 'rest', or 'test'. "
                "Intensity must be between 0.1 and 1.0. "
                "Output ONLY a valid JSON with keys 'action' and 'intensity'."
            )
            
            try:
                response = client.chat.completions.create(
                    model="meta-llama/Llama-2-70b-chat-hf",
                    messages=[
                        {"role": "system", "content": "You are an AI Tutor returning JSON. Only output valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"}
                )
                
                content = response.choices[0].message.content
                action_data = json.loads(content)
                act = action_data.get("action", "study")
                intensity = float(action_data.get("intensity", 0.5))
            except Exception:
                # Fallback if unparsable or API issue
                act = "study"
                intensity = 0.5
                
            action = TutorAction(action=act, intensity=intensity)
            
            obs, reward, done, info = env.step(action)
            print(f"[STEP] Action: {act}, Intensity: {intensity:.2f} | Reward: {reward:.3f} | Observation: {obs.model_dump()}")
            
        score = env.grader()
        print(f"[END] Task {task} Completed | Final Score: {score:.3f}\n")

if __name__ == "__main__":
    main()
