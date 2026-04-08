from fastapi import FastAPI
from server.gnan_tutor_environment import GnanTutorEnv
from server.models import TutorAction

app = FastAPI()
env = GnanTutorEnv()

@app.get("/")
def root():
    return {"status": "Gnan Tutor Environment Running 🚀"}

@app.post("/reset")
def reset():
    obs = env.reset()
    obs_data = obs.model_dump() if hasattr(obs, 'model_dump') else obs.dict() if hasattr(obs, 'dict') else obs
    return {"observation": obs_data}

@app.post("/step")
def step(action: dict):
    action_obj = TutorAction(**action)
    obs, reward, done, info = env.step(action_obj)
    
    obs_data = obs.model_dump() if hasattr(obs, 'model_dump') else obs.dict() if hasattr(obs, 'dict') else obs
        
    return {
        "observation": obs_data,
        "reward": float(reward),
        "done": bool(done),
        "info": info
    }

@app.get("/state")
def state():
    state_data = env.state() if hasattr(env, 'state') else {}
    return {"state": state_data}
