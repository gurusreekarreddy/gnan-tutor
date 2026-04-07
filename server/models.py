from pydantic import BaseModel, Field
from openenv.core.env_server import Action, Observation

class TutorAction(Action):
    action: str = Field(..., description="Action to perform: study, rest, or test")
    intensity: float = Field(..., ge=0.1, le=1.0, description="Intensity of the action")

class StudentObservation(Observation):
    mastery: float = Field(..., description="Current student mastery level [0-1]")
    energy: float = Field(..., description="Current student energy level [0-1]")
    steps_left: int = Field(..., description="Remaining steps in episode")
    last_mastery_gain: float = Field(..., description="Mastery gained in the last step")
