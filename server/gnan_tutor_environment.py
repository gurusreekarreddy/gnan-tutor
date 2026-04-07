import random
from typing import Tuple, Dict, Any
from openenv.core.env_server import Environment
from .models import TutorAction, StudentObservation

class GnanTutorEnv(Environment):
    def __init__(self):
        super().__init__()
        self.mastery = 0.0
        self.energy = 1.0
        self.steps_left = 0
        self.last_mastery_gain = 0.0
        self.task_drain_multiplier = 1.0
        self.burnout = False
        self.is_done = False
        self.task_id = ""

    def reset(self, task_id: str = "easy", **kwargs) -> Any:
        self.task_id = task_id
        
        if self.task_id == "easy":
            self.steps_left = 10
            self.energy = 1.0
            self.task_drain_multiplier = 1.0
        elif self.task_id == "medium":
            self.steps_left = 15
            self.energy = 1.0
            self.task_drain_multiplier = 1.0
        elif self.task_id == "hard":
            self.steps_left = 8
            self.energy = 0.4
            self.task_drain_multiplier = 1.5
        else:
            self.steps_left = 10
            self.energy = 1.0
            self.task_drain_multiplier = 1.0
            
        self.last_mastery_gain = 0.0
        self.mastery = 0.0
        self.burnout = False
        self.is_done = False
        
        return self.state()

    def step(self, action: TutorAction) -> Tuple[Any, float, bool, Dict[str, Any]]:
        if self.is_done:
            return self.state(), 0.0, self.is_done, {"error": "Environment already done."}

        # 1. Update variables
        self.steps_left -= 1
        random.seed(self.steps_left + sum(ord(c) for c in self.task_id))
        
        act = action.action
        intensity = action.intensity
        
        gain = 0.0
        energy_drain = 0.0
        reward = 0.0
        
        if act == 'study':
            base_gain = 0.2 * intensity
            if self.energy < 0.3:
                base_gain *= 0.5
            raw_gain = base_gain * random.uniform(0.8, 1.2)
            gain = max(0.0, raw_gain)
            energy_drain = 0.3 * intensity * self.task_drain_multiplier
            reward = gain
        elif act == 'rest':
            self.energy += 0.4
            reward = 0.05
        elif act == 'test':
            energy_drain = 0.2
            if self.mastery > 0.8:
                reward = 0.5
            else:
                reward = 0.0

        self.mastery += gain
        self.energy -= energy_drain

        # 2. Clamp values
        self.mastery = min(1.0, max(0.0, self.mastery))
        
        # 3. Check Burnout
        if self.energy <= 0.0:
            self.burnout = True
            self.is_done = True
            self.energy = max(0.0, min(1.0, self.energy))
            self.last_mastery_gain = min(1.0, gain)
            return self.state(), -1.0, self.is_done, {"info": "burnout"}
            
        self.energy = max(0.0, min(1.0, self.energy))
        self.last_mastery_gain = min(1.0, gain)

        if self.steps_left <= 0:
            self.is_done = True
            
        # 4. Assign Reward
        return self.state(), reward, self.is_done, {}

    def state(self) -> StudentObservation:
        return StudentObservation(
            mastery=self.mastery,
            energy=self.energy,
            steps_left=self.steps_left,
            last_mastery_gain=self.last_mastery_gain
        )
        
    def grader(self) -> float:
        if self.burnout:
            return 0.0
        return self.mastery
