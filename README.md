---
title: Gnan AI Tutor
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# 🧠 Gnan AI Tutor - Cognitive RL Environment

This environment models human-like learning under fatigue constraints, bridging reinforcement learning with cognitive science principles.

**Gnan Tutor** is a fatigue-aware Reinforcement Learning environment designed for the **Meta OpenEnv Challenge (2026)**. It moves beyond simple point-scoring by simulating cognitive exhaustion—agents must balance study intensity, energy depletion, and scheduled rest to maximize mastery. Built with a deterministic Smart Fallback Policy for LLM fault tolerance, Gnan Tutor provides a strictly clamped, exploit-proof MDP optimized for high-performance evaluation under tight hardware constraints.

## 🚀 Key Architecture
- **Backend:** FastAPI (Python 3.10)
- **Dependency Management:** `uv` for ultra-fast, frozen builds.
- **Containerization:** Fully Dockerized for seamless deployment and reproducibility.
- **Author:** Anta (Guru Sreekar Reddy)

## 📊 Environment Specification

### Observation Space
The agent receives a state dictionary containing:
* **`mastery`**: [0.0 - 1.0] Current level of knowledge acquired.
* **`energy`**: [0.0 - 1.0] Remaining cognitive stamina.
* **`steps_left`**: Integer count of remaining attempts.
* **`last_mastery_gain`**: Delta of mastery from the previous action.

### Action Space
* **`action`**: `["study", "rest", "test"]`
* **`intensity`**: [0.0 - 1.0] Scalar multiplier for the chosen action's impact.

## 🧩 Tasks
The environment includes three progressively difficult tasks:
- **Easy:** 10 steps, full energy (1.0)
- **Medium:** 15 steps, full energy (1.0)
- **Hard:** 20 steps, reduced starting energy (0.8)

Each task evaluates the agent’s ability to balance learning efficiency with fatigue management under increasing constraints.

## ⚖️ Core Environment Logic & Realism
To prevent "exploitation" and ensure behavioral realism, Gnan Tutor implements strict constraints:

1.  **Cognitive Fatigue:** If student energy drops below **0.3**, the mastery gain from `study` actions is penalized by **50%**.
2.  **The Burnout Rule:** If `energy` hits **0.0**, the episode terminates immediately with a heavy penalty (`reward: -1.0`). 
3.  **The Win Condition:** A `test` action only succeeds if `mastery` is **>= 0.8**, rewarding the agent with a successful completion state. 
4.  **Reward Shaping:** The agent receives incremental rewards for mastery gains during study. Additional rewards/penalties are applied for test success/failure. Final performance correlates strongly with achieved mastery.
5.  **Bounds Clamping:** All state variables are strictly clamped (`0.0` to `1.0`) to prevent out-of-bounds exploits.

## 🛡️ Fault Tolerance & Smart Fallback
To ensure uninterrupted evaluation—even during API timeouts, rate limits, or LLM hallucinations—the inference engine includes a deterministic **Smart Fallback Policy**. 

If the LLM fails to return a valid JSON action, the agent autonomously reverts to a rule-based safety protocol:
- **Low Energy (< 0.3):** Forces `rest` to prevent burnout.
- **High Mastery (>= 0.8):** Forces `test` to secure the win condition.
- **Default:** Executes a safe `study` action with balanced intensity.

## 🧪 Running Baseline Evaluation
To evaluate an LLM agent across all tasks (easy, medium, hard) and generate the required structured logs (START / STEP / END):

```bash
export HF_TOKEN="your_huggingface_token_here"
python3 inference.py