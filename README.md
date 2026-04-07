---
title: Gnan AI Tutor
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
base_path: /web
---

# Gnan AI Tutor

---
title: Gnan AI Tutor
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

This environment captures the trade-off between cognitive effort, fatigue accumulation, and learning efficiency under constrained time — a core challenge in real-world education systems.

This environment is intentionally lightweight to comply with strict 2vCPU/8GB constraints while preserving behavioral realism.

## Action & Observation Spaces
- **TutorAction**: Includes `action` (study, rest, test) and `intensity` (0.1 to 1.0).
- **StudentObservation**: Tracks `mastery`, `energy`, `steps_left`, and `last_mastery_gain`. This explicit visibility helps the LLM understand the temporal feedback loop.

## Grader Logic & Realism
The objective is to maximize mastery without hitting 0 energy.
- **Grader**: Returns a final score `[0.0 - 1.0]` equal to final mastery. If the student reaches burnout (`energy <= 0.0`), the episode terminates immediately and the final_score MUST be `0.0`.
- **Bounds Clamping**: All system state variables are strictly clamped (`self.mastery = min(1.0, self.mastery)`, `self.energy = max(0.0, min(1.0, self.energy))`) to lock out exploitation.
- **Burnout Override**: Exhausting the student results in an immediate done-state with heavily a penalized reward (-1.0).
- **Anti-Exploit Anti-Spam (Realism mechanics)**:
  - `study`: Yields variable mastery, heavily penalized by 50% if energy is dangerously low (< 0.3).
  - `rest`: Crucial for safely restoring energy without drain.
  - `test`: Verifies progress securely while draining minimal energy, yielding reward only on `mastery > 0.8`.
