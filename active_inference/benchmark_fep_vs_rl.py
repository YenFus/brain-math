"""
benchmark_fep_vs_rl.py - Active Inference vs. Standard Reinforcement Learning Benchmark.

Compares:
  1. Active Inference Connectome (Free Energy Principle minimizing G = Risk + Ambiguity)
  2. Standard Q-Learning / TD Agent (Scalar reward maximization with epsilon-greedy)

Evaluates performance in environments with deceptive obstacle traps (scent leaking through walls).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, List
import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from active_inference.active_inference_agent import ActiveInferenceConnectomeAgent


class StandardQLearningAgent:
    """Standard Q-learning baseline for comparison."""

    def __init__(self, n_obs: int = 5, n_actions: int = 3, lr: float = 0.1, gamma: float = 0.9, epsilon: float = 0.15):
        self.Q = np.zeros((n_obs, n_actions))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.prev_obs = 0
        self.prev_action = 0

    def act(self, obs: int) -> int:
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.Q.shape[1])
        else:
            action = int(np.argmax(self.Q[obs, :]))
        self.prev_obs = obs
        self.prev_action = action
        return action

    def update(self, next_obs: int, reward: float):
        td_target = reward + self.gamma * np.max(self.Q[next_obs, :])
        self.Q[self.prev_obs, self.prev_action] += self.lr * (td_target - self.Q[self.prev_obs, self.prev_action])


def run_benchmark(steps: int = 1000) -> Dict:
    print("=" * 75)
    print("🧠  ACTIVE INFERENCE (FEP) vs REINFORCEMENT LEARNING BENCHMARK  🧠")
    print("=" * 75)

    # 1. Run Active Inference Agent
    print("   • Running Active Inference Connectome (FEP Minimization)...")
    agent_fep = ActiveInferenceConnectomeAgent(seed=42)
    fep_fe = []
    for _ in range(steps):
        info = agent_fep.step()
        fep_fe.append(info["free_energy"])

    res_fep = {
        "model": "active_inference_fep",
        "food_eaten": agent_fep.food_eaten,
        "collisions": agent_fep.collisions,
        "total_distance": round(agent_fep.total_distance, 2),
        "mean_free_energy": round(float(np.mean(fep_fe)), 3),
        "efficiency": round(agent_fep.food_eaten / (agent_fep.total_distance / 100.0 + 1e-4), 2),
    }

    # 2. Run Standard Q-Learning Agent on same environment
    print("   • Running Standard Q-Learning Baseline...")
    from experiments.exp1_embodied_organism import FluidArenaEnvironment
    from experiments.exp3_celegans_connectome import CelegansConnectome, WormBodyPhysics

    env = FluidArenaEnvironment(width=100.0, height=100.0, n_food=8, seed=42)
    conn = CelegansConnectome(n_segments=12)
    body = WormBodyPhysics(n_segments=12, segment_length=2.2)
    body.x = np.linspace(45.0, 45.0 + 11 * 2.2, 12)
    body.y = np.full(12, 50.0)

    q_agent = StandardQLearningAgent()
    q_food = 0
    q_collisions = 0
    q_dist = 0.0

    for _ in range(steps):
        hx, hy = float(body.x[0]), float(body.y[0])
        scent, min_dist = env.get_chemical_gradient(hx, hy)
        whiskers = env.sample_whiskers(hx, hy, float(body.heading))
        min_whisker = min(whiskers)

        obs = 3 if min_dist < 3.5 else (2 if min_whisker < 4.5 else (1 if scent > 0.15 else 0))
        act = q_agent.act(obs)

        # Connectome actuation
        i_avb = 14.0 if act in [0, 1] else 0.0
        i_ava = 26.0 if act == 2 else 0.0
        turn_bias = 0.4 if act == 1 else (-0.6 if act == 2 else 0.0)

        conn.AVB.V = float(np.clip(conn.AVB.V + i_avb * 0.15, -80.0, 20.0))
        conn.AVA.V = float(np.clip(conn.AVA.V + i_ava * 0.25, -80.0, 20.0))

        is_reversing = conn.step(
            anterior_touch=1.0 if obs == 2 or act == 2 else 0.0,
            posterior_touch=0.0,
            chemo_grad=scent,
            dt=2.0
        )
        if abs(turn_bias) > 0.01:
            conn.muscle_torques[0] = float(np.clip(conn.muscle_torques[0] + turn_bias * 0.5, -1.0, 1.0))

        body.step(conn.muscle_torques, is_reversing=is_reversing, dt=0.02)

        new_hx, new_hy = float(body.x[0]), float(body.y[0])
        collided = not (2.0 <= new_hx <= 98.0 and 2.0 <= new_hy <= 98.0)
        for obs_box in env.obstacles:
            if obs_box[0] <= new_hx <= obs_box[2] and obs_box[1] <= new_hy <= obs_box[3]:
                collided = True

        reward = -5.0 if collided else (15.0 if env.consume_food(new_hx, new_hy, radius=3.5) else -0.05)

        if collided:
            q_collisions += 1
            body.heading += math.pi * 0.7
        else:
            q_dist += abs(float(body.velocity)) * 0.02 * 8.0

        if reward > 5.0:
            q_food += 1

        next_obs = 3 if min_dist < 3.5 else (2 if min_whisker < 4.5 else (1 if scent > 0.15 else 0))
        q_agent.update(next_obs, reward)

    res_ql = {
        "model": "standard_q_learning",
        "food_eaten": q_food,
        "collisions": q_collisions,
        "total_distance": round(q_dist, 2),
        "efficiency": round(q_food / (q_dist / 100.0 + 1e-4), 2),
    }

    print("\n" + "-" * 75)
    print("📊  BENCHMARK RESULTS SUMMARY")
    print("-" * 75)
    print(f"  Active Inference (FEP) : Food = {res_fep['food_eaten']} | Collisions = {res_fep['collisions']} | Dist = {res_fep['total_distance']}m | Efficiency = {res_fep['efficiency']}")
    print(f"  Standard Q-Learning    : Food = {res_ql['food_eaten']} | Collisions = {res_ql['collisions']} | Dist = {res_ql['total_distance']}m | Efficiency = {res_ql['efficiency']}")
    print("=" * 75 + "\n")

    out_dir = Path(__file__).resolve().parent / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "fep_benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump({"active_inference": res_fep, "q_learning": res_ql}, f, indent=2)

    return {"active_inference": res_fep, "q_learning": res_ql}


if __name__ == "__main__":
    run_benchmark(steps=1000)
