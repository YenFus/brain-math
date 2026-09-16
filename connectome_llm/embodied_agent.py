"""
embodied_agent.py - Closed-Loop Language-Guided Connectome Agent.

Integrates:
  - Natural language instruction parser
  - Brainstem descending motor bridge
  - Biophysical 12-segment C. elegans connectome
  - Hydrodynamic fluid physics environment
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.exp1_embodied_organism import FluidArenaEnvironment
from experiments.exp3_celegans_connectome import CelegansConnectome, WormBodyPhysics
from connectome_llm.brainstem_bridge import ConnectomeBrainstemBridge, CognitiveSetpoint
from connectome_llm.language_policy import LanguageConnectomePolicy


class LanguageGuidedConnectomeAgent:
    """An embodied worm guided by natural language instructions."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.env = FluidArenaEnvironment(width=100.0, height=100.0, n_food=8, seed=seed)
        self.connectome = CelegansConnectome(n_segments=12)
        self.body = WormBodyPhysics(n_segments=12, segment_length=2.2)

        # Place worm at center
        self.body.x = np.linspace(45.0, 45.0 + 11 * 2.2, 12)
        self.body.y = np.full(12, 50.0)

        self.policy = LanguageConnectomePolicy()
        self.bridge = ConnectomeBrainstemBridge()

        self.glucose = 1.0
        self.food_eaten = 0
        self.collisions = 0
        self.total_distance = 0.0
        self.prev_scent = 0.0

    def set_goal(self, instruction: str):
        """Set or update the natural language goal."""
        self.policy.set_instruction(instruction)

    def step(self, dt_phys: float = 0.02) -> Dict:
        """Run one closed-loop step."""
        hx = float(self.body.x[0])
        hy = float(self.body.y[0])

        # 1. Sense environment
        scent, min_food_dist = self.env.get_chemical_gradient(hx, hy)
        scent_delta = scent - self.prev_scent
        self.prev_scent = scent

        whiskers = self.env.sample_whiskers(hx, hy, float(self.body.heading))
        min_whisker = min(whiskers)
        threat_level = float(np.clip((6.0 - min_whisker) / 6.0, 0.0, 1.0))

        # 2. Language policy evaluation
        state_ctx = {
            "head_x": hx,
            "head_y": hy,
            "heading": float(self.body.heading),
            "min_whisker": min_whisker,
            "food_near": min_food_dist < 4.0,
            "arena_size": self.env.width,
        }
        setpoint = self.policy.evaluate(state_ctx)

        # 3. Brainstem bridge: convert setpoint into descending currents
        currents = self.bridge.compute_descending_currents(setpoint, threat_level, scent_delta)

        # 4. Inject currents into connectome
        # Sensory / Touch
        anterior_touch = 1.0 if min_whisker < 3.5 or currents["I_AVA"] > 15.0 else 0.0
        chemo_input = scent * setpoint.chemo_attraction

        # Direct current injection into command interneurons
        self.connectome.AVB.V = float(np.clip(self.connectome.AVB.V + currents["I_AVB"] * 0.15, -80.0, 20.0))
        self.connectome.AVA.V = float(np.clip(self.connectome.AVA.V + currents["I_AVA"] * 0.20, -80.0, 20.0))

        # Run biophysical connectome step
        is_reversing = self.connectome.step(
            anterior_touch=anterior_touch,
            posterior_touch=0.0,
            chemo_grad=chemo_input,
            dt=2.0
        )

        # Apply steer bias from cognitive setpoint to head muscles
        if abs(currents["turn_bias"]) > 0.01:
            self.connectome.muscle_torques[0] = float(np.clip(self.connectome.muscle_torques[0] + currents["turn_bias"] * 0.5, -1.0, 1.0))

        # 5. Physics step
        self.body.step(self.connectome.muscle_torques, is_reversing=is_reversing, dt=dt_phys)

        # Collision check
        new_hx, new_hy = float(self.body.x[0]), float(self.body.y[0])
        collided = False
        if not (2.0 <= new_hx <= self.env.width - 2.0 and 2.0 <= new_hy <= self.env.height - 2.0):
            collided = True
        for obs in self.env.obstacles:
            if obs[0] <= new_hx <= obs[2] and obs[1] <= new_hy <= obs[3]:
                collided = True

        if collided:
            self.collisions += 1
            self.body.heading += math.pi * 0.7  # bounce / turn away
        else:
            self.total_distance += abs(float(self.body.velocity)) * dt_phys * 8.0

        # Food consumption
        if self.env.consume_food(new_hx, new_hy, radius=3.5):
            self.food_eaten += 1
            self.glucose = min(1.0, self.glucose + 0.3)

        self.glucose = max(0.05, self.glucose - 0.0008)

        return {
            "head_x": round(new_hx, 2),
            "head_y": round(new_hy, 2),
            "heading": round(float(self.body.heading), 3),
            "velocity": round(float(self.body.velocity), 3),
            "is_reversing": bool(is_reversing),
            "behavior_mode": setpoint.behavior_mode,
            "v_AVA": round(self.connectome.AVA.V, 1),
            "v_AVB": round(self.connectome.AVB.V, 1),
            "food_eaten": self.food_eaten,
            "glucose": round(self.glucose, 3),
            "body_x": [round(float(v), 2) for v in self.body.x],
            "body_y": [round(float(v), 2) for v in self.body.y],
        }
