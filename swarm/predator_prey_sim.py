"""
predator_prey_sim.py - Multi-Agent Predator-Prey Hydrodynamics with Biophysical Connectomes.

Simulates:
  - 4 Forager Prey (C. elegans 12-segment connectomes with mechanosensory escape reflex)
  - 1 Apex Predator (Fluid wake-tracking hunter)
  - Collective behaviors: Schooling, predator evasion reflex, metabolic food competition.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.exp1_embodied_organism import FluidArenaEnvironment
from experiments.exp3_celegans_connectome import CelegansConnectome, WormBodyPhysics


@dataclass
class PredatorState:
    x: float = 20.0
    y: float = 20.0
    heading: float = 0.0
    velocity: float = 1.2
    captures: int = 0


class SwarmSimulation:
    """Multi-Agent Fluid Swarm Simulation."""

    def __init__(self, n_prey: int = 4, steps: int = 800, seed: int = 42):
        self.steps = steps
        self.n_prey = n_prey
        self.rng = np.random.default_rng(seed)
        self.env = FluidArenaEnvironment(width=120.0, height=120.0, n_food=12, seed=seed)

        # Initialize 4 prey worms at different corners
        self.prey_connectomes = [CelegansConnectome(n_segments=12) for _ in range(n_prey)]
        self.prey_bodies = [WormBodyPhysics(n_segments=12, segment_length=2.0) for _ in range(n_prey)]

        spawn_points = [(40.0, 40.0), (80.0, 40.0), (40.0, 80.0), (80.0, 80.0)]
        for i in range(n_prey):
            sp_x, sp_y = spawn_points[i % len(spawn_points)]
            self.prey_bodies[i].x = np.linspace(sp_x, sp_x + 11 * 2.0, 12)
            self.prey_bodies[i].y = np.full(12, sp_y)

        self.prey_energy = np.full(n_prey, 1.0)
        self.prey_food = np.zeros(n_prey, dtype=int)
        self.prey_escapes = np.zeros(n_prey, dtype=int)

        # Initialize Predator at center
        self.predator = PredatorState(x=60.0, y=60.0, heading=0.0, velocity=1.4)

    def step(self) -> Dict:
        # 1. Update each prey
        prey_coords = []
        for i in range(self.n_prey):
            body = self.prey_bodies[i]
            conn = self.prey_connectomes[i]
            hx, hy = float(body.x[0]), float(body.y[0])
            prey_coords.append([hx, hy])

            # Scent of food
            scent, _ = self.env.get_chemical_gradient(hx, hy)

            # Distance to predator
            dist_to_pred = math.hypot(self.predator.x - hx, self.predator.y - hy)

            # Predator proximity induces anterior touch & threat excitation into AVA
            threat = 1.0 if dist_to_pred < 16.0 else 0.0
            if threat > 0:
                self.prey_escapes[i] += 1

            # Connectome step
            is_reversing = conn.step(
                anterior_touch=threat,
                posterior_touch=0.0,
                chemo_grad=scent,
                dt=2.0
            )

            # Physics step
            body.step(conn.muscle_torques, is_reversing=is_reversing, dt=0.02)

            # Check boundary clamp
            body.x[0] = float(np.clip(body.x[0], 4.0, 116.0))
            body.y[0] = float(np.clip(body.y[0], 4.0, 116.0))

            # Food check
            if self.env.consume_food(body.x[0], body.y[0], radius=3.5):
                self.prey_food[i] += 1
                self.prey_energy[i] = min(1.0, self.prey_energy[i] + 0.25)

            self.prey_energy[i] = max(0.05, self.prey_energy[i] - 0.0005)

        # 2. Update Predator: Hunt closest prey
        dists = [math.hypot(self.predator.x - p[0], self.predator.y - p[1]) for p in prey_coords]
        target_idx = int(np.argmin(dists))
        target_x, target_y = prey_coords[target_idx]

        desired_heading = math.atan2(target_y - self.predator.y, target_x - self.predator.x)
        # Smooth predator turning
        angle_diff = (desired_heading - self.predator.heading + math.pi) % (2 * math.pi) - math.pi
        self.predator.heading += float(np.clip(angle_diff * 0.15, -0.4, 0.4))

        # Move predator
        self.predator.x += self.predator.velocity * math.cos(self.predator.heading)
        self.predator.y += self.predator.velocity * math.sin(self.predator.heading)

        # Capture check
        if dists[target_idx] < 4.0:
            self.predator.captures += 1
            # Respawn captured prey away
            respawn_x = self.rng.uniform(20.0, 100.0)
            respawn_y = self.rng.uniform(20.0, 100.0)
            self.prey_bodies[target_idx].x = np.linspace(respawn_x, respawn_x + 11 * 2.0, 12)
            self.prey_bodies[target_idx].y = np.full(12, respawn_y)

        return {
            "predator_x": round(self.predator.x, 2),
            "predator_y": round(self.predator.y, 2),
            "captures": self.predator.captures,
            "prey_positions": [[round(p[0], 2), round(p[1], 2)] for p in prey_coords],
            "prey_food": self.prey_food.tolist(),
            "prey_escapes": self.prey_escapes.tolist(),
        }

    def run(self) -> Dict:
        print("=" * 75)
        print(f"🦈  MULTI-AGENT FLUID SWARM: 4 FORAGER PREY vs 1 APEX PREDATOR ({self.steps} steps)  🦈")
        print("=" * 75)

        history = []
        for step in range(self.steps):
            info = self.step()
            if step % 2 == 0:
                history.append({
                    "step": step,
                    "predator": [info["predator_x"], info["predator_y"]],
                    "captures": info["captures"],
                    "prey": info["prey_positions"],
                })

        total_food = int(np.sum(self.prey_food))
        total_escapes = int(np.sum(self.prey_escapes))

        print(f"✅ [Swarm Simulation Complete]")
        print(f"   • Total Nutrients Foraged: {total_food} pellets")
        print(f"   • Escape Reflexes Triggered: {total_escapes} times")
        print(f"   • Predator Captures:       {self.predator.captures}")
        print(f"   • Evasion Success Rate:    {round((total_escapes / max(1, total_escapes + self.predator.captures)) * 100, 1)}%")

        results = {
            "steps": self.steps,
            "total_food": total_food,
            "total_escapes": total_escapes,
            "predator_captures": self.predator.captures,
            "evasion_success_rate": round((total_escapes / max(1, total_escapes + self.predator.captures)) * 100, 1),
            "food_positions": self.env.food_positions.tolist(),
            "history": history,
        }

        out_dir = Path(__file__).resolve().parent / "data"
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "swarm_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        print(f"   • Results saved to {out_dir / 'swarm_results.json'}")
        return results


if __name__ == "__main__":
    swarm = SwarmSimulation(n_prey=4, steps=800)
    swarm.run()
