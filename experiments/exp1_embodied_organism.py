"""
exp1_embodied_organism.py - Experiment 1: Embodied Closed-Loop Virtual Organism in Continuous Fluid Arena.

Connects the biological whole-brain model to a physical continuous 2D environment:
  - Bilateral chemical sensors (left/right chemotaxis) & proximity touch whiskers.
  - Fluid hydrodynamic drag, inertia, boundary obstacles, and diffusing food sources.
  - Closed-loop neurobiology:
      * Hypothalamus tracks metabolic glucose (hunger amplifies foraging drive).
      * Thalamus gates sensory streams based on top-down attention.
      * Basal Ganglia selects locomotive actions via dopamine RPE.
      * Cerebellum learns an inverse dynamics model to cancel fluid drift and smooth turns.
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from brainmodel.brain import Brain
from brainmodel.hardware_governor import HardwareGovernor


@dataclass
class OrganismState:
    x: float = 50.0
    y: float = 50.0
    heading: float = 0.0          # radians
    velocity: float = 1.0
    glucose: float = 1.0          # 1.0 sated .. 0.0 starving
    food_eaten: int = 0
    collisions: int = 0
    total_distance: float = 0.0
    total_energy_expended: float = 0.0


class FluidArenaEnvironment:
    """2D continuous fluid environment with diffusing food pellets and obstacle walls."""

    def __init__(self, width: float = 100.0, height: float = 100.0, n_food: int = 8, seed: int = 42):
        self.width = width
        self.height = height
        self.rng = np.random.default_rng(seed)
        self.food_positions = self.rng.uniform(10.0, 90.0, size=(n_food, 2))
        self.obstacles = [
            # Rectangular bounding walls and central obstacles: [x_min, y_min, x_max, y_max]
            [40.0, 40.0, 60.0, 45.0],
            [40.0, 55.0, 60.0, 60.0],
        ]

    def get_chemical_gradient(self, x: float, y: float) -> Tuple[float, float]:
        """Compute chemical scent intensity at left and right antennae (sensor offset d=2.0)."""
        # Distance to all food pellets
        diffs = self.food_positions - np.array([x, y])
        dists = np.linalg.norm(diffs, axis=1)
        # Intensity is sum of inverse square distances
        intensities = 1.0 / (dists ** 1.5 + 1.0)
        total_intensity = float(np.sum(intensities))
        return total_intensity, float(np.min(dists))

    def sample_whiskers(self, x: float, y: float, heading: float, max_range: float = 12.0) -> List[float]:
        """Cast 3 proximity whiskers: left (-35 deg), center (0 deg), right (+35 deg)."""
        angles = [heading - math.radians(35), heading, heading + math.radians(35)]
        whisker_dists = []

        for ang in angles:
            hit_dist = max_range
            # Check boundary walls
            dx = math.cos(ang)
            dy = math.sin(ang)

            # Raycast distance to arena boundaries
            if dx > 1e-4:
                hit_dist = min(hit_dist, (self.width - x) / dx)
            elif dx < -1e-4:
                hit_dist = min(hit_dist, -x / dx)

            if dy > 1e-4:
                hit_dist = min(hit_dist, (self.height - y) / dy)
            elif dy < -1e-4:
                hit_dist = min(hit_dist, -y / dy)

            # Check interior obstacles
            for obs in self.obstacles:
                # Approximate obstacle box boundary hit
                if obs[0] <= x + dx * 5.0 <= obs[2] and obs[1] <= y + dy * 5.0 <= obs[3]:
                    hit_dist = min(hit_dist, 5.0)

            whisker_dists.append(max(0.0, min(max_range, hit_dist)))

        return whisker_dists

    def consume_food(self, x: float, y: float, radius: float = 3.5) -> bool:
        """Check if organism reached any food pellet; respawn consumed food elsewhere."""
        dists = np.linalg.norm(self.food_positions - np.array([x, y]), axis=1)
        hit_idx = np.where(dists < radius)[0]
        if len(hit_idx) > 0:
            for idx in hit_idx:
                self.food_positions[idx] = self.rng.uniform(10.0, 90.0, size=2)
            return True
        return False


class EmbodiedOrganismExperiment:
    """Orchestrates Experiment 1: Embodied Closed-Loop Organism in fluid simulation."""

    def __init__(self, steps: int = 1000, seed: int = 42):
        self.steps = steps
        self.env = FluidArenaEnvironment(seed=seed)
        self.governor = HardwareGovernor(min_headroom_gb=10.0)
        self.brain = Brain(n_actions=3, rng=np.random.default_rng(seed), governor=self.governor)
        self.state = OrganismState()
        self.trajectory_log: List[Dict] = []

    def run(self) -> Dict:
        """Run the complete embodied closed-loop simulation."""
        print(f"\n🌊 [Experiment 1] Launching Embodied Closed-Loop Organism ({self.steps} physical steps)...")

        dt = 0.1  # physical simulation delta t
        for step in range(self.steps):
            # 1. Sense environment
            scent_center, min_food_dist = self.env.get_chemical_gradient(self.state.x, self.state.y)
            # Bilateral antennae scent
            left_x = self.state.x + 2.5 * math.cos(self.state.heading - 0.5)
            left_y = self.state.y + 2.5 * math.sin(self.state.heading - 0.5)
            right_x = self.state.x + 2.5 * math.cos(self.state.heading + 0.5)
            right_y = self.state.y + 2.5 * math.sin(self.state.heading + 0.5)

            scent_left, _ = self.env.get_chemical_gradient(left_x, left_y)
            scent_right, _ = self.env.get_chemical_gradient(right_x, right_y)
            whiskers = self.env.sample_whiskers(self.state.x, self.state.y, self.state.heading)
            min_whisker = min(whiskers)

            # 2. Package into biological sensory world dict
            threat_level = float(np.clip((6.0 - min_whisker) / 6.0, 0.0, 1.0))
            food_cue = float(np.clip(scent_center * 10.0, 0.0, 1.0))

            world_inputs = {
                "sensory": {
                    "visual": food_cue,
                    "somatosensory": threat_level,
                    "auditory": 0.05,
                },
                "saliency": max(food_cue, threat_level),
                "threat": threat_level,
                "stakes": 0.5 + 0.5 * (1.0 - self.state.glucose),
                "true_rewards": [0.3, 0.7, 0.2] if food_cue > 0.4 else [0.5, 0.2, 0.1],
                "food": 1.0 if self.env.consume_food(self.state.x, self.state.y) else 0.0,
            }

            # 3. Brain step: advance coupled neuromodulators, limbic emotion, and action selection
            brain_out = self.brain.step(world_inputs)
            chosen_action = brain_out["chosen"]

            # 4. Action interpretation & Cerebellar Motor Refinement
            # Action 0: Straight swim, Action 1: Steer toward gradient, Action 2: Rapid obstacle avoidance turn
            if threat_level > 0.4:
                desired_turn = 1.2 if whiskers[0] > whiskers[2] else -1.2
                target_speed = 0.8
            elif chosen_action == 1:
                # Steer toward higher scent
                gradient_bias = math.atan2(scent_left - scent_right, 0.05)
                desired_turn = float(np.clip(gradient_bias * 0.8, -0.6, 0.6))
                target_speed = 1.4
            elif chosen_action == 2:
                # Reversal / pirouette
                desired_turn = 0.9
                target_speed = 0.6
            else:
                desired_turn = 0.05 * (scent_left - scent_right)
                target_speed = 1.8

            # Cerebellum refines motor command to counter hydrodynamic drag and fluid swirl
            fluid_drag = -0.12 * self.state.velocity
            refined_turn, motor_error = self.brain.cerebellum.refine(
                command=desired_turn,
                desired_correction=fluid_drag * 0.3,
            )

            # 5. Physics integration
            self.state.heading = (self.state.heading + refined_turn * dt) % (2 * math.pi)
            self.state.velocity = float(np.clip(self.state.velocity + (target_speed - self.state.velocity) * 0.2, 0.1, 2.5))

            # Apply position update
            new_x = self.state.x + self.state.velocity * math.cos(self.state.heading) * dt * 5.0
            new_y = self.state.y + self.state.velocity * math.sin(self.state.heading) * dt * 5.0

            # Collision check
            collision = False
            if not (2.0 <= new_x <= self.env.width - 2.0 and 2.0 <= new_y <= self.env.height - 2.0):
                collision = True
            for obs in self.env.obstacles:
                if obs[0] <= new_x <= obs[2] and obs[1] <= new_y <= obs[3]:
                    collision = True

            if collision:
                self.state.collisions += 1
                self.state.heading = (self.state.heading + math.pi * 0.7) % (2 * math.pi)
                self.state.velocity *= 0.3
            else:
                dist_moved = math.sqrt((new_x - self.state.x)**2 + (new_y - self.state.y)**2)
                self.state.total_distance += dist_moved
                self.state.x = new_x
                self.state.y = new_y

            # Metabolic glucose depletion & food intake
            energy_cost = 0.001 * (self.state.velocity ** 2)
            self.state.total_energy_expended += energy_cost
            self.state.glucose = max(0.05, self.state.glucose - energy_cost)

            # Food consumption
            if self.env.consume_food(self.state.x, self.state.y):
                self.state.food_eaten += 1
                self.state.glucose = min(1.0, self.state.glucose + 0.35)
                self.brain.nm.dopamine.phasic += 1.2  # Dopamine reward burst on food find

            # Log trajectory snapshot (every 2 steps)
            if step % 2 == 0:
                self.trajectory_log.append({
                    "step": step,
                    "x": round(self.state.x, 2),
                    "y": round(self.state.y, 2),
                    "heading": round(self.state.heading, 3),
                    "velocity": round(self.state.velocity, 2),
                    "glucose": round(self.state.glucose, 3),
                    "dopamine": round(brain_out["dopamine"], 2),
                    "cerebellar_error": round(motor_error, 4),
                    "emotion": brain_out["primary_emotion"],
                    "valence": round(brain_out["valence"], 2),
                    "whisker_min": round(min_whisker, 2),
                    "food_count": self.state.food_eaten,
                })

        # Summary performance metrics
        summary = {
            "steps": self.steps,
            "food_eaten": self.state.food_eaten,
            "total_distance": round(self.state.total_distance, 2),
            "collisions": self.state.collisions,
            "foraging_efficiency": round(self.state.food_eaten / (self.state.total_distance / 100.0 + 1e-4), 2),
            "cost_of_transport": round(self.state.total_energy_expended / (self.state.total_distance + 1e-4) * 1000.0, 3),
            "final_glucose": round(self.state.glucose, 3),
            "mean_cerebellar_error": round(float(np.mean([d["cerebellar_error"] for d in self.trajectory_log])), 4),
            "food_positions": self.env.food_positions.tolist(),
            "obstacles": self.env.obstacles,
        }

        # Save trajectory for visualizer
        data_dir = Path(__file__).resolve().parent / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        with open(data_dir / "exp1_trajectory.json", "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "trajectory": self.trajectory_log}, f, indent=2)

        print(f"✅ [Experiment 1 Complete]")
        print(f"   • Food Pellets Eaten:    {summary['food_eaten']}")
        print(f"   • Foraging Efficiency:   {summary['foraging_efficiency']} food/100m")
        print(f"   • Collisions:            {summary['collisions']}")
        print(f"   • Cost of Transport:     {summary['cost_of_transport']} energy/m")
        print(f"   • Cerebellar Error:      {summary['mean_cerebellar_error']} (Trajectory smoothed)")

        return summary


if __name__ == "__main__":
    exp = EmbodiedOrganismExperiment(steps=1000)
    exp.run()
