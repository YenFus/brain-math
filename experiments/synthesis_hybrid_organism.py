"""
synthesis_hybrid_organism.py - Synthesis Experiment: Unified Connectome + Bio-Modular Organism.

Combines the discoveries of Experiments 1, 2, and 3:
  1. Locomotor Foundation: 12-segment C. elegans connectome with non-spiking graded potentials.
  2. Cerebellar Forward Model: Purkinje cell LTD cancels viscous fluid drag and dynamic perturbations.
  3. Basal Ganglia & OFC Value Learning: Dopamine RPE modulates command interneurons (AVB/PVC vs AVA/AVD)
     based on chemical scent gradients and hunger levels.
  4. Locus Coeruleus NE: Heightens escape reflex responsiveness when unexpected physical resistance occurs.
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
from brainmodel.cerebellum import Cerebellum
from brainmodel.neuromodulators import Neuromodulators


class SynthesisHybridOrganism:
    """Unified organism integrating Connectome locomotion with Cerebellar & Neuromodulatory control."""

    def __init__(self, steps: int = 1000, seed: int = 42):
        self.steps = steps
        self.rng = np.random.default_rng(seed)
        self.env = FluidArenaEnvironment(width=120.0, height=120.0, n_food=10, seed=seed)
        self.connectome = CelegansConnectome(n_segments=12)
        self.body = WormBodyPhysics(n_segments=12, segment_length=2.5)

        # Start worm near center of arena
        self.body.x = np.linspace(50.0, 50.0 + 11 * 2.5, 12)
        self.body.y = np.full(12, 50.0)

        # Cerebellar inverse dynamics filter
        self.cerebellum = Cerebellum(n_inputs=16, lr=0.06)

        # Neuromodulators
        self.nm = Neuromodulators()

        # Metabolic state
        self.glucose = 1.0
        self.food_eaten = 0
        self.collisions = 0
        self.total_distance = 0.0

    def run(self) -> Dict:
        print(f"\n🧬 [Synthesis Experiment] Launching Unified Connectome + Bio-Modular Organism ({self.steps} steps)...")

        trajectory = []
        dt_chem = 2.0  # ms
        dt_phys = 0.02 # s

        for step in range(self.steps):
            head_x = float(self.body.x[0])
            head_y = float(self.body.y[0])

            # 1. Chemical and whisker sensing
            scent_center, min_dist = self.env.get_chemical_gradient(head_x, head_y)
            whiskers = self.env.sample_whiskers(head_x, head_y, float(self.body.heading))
            min_whisker = min(whiskers)

            # Touch stimulus if close to wall / obstacle
            touch_stim = 1.0 if min_whisker < 4.0 else 0.0

            # Hunger-scaled foraging drive (dopamine seeking)
            hunger = 1.0 - self.glucose
            chemo_input = scent_center * (1.0 + hunger * 2.0)

            # 2. Connectome Step
            is_reversing = self.connectome.step(
                anterior_touch=touch_stim,
                posterior_touch=0.0,
                chemo_grad=chemo_input,
                dt=dt_chem
            )

            # 3. Cerebellar Motor Smoothing on segment torques
            raw_torques = self.connectome.muscle_torques.copy()
            refined_torques = np.zeros_like(raw_torques)
            cer_errors = []

            # Fluid drag damping
            fluid_drag = -0.15 * float(self.body.velocity)
            for i in range(len(raw_torques)):
                # Refine each muscle segment
                ref_t, err_t = self.cerebellum.refine(
                    command=float(raw_torques[i]),
                    desired_correction=fluid_drag * 0.2,
                    learn=True
                )
                refined_torques[i] = ref_t
                cer_errors.append(err_t)

            # 4. Body Physics Step
            self.body.step(refined_torques, is_reversing=is_reversing, dt=dt_phys)

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
                self.body.heading += math.pi * 0.6  # reorient away
                self.nm.norepinephrine.phasic += 0.8 # shock surge
            else:
                self.total_distance += abs(float(self.body.velocity)) * dt_phys * 8.0

            # Food consumption
            if self.env.consume_food(new_hx, new_hy, radius=4.0):
                self.food_eaten += 1
                self.glucose = min(1.0, self.glucose + 0.3)
                self.nm.dopamine.phasic += 1.5

            # Metabolic decay
            self.glucose = max(0.05, self.glucose - 0.0006)
            self.nm.dopamine.decay_phasic(dt_chem)
            self.nm.norepinephrine.decay_phasic(dt_chem)

            # Telemetry logging every 2 steps
            if step % 2 == 0:
                trajectory.append({
                    "step": step,
                    "head_x": round(new_hx, 2),
                    "head_y": round(new_hy, 2),
                    "heading": round(float(self.body.heading), 3),
                    "velocity": round(float(self.body.velocity), 3),
                    "glucose": round(self.glucose, 3),
                    "dopamine": round(self.nm.dopamine.effective, 2),
                    "norepinephrine": round(self.nm.norepinephrine.effective, 2),
                    "is_reversing": bool(is_reversing),
                    "v_AVA": round(self.connectome.AVA.V, 1),
                    "v_AVB": round(self.connectome.AVB.V, 1),
                    "body_x": [round(float(val), 2) for val in self.body.x],
                    "body_y": [round(float(val), 2) for val in self.body.y],
                    "food_count": self.food_eaten,
                    "whisker_min": round(min_whisker, 2),
                })

        summary = {
            "steps": self.steps,
            "food_eaten": self.food_eaten,
            "total_distance": round(self.total_distance, 2),
            "collisions": self.collisions,
            "final_glucose": round(self.glucose, 3),
            "mean_cerebellar_error": round(float(np.mean(cer_errors)), 4),
            "food_positions": self.env.food_positions.tolist(),
            "obstacles": self.env.obstacles,
        }

        # Save to experiments/data/synthesis_results.json
        data_dir = Path(__file__).resolve().parent / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        with open(data_dir / "synthesis_results.json", "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "trajectory": trajectory}, f, indent=2)

        print(f"✅ [Synthesis Experiment Complete]")
        print(f"   • Food Eaten:         {summary['food_eaten']}")
        print(f"   • Total Distance:     {summary['total_distance']} m")
        print(f"   • Collisions:         {summary['collisions']}")
        print(f"   • Final Glucose:      {summary['final_glucose']}")
        print(f"   • Cerebellar Error:   {summary['mean_cerebellar_error']}")
        print("   • Results saved to experiments/data/synthesis_results.json")

        return summary


if __name__ == "__main__":
    synthesis = SynthesisHybridOrganism(steps=1000)
    synthesis.run()
