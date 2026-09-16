"""
evolver.py - Neuroevolutionary Engine evolving Connectome Topologies from Scratch.

Tests whether artificial evolution under foraging & locomotion pressure independently
discovers C. elegans biophysical motifs (reciprocal inhibition, forward/reverse command separation).
"""

from __future__ import annotations

import copy
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from neuroevolution.genome import ConnectomeGenome
from experiments.exp1_embodied_organism import FluidArenaEnvironment
from experiments.exp3_celegans_connectome import WormBodyPhysics


class ConnectomeEvaluator:
    """Evaluates a single genome's phenotype in physical continuous simulation."""

    def __init__(self, steps: int = 300, seed: int = 42):
        self.steps = steps
        self.seed = seed

    def evaluate(self, genome: ConnectomeGenome) -> float:
        env = FluidArenaEnvironment(width=100.0, height=100.0, n_food=6, seed=self.seed)
        body = WormBodyPhysics(n_segments=12, segment_length=2.2)

        # Center worm
        body.x = np.linspace(45.0, 45.0 + 11 * 2.2, 12)
        body.y = np.full(12, 50.0)

        # Neural potentials (8 nodes)
        V = np.zeros(genome.n_nodes)
        dt = 0.05  # 50ms integration tick

        dorsal_history = []
        ventral_history = []
        food_eaten = 0
        collisions = 0
        total_dist = 0.0

        for step in range(self.steps):
            hx = float(body.x[0])
            hy = float(body.y[0])

            # Scent and whisker inputs
            scent, _ = env.get_chemical_gradient(hx, hy)
            whiskers = env.sample_whiskers(hx, hy, float(body.heading))
            min_whisker = min(whiskers)

            # Proprioceptive average curvature
            proprio = float(np.mean(body.angles)) if len(body.angles) > 0 else 0.0

            # External input vector into 8 nodes
            I_ext = np.zeros(genome.n_nodes)
            I_ext[0] = scent * 5.0 - (5.0 if min_whisker < 4.0 else 0.0)  # SENS_HEAD
            I_ext[7] = proprio * 3.0                                       # PROPRIO

            # Graded synaptic activation: sigmoid
            act = 1.0 / (1.0 + np.exp(-V))

            # Membrane voltage derivative: tau * dV/dt = -V + W * act + bias + I_ext
            dV = (-V + genome.W @ act + genome.bias + I_ext) / genome.tau
            V += dV * (dt * 10.0)
            V = np.clip(V, -6.0, 6.0)

            # Motor output: Dorsal (node 3) vs Ventral (node 4)
            dorsal_act = act[3]
            ventral_act = act[4]
            dorsal_history.append(dorsal_act)
            ventral_history.append(ventral_act)

            # Segmental torque wave driven by evolved motors
            segment_torques = np.zeros(12)
            for i in range(12):
                phase_lag = i * 0.5
                segment_torques[i] = (dorsal_act - ventral_act) * math.cos(step * 0.2 - phase_lag)

            # Step body physics
            body.step(segment_torques, is_reversing=False, dt=0.02)

            # Collisions & distance
            new_hx, new_hy = float(body.x[0]), float(body.y[0])
            collided = not (2.0 <= new_hx <= 98.0 and 2.0 <= new_hy <= 98.0)
            for obs in env.obstacles:
                if obs[0] <= new_hx <= obs[2] and obs[1] <= new_hy <= obs[3]:
                    collided = True

            if collided:
                collisions += 1
                body.heading += math.pi * 0.7
            else:
                total_dist += abs(float(body.velocity)) * 0.02 * 8.0

            if env.consume_food(new_hx, new_hy, radius=3.5):
                food_eaten += 1

        # Anti-phase coordination bonus: correlation between dorsal and ventral activations
        # In functional locomotion, dorsal and ventral must be inversely correlated (corr < 0)
        corr = float(np.corrcoef(dorsal_history, ventral_history)[0, 1]) if len(dorsal_history) > 10 else 0.0
        rhythm_bonus = 30.0 if corr < -0.2 else (-15.0 if corr > 0.5 else 0.0)

        fitness = (total_dist * 2.5) + (food_eaten * 60.0) - (collisions * 8.0) + rhythm_bonus
        return float(fitness)


class NeuroevolutionExperiment:
    """Executes multi-generation connectome evolution."""

    def __init__(self, pop_size: int = 24, generations: int = 20, seed: int = 42):
        self.pop_size = pop_size
        self.generations = generations
        self.evaluator = ConnectomeEvaluator(steps=300, seed=seed)
        self.rng = np.random.default_rng(seed)

        # Initialize random population
        self.population = [ConnectomeGenome() for _ in range(pop_size)]

    def run(self) -> Dict:
        print("=" * 75)
        print("🧬  CONNECTOME NEUROEVOLUTION ENGINE: EVOLVING BRAINS FROM SCRATCH  🧬")
        print("=" * 75)
        print(f"Population: {self.pop_size} | Generations: {self.generations} | Prior Knowledge: NONE (Random Init)")

        history = []
        best_overall = None
        best_overall_fitness = -1e9

        for gen in range(self.generations):
            # 1. Evaluate fitness of population
            fitnesses = []
            for indiv in self.population:
                indiv.fitness = self.evaluator.evaluate(indiv)
                fitnesses.append(indiv.fitness)

            # Sort descending
            self.population.sort(key=lambda g: g.fitness, reverse=True)
            best_gen = self.population[0]
            mean_fitness = float(np.mean(fitnesses))

            if best_gen.fitness > best_overall_fitness:
                best_overall_fitness = best_gen.fitness
                best_overall = copy.deepcopy(best_gen)

            # Analyze evolved connectivity: Reciprocal inhibition between dorsal (3) and ventral (4)
            w_3_to_4 = best_gen.W[4, 3]  # effect of dorsal on ventral
            w_4_to_3 = best_gen.W[3, 4]  # effect of ventral on dorsal
            reciprocal_inhibition = bool(w_3_to_4 < 0 and w_4_to_3 < 0)

            print(f"Gen {gen:>2}: Best Fit: {best_gen.fitness:>6.1f} | Mean Fit: {mean_fitness:>6.1f} | W(D->V): {w_3_to_4:>5.2f} | W(V->D): {w_4_to_3:>5.2f} | Reciprocal Inh: {reciprocal_inhibition}")

            history.append({
                "generation": gen,
                "best_fitness": round(best_gen.fitness, 1),
                "mean_fitness": round(mean_fitness, 1),
                "w_dorsal_to_ventral": round(float(w_3_to_4), 3),
                "w_ventral_to_dorsal": round(float(w_4_to_3), 3),
                "has_reciprocal_inhibition": reciprocal_inhibition,
            })

            # 2. Next generation: Elite preservation (top 3) + Crossover + Mutation
            next_pop = [copy.deepcopy(self.population[i]) for i in range(3)]

            while len(next_pop) < self.pop_size:
                # Tournament selection of size 3
                idx_pool = self.rng.choice(len(self.population), size=3, replace=False)
                tournament = [self.population[i] for i in idx_pool]
                tournament.sort(key=lambda g: g.fitness, reverse=True)
                parent_a = tournament[0]

                if self.rng.random() < 0.65:
                    # Crossover with second candidate
                    idx_pool2 = self.rng.choice(len(self.population), size=3, replace=False)
                    parent_b = sorted([self.population[i] for i in idx_pool2], key=lambda g: g.fitness, reverse=True)[0]
                    child = ConnectomeGenome.crossover(parent_a, parent_b)
                else:
                    child = copy.deepcopy(parent_a)

                # Mutate
                child = child.mutate(rate=0.20, scale=0.40)
                next_pop.append(child)

            self.population = next_pop

        print("=" * 75)
        print(f"✅ [Neuroevolution Complete] Peak Fitness: {best_overall_fitness:.1f}")
        print(f"   • Best Genome Synaptic Matrix shape: {best_overall.W.shape}")
        print(f"   • Emerged Reciprocal Inhibition: {history[-1]['has_reciprocal_inhibition']}")

        results = {
            "generations": self.generations,
            "pop_size": self.pop_size,
            "peak_fitness": round(best_overall_fitness, 1),
            "history": history,
            "best_genome": {
                "W": best_overall.W.tolist(),
                "tau": best_overall.tau.tolist(),
                "bias": best_overall.bias.tolist(),
            }
        }

        out_dir = Path(__file__).resolve().parent / "data"
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "evolution_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        print(f"   • Results saved to {out_dir / 'evolution_results.json'}")
        return results


if __name__ == "__main__":
    exp = NeuroevolutionExperiment(pop_size=24, generations=15)
    exp.run()
