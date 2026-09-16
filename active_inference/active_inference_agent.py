"""
active_inference_agent.py - Active Inference & Predictive Processing Connectome.

Implements Karl Friston's Free Energy Principle (FEP) embodied in C. elegans:
  - Generative Model (A, B, C, D matrices)
  - Variational Free Energy minimization (F) for state estimation / perception
  - Expected Free Energy minimization (G) for action selection:
      G(u) = Pragmatic Value (Risk relative to C) + Epistemic Value (Information Gain / Ambiguity)
  - Connectome Descending Drive: Injects currents into AVB (forward crawl), AVA (pirouette reversal),
    and ASE (chemosensory gain).
"""

from __future__ import annotations

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


class ActiveInferenceGenerativeModel:
    """Discrete-state Active Inference Generative Model (POMDP formulation)."""

    def __init__(self, n_states: int = 5, n_obs: int = 5, n_actions: int = 3):
        # Hidden states: 0: CLEAR_OPEN, 1: GRADIENT_ASCENT, 2: NEAR_WALL, 3: TRAPPED_DEAD_END, 4: FEEDING_ZONE
        self.n_states = n_states
        # Observations: 0: LOW_SCENT, 1: HIGH_SCENT, 2: WHISKER_HIT, 3: FOOD_CONTACT, 4: STAGNANT_ODOR
        self.n_obs = n_obs
        # Actions: 0: FORWARD_CRAWL, 1: GRADIENT_STEER, 2: REVERSAL_PIROUETTE
        self.n_actions = n_actions

        # A matrix: P(o | s) (Likelihood) - shape (n_obs, n_states)
        self.A = np.array([
            # CLEAR, GRAD_ASC, NEAR_WALL, TRAPPED, FEEDING
            [0.75,   0.10,     0.30,      0.10,    0.05],  # 0: LOW_SCENT
            [0.15,   0.75,     0.15,      0.30,    0.15],  # 1: HIGH_SCENT
            [0.05,   0.05,     0.50,      0.50,    0.05],  # 2: WHISKER_HIT
            [0.01,   0.05,     0.01,      0.01,    0.70],  # 3: FOOD_CONTACT
            [0.04,   0.05,     0.04,      0.09,    0.05],  # 4: STAGNANT_ODOR
        ])
        # Normalize columns
        self.A = self.A / np.sum(self.A, axis=0, keepdims=True)

        # B matrix: P(s_t+1 | s_t, u) - shape (n_states, n_states, n_actions)
        self.B = np.zeros((n_states, n_states, n_actions))
        for u in range(n_actions):
            if u == 0:  # FORWARD_CRAWL
                self.B[:, :, u] = np.array([
                    [0.6, 0.1, 0.1, 0.0, 0.0],
                    [0.2, 0.6, 0.1, 0.0, 0.1],
                    [0.2, 0.2, 0.5, 0.2, 0.0],
                    [0.0, 0.1, 0.3, 0.8, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.9],
                ])
            elif u == 1:  # GRADIENT_STEER
                self.B[:, :, u] = np.array([
                    [0.4, 0.2, 0.1, 0.0, 0.0],
                    [0.4, 0.7, 0.1, 0.0, 0.1],
                    [0.1, 0.1, 0.3, 0.1, 0.0],
                    [0.1, 0.0, 0.5, 0.8, 0.0],
                    [0.0, 0.0, 0.0, 0.1, 0.9],
                ])
            elif u == 2:  # REVERSAL_PIROUETTE (breaks out of trapped states!)
                self.B[:, :, u] = np.array([
                    [0.8, 0.3, 0.7, 0.8, 0.1],  # Escapes back into CLEAR_OPEN
                    [0.1, 0.4, 0.1, 0.1, 0.1],
                    [0.05, 0.1, 0.1, 0.05, 0.0],
                    [0.05, 0.1, 0.1, 0.05, 0.0],
                    [0.0, 0.1, 0.0, 0.0, 0.8],
                ])
            # Normalize columns
            self.B[:, :, u] = self.B[:, :, u] / np.sum(self.B[:, :, u], axis=0, keepdims=True)

        # C vector: ln P(o) (Prior preferences: loves food contact, hates obstacle hits & stagnation)
        self.C = np.array([0.0, 1.5, -2.5, 4.0, -1.5])
        self.C_prob = np.exp(self.C) / np.sum(np.exp(self.C))

        # D vector: P(s_0)
        self.D = np.array([0.5, 0.2, 0.1, 0.1, 0.1])
        self.belief = self.D.copy()

    def update_belief(self, obs_idx: int, prev_action: int) -> Tuple[np.ndarray, float]:
        """Perceptual Inference: update hidden state beliefs by minimizing Free Energy F."""
        # Prior prediction: s_pred = B(u) * s_prev
        s_pred = self.B[:, :, prev_action] @ self.belief

        # Likelihood of observation: A[obs_idx, :]
        likelihood = self.A[obs_idx, :]

        # Posterior belief q(s) \propto likelihood * s_pred
        unnorm = likelihood * s_pred
        q_s = unnorm / (np.sum(unnorm) + 1e-12)

        # Variational Free Energy F = D_KL(q(s) || s_pred) - ln P(o | s)
        kl = np.sum(q_s * np.log((q_s + 1e-12) / (s_pred + 1e-12)))
        log_like = np.sum(q_s * np.log(likelihood + 1e-12))
        free_energy = float(kl - log_like)

        self.belief = q_s
        return q_s, free_energy

    def compute_expected_free_energy(self) -> np.ndarray:
        """Computes Expected Free Energy G(u) for each candidate action."""
        G = np.zeros(self.n_actions)

        for u in range(self.n_actions):
            # Predicted future state: q(s_next | u) = B(u) @ belief
            q_s_next = self.B[:, :, u] @ self.belief

            # Predicted observation distribution: q(o | u) = A @ q_s_next
            q_o_next = self.A @ q_s_next

            # 1. Pragmatic Value: D_KL(q(o|u) || P(o)) = sum q(o) ln(q(o) / P(o))
            pragmatic_risk = np.sum(q_o_next * np.log((q_o_next + 1e-12) / (self.C_prob + 1e-12)))

            # 2. Epistemic Value: Ambiguity = sum_s q(s) H(A[:, s])
            # Information gain is negative ambiguity (lower entropy in observation likelihood)
            ambiguity = 0.0
            for s in range(self.n_states):
                p_o_given_s = self.A[:, s]
                ambiguity += q_s_next[s] * (-np.sum(p_o_given_s * np.log(p_o_given_s + 1e-12)))

            # Expected Free Energy: G(u) = Pragmatic Risk + Ambiguity
            G[u] = pragmatic_risk + 0.5 * ambiguity

        return G

    def select_action(self, gamma: float = 2.0) -> int:
        """Selects action via softmax over negative Expected Free Energy: P(u) \propto exp(-gamma * G)."""
        G = self.compute_expected_free_energy()
        # Lower G is better
        logits = -gamma * G
        probs = np.exp(logits - np.max(logits))
        probs = probs / np.sum(probs)
        return int(np.random.choice(self.n_actions, p=probs))


class ActiveInferenceConnectomeAgent:
    """Embodied Connectome Organism driven by Active Inference."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.env = FluidArenaEnvironment(width=100.0, height=100.0, n_food=8, seed=seed)
        self.connectome = CelegansConnectome(n_segments=12)
        self.body = WormBodyPhysics(n_segments=12, segment_length=2.2)

        # Center worm
        self.body.x = np.linspace(45.0, 45.0 + 11 * 2.2, 12)
        self.body.y = np.full(12, 50.0)

        self.fep = ActiveInferenceGenerativeModel()
        self.prev_action = 0
        self.food_eaten = 0
        self.collisions = 0
        self.total_distance = 0.0
        self.free_energy_history = []

    def step(self, dt_phys: float = 0.02) -> Dict:
        hx = float(self.body.x[0])
        hy = float(self.body.y[0])

        scent, min_food_dist = self.env.get_chemical_gradient(hx, hy)
        whiskers = self.env.sample_whiskers(hx, hy, float(self.body.heading))
        min_whisker = min(whiskers)

        # Discretize continuous sensations into observation index:
        # 0: LOW_SCENT, 1: HIGH_SCENT, 2: WHISKER_HIT, 3: FOOD_CONTACT, 4: STAGNANT_ODOR
        if min_food_dist < 3.5:
            obs_idx = 3  # FOOD_CONTACT
        elif min_whisker < 4.5:
            obs_idx = 2  # WHISKER_HIT
        elif scent > 0.15:
            obs_idx = 1  # HIGH_SCENT
        elif abs(scent - 0.05) < 0.01:
            obs_idx = 4  # STAGNANT_ODOR
        else:
            obs_idx = 0  # LOW_SCENT

        # 1. Update Beliefs (Perception) & compute Free Energy F
        belief, free_energy = self.fep.update_belief(obs_idx, self.prev_action)
        self.free_energy_history.append(free_energy)

        # 2. Select Action minimizing Expected Free Energy G(u)
        action = self.fep.select_action(gamma=2.5)
        self.prev_action = action

        # 3. Translate Active Inference Action into Descending Connectome Currents
        # 0: FORWARD_CRAWL -> AVB excitation
        # 1: GRADIENT_STEER -> AVB + head steer bias
        # 2: REVERSAL_PIROUETTE -> AVA excitation (reversal wave!)
        if action == 0:
            i_avb = 14.0
            i_ava = 0.0
            turn_bias = 0.0
        elif action == 1:
            i_avb = 16.0
            i_ava = 0.0
            turn_bias = 0.4
        else:  # REVERSAL
            i_avb = 0.0
            i_ava = 26.0  # triggers AVA reversal escape wave!
            turn_bias = -0.6

        self.connectome.AVB.V = float(np.clip(self.connectome.AVB.V + i_avb * 0.15, -80.0, 20.0))
        self.connectome.AVA.V = float(np.clip(self.connectome.AVA.V + i_ava * 0.25, -80.0, 20.0))

        # Step biophysical connectome
        is_reversing = self.connectome.step(
            anterior_touch=1.0 if obs_idx == 2 or action == 2 else 0.0,
            posterior_touch=0.0,
            chemo_grad=scent,
            dt=2.0
        )

        if abs(turn_bias) > 0.01:
            self.connectome.muscle_torques[0] = float(np.clip(self.connectome.muscle_torques[0] + turn_bias * 0.5, -1.0, 1.0))

        # Physics step
        self.body.step(self.connectome.muscle_torques, is_reversing=is_reversing, dt=dt_phys)

        # Collision check
        new_hx, new_hy = float(self.body.x[0]), float(self.body.y[0])
        collided = not (2.0 <= new_hx <= 98.0 and 2.0 <= new_hy <= 98.0)
        for obs in self.env.obstacles:
            if obs[0] <= new_hx <= obs[2] and obs[1] <= new_hy <= obs[3]:
                collided = True

        if collided:
            self.collisions += 1
            self.body.heading += math.pi * 0.7
        else:
            self.total_distance += abs(float(self.body.velocity)) * dt_phys * 8.0

        if self.env.consume_food(new_hx, new_hy, radius=3.5):
            self.food_eaten += 1

        return {
            "head_x": round(new_hx, 2),
            "head_y": round(new_hy, 2),
            "action": action,
            "free_energy": round(free_energy, 3),
            "is_reversing": bool(is_reversing),
            "food_eaten": self.food_eaten,
            "collisions": self.collisions,
        }
