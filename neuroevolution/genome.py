"""
genome.py - Evolvable Connectome Genome with Synaptic & Biophysical Parameters.

Defines an evolvable 8-node canonical recurrent connectome:
  0: SENS_HEAD (sensory chemo/touch)
  1: CMD_FWD   (forward command)
  2: CMD_REV   (reversal command)
  3: MOT_DORS  (dorsal motor)
  4: MOT_VENT  (ventral motor)
  5: INH_DORS  (dorsal cross-inhibitor)
  6: INH_VENT  (ventral cross-inhibitor)
  7: PROPRIO   (body curvature feedback)
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import numpy as np


@dataclass
class ConnectomeGenome:
    """Genetic representation of a minimal locomotor connectome."""
    n_nodes: int = 8
    W: np.ndarray = field(default=None)        # Synaptic weights [-3.0, +3.0]
    tau: np.ndarray = field(default=None)      # Membrane time constants [5.0, 35.0] ms
    bias: np.ndarray = field(default=None)     # Tonic bias drive [-2.0, +2.0]
    fitness: float = -1e9
    generation: int = 0

    def __post_init__(self):
        if self.W is None:
            # Initialize with small random weights (zero prior knowledge)
            self.W = np.random.normal(0.0, 0.5, size=(self.n_nodes, self.n_nodes))
        if self.tau is None:
            self.tau = np.random.uniform(10.0, 30.0, size=self.n_nodes)
        if self.bias is None:
            self.bias = np.random.normal(0.0, 0.3, size=self.n_nodes)

    def mutate(self, rate: float = 0.20, scale: float = 0.35) -> ConnectomeGenome:
        """Create a mutated clone with structural/parametric changes."""
        child_w = self.W.copy()
        mask_w = np.random.random(child_w.shape) < rate
        child_w[mask_w] += np.random.normal(0.0, scale, size=np.sum(mask_w))
        child_w = np.clip(child_w, -4.0, 4.0)

        # Mutate time constants
        child_tau = self.tau.copy()
        mask_tau = np.random.random(child_tau.shape) < rate
        child_tau[mask_tau] += np.random.normal(0.0, scale * 5.0, size=np.sum(mask_tau))
        child_tau = np.clip(child_tau, 5.0, 40.0)

        # Mutate bias
        child_bias = self.bias.copy()
        mask_b = np.random.random(child_bias.shape) < rate
        child_bias[mask_b] += np.random.normal(0.0, scale, size=np.sum(mask_b))
        child_bias = np.clip(child_bias, -3.0, 3.0)

        return ConnectomeGenome(
            n_nodes=self.n_nodes,
            W=child_w,
            tau=child_tau,
            bias=child_bias,
            generation=self.generation + 1
        )

    @classmethod
    def crossover(cls, parent_a: ConnectomeGenome, parent_b: ConnectomeGenome) -> ConnectomeGenome:
        """Uniform crossover between two parent genomes."""
        mask_w = np.random.random(parent_a.W.shape) < 0.5
        child_w = np.where(mask_w, parent_a.W, parent_b.W)

        mask_tau = np.random.random(parent_a.tau.shape) < 0.5
        child_tau = np.where(mask_tau, parent_a.tau, parent_b.tau)

        mask_b = np.random.random(parent_a.bias.shape) < 0.5
        child_b = np.where(mask_b, parent_a.bias, parent_b.bias)

        return cls(
            n_nodes=parent_a.n_nodes,
            W=child_w,
            tau=child_tau,
            bias=child_b,
            generation=max(parent_a.generation, parent_b.generation) + 1
        )
