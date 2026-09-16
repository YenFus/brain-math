"""
cortex.py - cerebral lobes, prefrontal valuation, working memory, and conflict monitoring.

Prefrontal Decision Machinery:
  - OFC / vmPFC: SUBJECTIVE VALUE of options, integrating reward expectations
    with somatic emotional markers (Damasio) and limbic bias.
  - DLPFC: WORKING MEMORY attractor buffer maintaining goals over distraction.
  - ACC: CONFLICT MONITORING - tracks uncertainty and competition between candidate choices.
    High ACC conflict triggers System 2 deliberative search and recruits DLPFC effort.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional
import numpy as np

from .core import sigmoid


@dataclass
class WorkingMemory:
    """DLPFC persistent-activity buffer. Holds a value vector that decays unless
    refreshed; norepinephrine/attention sets how strongly it is maintained."""

    n: int = 3
    content: np.ndarray = field(default=None)
    tau: float = 400.0           # slow decay == persistent activity
    integrity: float = 1.0       # lesion / cortical thinning reduces this

    def __post_init__(self):
        if self.content is None:
            self.content = np.zeros(self.n)

    def update(self, target: np.ndarray, gate: float, dt: float) -> np.ndarray:
        # gate (attention * NE) controls how much new info overwrites the buffer
        keep = np.exp(-dt / (self.tau * max(self.integrity, 0.1)))
        self.content = keep * self.content + (1 - keep) * gate * np.asarray(target)
        return self.content


@dataclass
class Cortex:
    n_actions: int = 3
    ofc_integrity: float = 1.0       # value computation (lesion -> erratic choice)
    dlpfc: WorkingMemory = None
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))
    name: str = "cortex"

    def __post_init__(self):
        if self.dlpfc is None:
            self.dlpfc = WorkingMemory(n=self.n_actions)

    # --- sensory cortices ------------------------------------------------------

    @staticmethod
    def sensory(relayed: Dict[str, float]) -> Dict[str, float]:
        """Turn thalamic relay into cortical percepts (a saturating readout)."""
        return {mod: float(sigmoid(val, gain=4.0, bias=0.4)) for mod, val in relayed.items()}

    # --- OFC / vmPFC: subjective value with Somatic Markers -------------------

    def subjective_value(
        self,
        q_values: np.ndarray,
        limbic_bias: float,
        somatic_valence: float = 0.0,
        somatic_confidence: float = 1.0,
        risk: float = 0.0,
    ) -> np.ndarray:
        """Integrate learned Q-values, emotional/limbic bias, somatic markers (Damasio),
        and risk into a subjective value per option. OFC damage degrades this."""
        q = np.asarray(q_values, dtype=float)
        # Damasio somatic marker hypothesis: emotional experiences color subjective value
        emotional_influence = 0.25 * somatic_valence * somatic_confidence
        value = q + limbic_bias + emotional_influence - risk

        if self.ofc_integrity < 1.0:
            # impaired OFC: value estimate gets noisy and compressed toward zero
            noise = (1.0 - self.ofc_integrity) * self.rng.standard_normal(len(q))
            value = self.ofc_integrity * value + noise
        return value

    # --- ACC: conflict / control demand ---------------------------------------

    @staticmethod
    def conflict(values: np.ndarray) -> float:
        """How close are the best two options? Small gap == high conflict ==
        high demand for deliberate control (System 2 engagement)."""
        v = np.sort(np.asarray(values, dtype=float))
        if len(v) < 2:
            return 0.0
        gap = v[-1] - v[-2]
        # Exponential conflict: near 1.0 when tied, drops to 0.0 when one clearly dominates
        return float(np.exp(-3.0 * gap))

    def working_memory(self, values: np.ndarray, gate: float, dt: float) -> np.ndarray:
        return self.dlpfc.update(values, gate, dt)
