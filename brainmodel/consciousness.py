"""
consciousness.py - access and (toy) integrated information.

Two frameworks from the document, both deliberately reduced to something
computable. Neither claims to *solve* consciousness; they make its two leading
theories concrete enough to run.

1. Global Workspace / ignition (Dehaene)
----------------------------------------
Information competes in a sensory buffer that decays fast (the document's ~300ms
degradation, ~700ms loss). When accumulated activation for an item crosses a
critical threshold it *ignites*: a self-amplifying, all-or-none "broadcast" that
makes the content globally available (== conscious). Modeled as a bistable unit:

    tau da/dt = -a + phi(input + rec * a)        # recurrent self-excitation
    conscious  if  a > ignition_threshold

The recurrent term `rec*a` is what makes ignition non-linear and all-or-none:
sub-threshold inputs decay; supra-threshold inputs explode into the workspace.

2. Integrated information (Tononi, IIT) - toy Phi
-------------------------------------------------
Phi measures how much a system's cause-effect structure is *irreducible* - how
much is lost when you cut it into parts. We compute a small, honest proxy for a
binary system with a transition probability matrix T(u'|u): the effective
information across the minimum-information bipartition, using KL divergence
between the whole system's effect distribution and the product of its parts'.
This is computationally faithful in spirit but only tractable for a handful of
units - which is exactly the real-world limitation of IIT.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import numpy as np

from .core import Subsystem, sigmoid


@dataclass
class GlobalWorkspace(Subsystem):
    tau: float = 25.0
    rec: float = 2.2               # recurrent self-excitation (drives ignition)
    gain: float = 6.0
    bias: float = 0.7
    ignition_threshold: float = 0.5
    buffer_decay_tau: float = 200.0  # sensory buffer half-life (ms)
    a: float = 0.0                  # workspace activation
    buffer: float = 0.0
    name: str = "global_workspace"

    def reset(self):
        self.a = 0.0
        self.buffer = 0.0

    def step(self, dt: float, inputs):
        stimulus = float(inputs.get("stimulus", 0.0))
        attention = float(inputs.get("attention", 1.0))   # salience/TPN gate

        # fast-decaying sensory buffer
        self.buffer += dt * (-(self.buffer) / self.buffer_decay_tau + stimulus)
        drive = attention * self.buffer + self.rec * self.a
        target = sigmoid(drive, self.gain, self.bias)
        self.a += dt * (-self.a + target) / self.tau
        self.a = float(np.clip(self.a, 0.0, 1.0))

        conscious = self.a > self.ignition_threshold
        return {"workspace": self.a, "conscious": bool(conscious), "buffer": self.buffer}


def _entropy(probs: np.ndarray) -> float:
    p = np.clip(probs, 1e-12, 1.0)
    return float(-np.sum(p * np.log2(p)))


def integrated_information(T: np.ndarray, state_probs: np.ndarray = None) -> float:
    """Toy Phi for a binary system: the total correlation (multi-information) of
    the system's effect distribution.

    Parameters
    ----------
    T : (2**n, n) array. Row i gives the probability that each of the n units is
        ON at t+1 given the system is in state i at t (independent-channel form).
    state_probs : prior over the 2**n current states (default uniform).

    Integration is measured as how far the joint next-state distribution is from
    the product of its single-unit marginals:

        Phi ~ sum_j H(u'_j) - H(u'_1, ..., u'_n)

    This is exactly 0 when the units' futures are independent (a system that is
    *reducible* to its parts) and positive when the whole carries cause-effect
    structure the parts cannot - the computable heart of IIT's claim, honest
    about being tractable only for a handful of units.
    """
    n = T.shape[1]
    if state_probs is None:
        state_probs = np.full(2 ** n, 1.0 / 2 ** n)

    # joint distribution over the 2**n possible next states
    joint = np.zeros(2 ** n)
    for next_state in range(2 ** n):
        bits = [(next_state >> b) & 1 for b in range(n)]
        # P(next_state) = sum_u p(u) * prod_j P(u'_j = bit_j | u)
        per_state = np.ones(2 ** n)
        for j in range(n):
            p_on = T[:, j]
            per_state *= p_on if bits[j] else (1.0 - p_on)
        joint[next_state] = float(np.dot(state_probs, per_state))
    joint /= joint.sum()

    # marginal entropy of each unit's next state
    marginal_H = 0.0
    for j in range(n):
        p_on = float(np.dot(state_probs, T[:, j]))
        marginal_H += _entropy(np.array([p_on, 1.0 - p_on]))

    return max(0.0, marginal_H - _entropy(joint))
