"""
decision.py - value-based decision making.

Implements two complementary formalisms from the document.

1. Drift Diffusion Model (DDM)
------------------------------
Two-alternative choice as noisy evidence accumulation:

    dx = v * dt + sigma * sqrt(dt) * N(0,1)

starting at x = z, deciding when x reaches an upper (a) or lower (0) boundary.
Parameters are exactly the document's:
    v     drift rate    (evidence quality / clarity)
    a     threshold     (speed-accuracy trade-off)
    z     start point   (prior bias)
    t0    non-decision time (sensory encoding + motor)

Returns choice and reaction time. PFC quality and arousal raise `v`; cautious
states raise `a`.

2. Reinforcement learning == the 5-stage neuroeconomic loop
-----------------------------------------------------------
    representation -> valuation -> action selection -> outcome -> learning

Valuation holds Q-values per action. Action selection is softmax over Q (the
"choose highest subjective value" step). The learning step is temporal-difference:

    delta = r + gamma * max_a' Q(s',a') - Q(s,a)      # reward prediction error
    Q(s,a) <- Q(s,a) + alpha * delta

The key bridge in the document: **phasic dopamine == delta** (reward prediction
error). So this module emits `delta` as the dopamine burst that the rest of the
brain (basal ganglia gating, plasticity) reads.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

from .core import Subsystem


@dataclass
class DriftDiffusion(Subsystem):
    v: float = 0.3       # drift rate
    a: float = 1.0       # threshold separation
    z: float = 0.5       # relative start (0..1) -> absolute z*a
    t0: float = 100.0    # non-decision time (ms)
    sigma: float = 0.4   # noise scale
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))

    x: float = field(default=None)
    elapsed: float = 0.0
    decided: Optional[int] = None
    rt: Optional[float] = None
    name: str = "ddm"

    def reset(self):
        self.x = self.z * self.a
        self.elapsed = 0.0
        self.decided = None
        self.rt = None

    def step(self, dt: float, inputs):
        if self.x is None:
            self.reset()
        v = float(inputs.get("drift", self.v))
        if self.decided is None:
            self.x += v * dt + self.sigma * np.sqrt(dt) * self.rng.standard_normal()
            self.elapsed += dt
            if self.x >= self.a:
                self.decided, self.rt = 1, self.elapsed + self.t0
            elif self.x <= 0.0:
                self.decided, self.rt = 0, self.elapsed + self.t0
        return {"evidence": self.x, "choice": self.decided, "rt": self.rt}

    def simulate_choice(self, max_ms: float = 5000.0, dt: float = 1.0):
        self.reset()
        steps = int(max_ms / dt)
        for _ in range(steps):
            self.step(dt, {})
            if self.decided is not None:
                break
        return self.decided, self.rt


@dataclass
class ValueLearner(Subsystem):
    """Tabular TD(0) learner; emits reward-prediction-error as the dopamine signal."""

    n_actions: int = 3
    alpha: float = 0.2     # learning rate (modulated by dopamine availability)
    beta: float = 3.0      # softmax inverse temperature (exploration)
    gamma: float = 0.9     # discount
    Q: np.ndarray = field(default=None)
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))
    name: str = "value"

    def __post_init__(self):
        if self.Q is None:
            self.Q = np.zeros(self.n_actions)

    def valuation(self) -> np.ndarray:
        """Subjective values currently assigned to each action."""
        return self.Q.copy()

    def action_selection(self) -> int:
        z = self.beta * (self.Q - self.Q.max())
        p = np.exp(z) / np.exp(z).sum()
        return int(self.rng.choice(self.n_actions, p=p)), p

    def learn(self, action: int, reward: float, next_value: float = 0.0) -> float:
        """One outcome-evaluation + learning step. Returns prediction error (==DA burst)."""
        delta = reward + self.gamma * next_value - self.Q[action]
        self.Q[action] += self.alpha * delta
        return float(delta)
