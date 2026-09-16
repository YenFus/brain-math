"""
cerebellum.py - coordination, forward models, and supervised motor learning.

The document: the cerebellum "coordinates skeletal muscle activity, maintains
posture and equilibrium, and processes sensory feedback to execute fine, rapid
motor skills." Computationally it is the brain's *forward model / adaptive
filter*: it predicts the sensory consequence of a motor command and emits a
correction that cancels error, making movement smooth and accurate.

Circuit and plasticity
----------------------
  Mossy fibers (context) -> granule cells -> parallel fibers -> Purkinje cells
  Climbing fibers (inferior olive) -> Purkinje cells   == the ERROR signal

The Purkinje cell computes a correction as a weighted sum of parallel-fiber
activity. The cerebellum's signature plasticity is *parallel-fiber -> Purkinje
LTD*: when a climbing fiber signals error, the weights of co-active parallel
fibers are depressed. This is supervised learning by error:

    correction = w . x
    error      = desired_correction - correction        (climbing fiber)
    w          <- w + lr * error * x                     (PF-Purkinje plasticity)

A trained cerebellum drives motor error and tremor toward zero. Damage (ataxic
CP, cerebellar stroke) = noisy/incomplete weights -> uncorrected error, OVERSHOOT
(dysmetria) and INTENTION TREMOR, exactly the clinical picture.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class Cerebellum:
    n_inputs: int = 16           # parallel-fiber context dimensionality
    lr: float = 0.05             # PF-Purkinje learning rate (LTD-dominated)
    damage: float = 0.0          # 0 healthy .. 1 fully ataxic
    tremor_gain: float = 0.0     # intrinsic oscillatory noise when damaged
    w: np.ndarray = field(default=None)
    centers: np.ndarray = field(default=None)
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))
    _phase: float = 0.0
    name: str = "cerebellum"

    def __post_init__(self):
        if self.w is None:
            self.w = np.zeros(self.n_inputs)
        if self.centers is None:
            # granule-cell layer = fixed RBF tiling of command space, so the
            # Purkinje weights learn a SMOOTH, generalizable forward model.
            self.centers = np.linspace(-1.2, 1.2, self.n_inputs)

    def _context(self, command: float) -> np.ndarray:
        """Expand the motor command into parallel-fiber (granule-cell) activity
        as a bank of Gaussian radial basis functions tiling command space."""
        return np.exp(-((command - self.centers) ** 2) / (2 * 0.25 ** 2))

    def refine(self, command: float, desired_correction: float, learn: bool = True):
        """Apply the forward-model correction to a motor command and learn from
        the residual error. Returns (refined_command, residual_error)."""
        x = self._context(command)
        correction = float(self.w @ x)

        # damage corrupts the correction and injects intention tremor (grows with
        # movement amplitude, hence "intention" tremor)
        self._phase += 0.6
        tremor = self.tremor_gain * np.sin(self._phase) * abs(command)
        effective_corr = (1.0 - self.damage) * correction + tremor

        refined = command + effective_corr
        residual = desired_correction - effective_corr

        if learn:
            # climbing-fiber error drives PF->Purkinje plasticity; damage impairs it
            self.w += self.lr * (1.0 - self.damage) * residual * x

        return refined, float(residual)
