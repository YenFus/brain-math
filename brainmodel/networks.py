"""
networks.py - large-scale network dynamics (DMN / TPN / Salience).

Three coupled rate units reproduce the document's "triple network" and the
salience network's role as a *dynamic switch* between internally-focused rest
(Default Mode) and externally-focused task (Task-Positive):

    tau dD/dt = -D + phi( I_internal  - w_TD*T - w_SD*S )      Default Mode
    tau dT/dt = -T + phi( I_external  - w_DT*D + w_ST*S )      Task-Positive
    tau dS/dt = -S + phi( saliency_in + w_TS*T )               Salience

DMN and TPN are mutually inhibitory (anti-correlated, as observed). The Salience
node, driven by behaviorally relevant input, *suppresses DMN and boosts TPN* -
flipping the brain from rest to task. Remove salient input and the system relaxes
back to DMN dominance (mind-wandering).

Pathology hooks:
  - Hallucination / Alzheimer's: DMN fails to deactivate (w_SD too small) ->
    internal simulations are not suppressed during external input.
  - Acetylcholine sets the rest<->task baseline (cholinergic loss biases toward
    a degraded, poorly-switching regime).
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .core import Subsystem, sigmoid


@dataclass
class TripleNetwork(Subsystem):
    tau: float = 20.0
    gain: float = 4.0
    bias: float = 0.5

    # coupling strengths
    w_TD: float = 1.2   # TPN inhibits DMN
    w_DT: float = 1.2   # DMN inhibits TPN
    w_SD: float = 1.5   # Salience suppresses DMN (the switch)
    w_ST: float = 1.0   # Salience boosts TPN
    w_TS: float = 0.4   # TPN feeds Salience

    D: float = 0.7      # default-mode activity (rest dominates initially)
    T: float = 0.1      # task-positive activity
    S: float = 0.1      # salience activity
    name: str = "triple_network"

    def reset(self):
        self.D, self.T, self.S = 0.7, 0.1, 0.1

    def step(self, dt: float, inputs):
        I_internal = float(inputs.get("internal", 0.5))
        I_external = float(inputs.get("external", 0.0))
        saliency = float(inputs.get("saliency", 0.0))
        ach = float(inputs.get("acetylcholine", 1.0))  # arousal scales gain

        g = self.gain * ach
        D_t = sigmoid(I_internal - self.w_TD * self.T - self.w_SD * self.S, g, self.bias)
        T_t = sigmoid(I_external - self.w_DT * self.D + self.w_ST * self.S, g, self.bias)
        S_t = sigmoid(saliency + self.w_TS * self.T, g, self.bias)

        self.D += dt * (-self.D + D_t) / self.tau
        self.T += dt * (-self.T + T_t) / self.tau
        self.S += dt * (-self.S + S_t) / self.tau

        for a in ("D", "T", "S"):
            setattr(self, a, float(np.clip(getattr(self, a), 0.0, 1.0)))

        mode = "task" if self.T > self.D else "rest"
        return {"DMN": self.D, "TPN": self.T, "Salience": self.S, "mode": mode}
