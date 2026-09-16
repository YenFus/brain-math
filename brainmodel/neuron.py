"""
neuron.py - the cellular layer.

Two levels of description, both used elsewhere in the model:

1. LIFNeuron       - a spiking leaky integrate-and-fire neuron. This is the
                     literal "electrical action potential travels down the axon"
                     story from the document, reduced to its canonical equation.

2. RatePopulation  - a population firing *rate* unit (mean-field). Large-scale
                     subsystems (networks, basal ganglia) operate at this level
                     because tracking billions of spikes is neither tractable
                     nor informative for system-level behavior.

Leaky integrate-and-fire
-------------------------
    tau_m dV/dt = -(V - V_rest) + R * I(t)

When V crosses V_thresh the neuron emits a spike and V resets to V_reset,
then is held at reset for an absolute refractory period. This captures
threshold, reset, and leak - the three features that make a neuron a neuron.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from .core import Subsystem, sigmoid


@dataclass
class LIFNeuron(Subsystem):
    """Single leaky integrate-and-fire neuron (membrane potential in mV)."""

    tau_m: float = 20.0      # membrane time constant (ms)
    V_rest: float = -65.0    # resting potential (mV)
    V_reset: float = -70.0   # post-spike reset (mV)
    V_thresh: float = -50.0  # spike threshold (mV)
    R: float = 10.0          # membrane resistance (Mohm)
    refractory: float = 2.0  # absolute refractory period (ms)

    V: float = field(default=-65.0)
    _refrac_left: float = 0.0
    name: str = "lif"

    def reset(self) -> None:
        self.V = self.V_rest
        self._refrac_left = 0.0

    def step(self, dt: float, inputs):
        I = float(inputs.get("I", 0.0))  # input current (nA)
        spike = 0.0
        if self._refrac_left > 0.0:
            self._refrac_left -= dt
            self.V = self.V_reset
        else:
            dV = (-(self.V - self.V_rest) + self.R * I) / self.tau_m
            self.V += dV * dt
            if self.V >= self.V_thresh:
                self.V = self.V_reset
                self._refrac_left = self.refractory
                spike = 1.0
        return {"V": self.V, "spike": spike}


@dataclass
class RatePopulation(Subsystem):
    """Mean-field rate unit: a Wilson-Cowan-style population.

        tau dr/dt = -r + phi(input)

    `r` is normalized firing rate in (0, 1). `phi` is a saturating sigmoid whose
    `gain` and `bias` are set by neuromodulators (E/I balance, arousal). This is
    the workhorse unit for every system-level subsystem.
    """

    tau: float = 10.0     # population time constant (ms)
    gain: float = 1.0     # slope of transfer function
    bias: float = 0.0     # threshold of transfer function
    r: float = 0.0        # current firing rate (0..1)
    name: str = "pop"

    def reset(self) -> None:
        self.r = 0.0

    def drive(self, total_input: float, dt: float) -> float:
        target = sigmoid(total_input, gain=self.gain, bias=self.bias)
        self.r += dt * (-self.r + target) / self.tau
        self.r = float(np.clip(self.r, 0.0, 1.0))
        return self.r

    def step(self, dt: float, inputs):
        total = float(inputs.get("input", 0.0))
        return {"r": self.drive(total, dt)}
