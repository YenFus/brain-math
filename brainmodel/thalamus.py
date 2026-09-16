"""
thalamus.py - the sensory gateway and cortical arousal switch.

The document calls the thalamus "the switchboard of the central nervous system":
every sense except smell relays through it on the way to cortex, and it decides
what gets through. We model the named relay nuclei and a single gating gain.

Relay nuclei
------------
  LGN  -> visual cortex        VPL/VPM -> somatosensory cortex
  MGN  -> auditory cortex      (a generic relay stands in for the rest)

Gating
------
Each channel's throughput is a gain G in [0, 1]:

    G = arousal * (baseline_gate + attention_boost) * (1 - TRN_inhibition)

- arousal: in sleep/anaesthesia the thalamus switches to *burst mode* and stops
  faithfully relaying - the cortex is disconnected from the world.
- attention_boost: top-down signals (TPN / salience) raise the gain of the
  attended channel - the neural basis of selective attention.
- TRN (thalamic reticular nucleus): a GABAergic shell that inhibits relay; it is
  how the thalamus suppresses irrelevant input.

Hallucination = gating failure: when prefrontal control is weak, internally
generated noise leaks through the gate and reaches sensory cortex as if it were
real. We expose `reality_monitoring` (PFC strength) so this falls out directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict
import numpy as np

from .core import Subsystem


CHANNELS = ("visual", "auditory", "somato")


@dataclass
class Thalamus(Subsystem):
    baseline_gate: float = 0.5
    trn_inhibition: float = 0.2
    internal_noise: float = 0.05     # spontaneous activity that can leak through
    reality_monitoring: float = 1.0  # PFC veto on internal signals (low -> hallucination)
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))
    name: str = "thalamus"

    def step(self, dt: float, inputs):
        sensory: Dict[str, float] = inputs.get("sensory", {})
        arousal = float(inputs.get("arousal", 1.0))
        attention = float(inputs.get("attention", 0.0))   # top-down boost (0..1)
        attended = inputs.get("attended_channel", None)

        relayed = {}
        leaked = 0.0
        for ch in CHANNELS:
            real = float(sensory.get(ch, 0.0))
            boost = attention if ch == attended else 0.0
            G = arousal * (self.baseline_gate + boost) * (1.0 - self.trn_inhibition)
            G = float(np.clip(G, 0.0, 1.0))

            # internally generated noise; normally vetoed by reality monitoring
            noise = self.internal_noise * abs(self.rng.standard_normal())
            phantom = noise * (1.0 - self.reality_monitoring)
            leaked += phantom

            relayed[ch] = float(np.clip(G * real + phantom, 0.0, 1.0))

        return {
            "relayed": relayed,
            "salient_input": max(relayed.values()) if relayed else 0.0,
            "phantom_leak": leaked,          # >0 == hallucinatory intrusion
            "mode": "tonic" if arousal > 0.4 else "burst",
        }
