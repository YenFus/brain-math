"""
neuromodulators.py - biologically grounded coupled neurochemical dynamical system.

Neurotransmitters are not disconnected scalars. In the biological brain, they form
a coupled dynamical system with reciprocal inhibition, cross-talk, and functional opponency:
  - Dopamine (DA): Reward prediction error (phasic), incentive salience, seeking drive.
  - Serotonin (5-HT): Patience, behavioral inhibition, stability, mood control.
      Opposes impulsive dopamine bursts (Daw, Kakade, Dayan 2002).
  - Norepinephrine / Noradrenaline (NE): Encodes unexpected uncertainty (Yu & Dayan 2005),
      arousal, vigilance, and cognitive exploration gain (Aston-Jones & Cohen 2005).
  - Acetylcholine (ACh): Encodes expected uncertainty and familiarity; gates hippocampal
      synaptic plasticity and memory encoding.
  - Glutamate & GABA: Global excitation-inhibition (E/I) balance and network stability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict
import numpy as np


@dataclass
class Modulator:
    level: float = 1.0          # tonic concentration (1.0 = healthy baseline)
    baseline: float = 1.0       # homeostatic setpoint
    sensitivity: float = 1.0    # receptor sensitivity (receptor up/down-regulation)
    phasic: float = 0.0         # transient deviation (burst or dip)
    tau_tonic: float = 5000.0   # slow adaptation time constant (ms)
    tau_phasic: float = 120.0   # rapid transient decay time constant (ms)

    @property
    def effective(self) -> float:
        """What target circuits actually experience: (tonic + phasic) * receptor sensitivity."""
        return float(np.clip((self.level + self.phasic) * self.sensitivity, 0.0, 5.0))

    def decay_phasic(self, dt: float, tau: float = None) -> None:
        decay_tau = tau if tau is not None else self.tau_phasic
        self.phasic += dt * (-self.phasic) / max(decay_tau, 1.0)


@dataclass
class Neuromodulators:
    """Coupled multi-transmitter system reflecting human neurochemistry."""

    glutamate: Modulator = field(default_factory=Modulator)
    gaba: Modulator = field(default_factory=Modulator)
    dopamine: Modulator = field(default_factory=Modulator)
    serotonin: Modulator = field(default_factory=Modulator)
    acetylcholine: Modulator = field(default_factory=Modulator)
    norepinephrine: Modulator = field(default_factory=Modulator)

    # Cross-talk coupling parameters
    k_5ht_da_inhibit: float = 0.25   # 5-HT dampens impulsive DA spikes (Daw et al.)
    k_ne_surprise: float = 0.40      # Prediction error magnitude drives NE vigilance
    k_ach_familiarity: float = 0.20  # Task familiarity boosts ACh
    k_da_seeking: float = 0.30       # Intrinsic curiosity drive for dopamine

    def step(self, dt: float, inputs: Dict[str, float]) -> Dict[str, float]:
        """Advance the coupled transmitter equations by dt (ms).
        Inputs may supply:
          rpe: reward prediction error (phasic DA drive)
          threat: environmental or task threat/risk
          surprise: unexpected task failure or state divergence
          familiarity: known vs novel state (0..1)
          crf: stress factor from HPA axis
        """
        rpe = float(inputs.get("rpe", 0.0))
        threat = float(inputs.get("threat", 0.0))
        surprise = float(inputs.get("surprise", abs(rpe)))
        familiarity = float(inputs.get("familiarity", 0.5))
        crf = float(inputs.get("crf", 0.0))

        # 1. Phasic Dopamine driven by RPE, tempered by tonic Serotonin
        da_suppression = self.k_5ht_da_inhibit * max(0.0, self.serotonin.effective - 1.0)
        self.dopamine.phasic += dt * (rpe - da_suppression) / self.dopamine.tau_phasic

        # 2. Norepinephrine surges with unexpected surprise and HPA stress
        ne_drive = self.k_ne_surprise * surprise + 0.3 * crf
        self.norepinephrine.phasic += dt * (ne_drive - self.norepinephrine.phasic) / self.norepinephrine.tau_phasic

        # 3. Serotonin tracks safety and long-term stability
        safety = max(0.0, 1.0 - threat - 0.5 * crf)
        d_5ht_tonic = (safety - self.serotonin.level) / self.serotonin.tau_tonic
        self.serotonin.level = float(np.clip(self.serotonin.level + dt * d_5ht_tonic, 0.2, 2.5))

        # 4. Acetylcholine tracks task familiarity and cognitive engagement
        ach_drive = self.k_ach_familiarity * familiarity + 0.1 * self.norepinephrine.effective
        d_ach_tonic = (ach_drive - (self.acetylcholine.level - 0.8)) / self.acetylcholine.tau_tonic
        self.acetylcholine.level = float(np.clip(self.acetylcholine.level + dt * d_ach_tonic, 0.3, 2.0))

        # 5. Natural passive decay of phasic bursts
        self.decay(dt)

        return self.snapshot()

    def excitation_inhibition_ratio(self) -> float:
        """E/I balance: Glutamate vs GABA. >1.0 = excited, <1.0 = inhibited."""
        return self.glutamate.effective / max(self.gaba.effective, 1e-6)

    def cortical_gain(self) -> float:
        """Norepinephrine and E/I balance sharpen cortical neural activation functions."""
        return float(0.5 + 0.3 * self.excitation_inhibition_ratio() + 0.4 * self.norepinephrine.effective)

    def decay(self, dt: float) -> None:
        """Decay all phasic modulations back toward zero."""
        for m in (
            self.glutamate, self.gaba, self.dopamine,
            self.serotonin, self.acetylcholine, self.norepinephrine,
        ):
            m.decay_phasic(dt)

    def snapshot(self) -> Dict[str, float]:
        return {
            "dopamine": self.dopamine.effective,
            "serotonin": self.serotonin.effective,
            "norepinephrine": self.norepinephrine.effective,
            "acetylcholine": self.acetylcholine.effective,
            "glutamate": self.glutamate.effective,
            "gaba": self.gaba.effective,
            "E/I": self.excitation_inhibition_ratio(),
        }
