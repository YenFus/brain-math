"""
brainstem.py - the source nuclei and autonomic core.

The document treats neuromodulators as substances, but anatomically each is
manufactured by a specific brainstem/basal-forebrain nucleus whose firing rate
*is* the tonic level the rest of the brain feels. This module grounds the
chemical layer in that anatomy and adds the medulla's autonomic survival
reflexes.

Source nuclei -> neuromodulator tonic levels
--------------------------------------------
  Raphe nuclei             -> serotonin      (high in wake, low in REM sleep)
  VTA / substantia nigra   -> dopamine       (tonic firing; phasic bursts elsewhere)
  Locus coeruleus          -> norepinephrine (arousal; inverted-U task gain)
  Tuberomammillary nucleus -> histamine      (wakefulness)
  Pedunculopontine / basal -> acetylcholine  (arousal; high in wake AND REM)

Each nucleus' output is a function of a single global `arousal` state (0 sleep ->
1 alert) and circadian drive. This is why anaesthesia, sleep, and stress move
*many* transmitters at once: they move the nuclei, not the molecules.

Medulla (autonomic)
-------------------
Heart rate and respiratory rate are set by sympathetic/parasympathetic balance,
itself pushed by the stress hormone CRF and norepinephrine. This is the
"regulating cardiac rhythm, respiratory rate" survival role of the medulla.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .core import Subsystem


@dataclass
class Brainstem(Subsystem):
    arousal: float = 1.0          # 0 deep sleep ... 1 fully alert
    tau_arousal: float = 200.0

    # baseline (alert) firing of each source nucleus
    raphe_max: float = 1.0        # serotonin
    vta_max: float = 1.0          # dopamine (tonic)
    lc_max: float = 1.0           # norepinephrine
    tmn_max: float = 1.0          # histamine
    ppt_max: float = 1.0          # acetylcholine

    # autonomic state
    heart_rate: float = 70.0      # bpm
    resp_rate: float = 14.0       # breaths/min
    name: str = "brainstem"

    def set_arousal_target(self, target: float, dt: float):
        # exponential relaxation: stable for any dt (including coarse circadian steps)
        alpha = 1.0 - np.exp(-dt / self.tau_arousal)
        self.arousal += alpha * (target - self.arousal)
        self.arousal = float(np.clip(self.arousal, 0.0, 1.0))

    def step(self, dt: float, inputs):
        target_arousal = float(inputs.get("arousal_drive", self.arousal))
        crf = float(inputs.get("crf", 0.0))       # stress hormone -> sympathetic push
        rem = float(inputs.get("rem", 0.0))       # REM sleep flag (0..1)
        self.set_arousal_target(target_arousal, dt)

        a = self.arousal
        # serotonin & histamine collapse in sleep (and 5-HT especially in REM)
        serotonin = self.raphe_max * a * (1.0 - 0.8 * rem)
        histamine = self.tmn_max * a
        # ACh stays high in REM even though arousal (as wakefulness) is low
        acetylcholine = self.ppt_max * max(a, 0.7 * rem)
        # dopamine tonic floor; norepinephrine follows arousal + stress
        dopamine = self.vta_max * (0.3 + 0.7 * a)
        norepinephrine = self.lc_max * np.clip(0.2 + 0.8 * a + 0.5 * crf, 0, 2)

        # medulla autonomic output
        sympathetic = np.clip(0.3 + 0.4 * a + 0.8 * crf, 0, 2)
        self.heart_rate = 60.0 + 50.0 * sympathetic
        self.resp_rate = 10.0 + 10.0 * sympathetic

        return {
            "arousal": a,
            "serotonin": serotonin,
            "dopamine": dopamine,
            "norepinephrine": norepinephrine,
            "histamine": histamine,
            "acetylcholine": acetylcholine,
            "heart_rate": self.heart_rate,
            "resp_rate": self.resp_rate,
        }

    def apply_to(self, nm, out: dict) -> None:
        """Write nucleus outputs into a Neuromodulators object as tonic levels.
        Phasic dopamine (reward prediction error) is added elsewhere and rides on
        top of this tonic floor."""
        nm.serotonin.level = out["serotonin"]
        nm.dopamine.level = out["dopamine"]
        nm.norepinephrine.level = out["norepinephrine"]
        nm.acetylcholine.level = out["acetylcholine"]
