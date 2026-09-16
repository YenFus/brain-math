"""
brainstem_bridge.py - Descending Motor Interface connecting Cognitive/Language Setpoints
to the 302-Neuron C. elegans Biophysical Connectome.

Translates high-level behavioral strategies into descending drive currents:
  - I_AVB: Forward crawling drive current (pA)
  - I_AVA: Reversal / Pirouette command current (pA)
  - I_ASE: Chemosensory gain bias
  - Neuromodulatory tone: DA (foraging/slowing), 5-HT (dwelling), NE/OA (vigilance/roaming)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np


@dataclass
class CognitiveSetpoint:
    """High-level behavioral setpoint emitted by language/cortex layer."""
    behavior_mode: str = "FORAGE"     # FORAGE, SPRINT, PATROL, EVADE, DWELL, STANDBY
    forward_drive: float = 1.0        # 0.0 to 2.0
    turn_bias: float = 0.0            # -1.0 (left) .. +1.0 (right)
    reversal_urgency: float = 0.0     # 0.0 to 1.0 (triggers AVA reversal)
    vigilance: float = 0.5            # 0.0 relaxed .. 1.0 hyper-alert
    chemo_attraction: float = 1.0     # gain on chemical gradients


class ConnectomeBrainstemBridge:
    """Translates CognitiveSetpoints into graded micro-currents for the connectome."""

    def __init__(self):
        # Baseline tonic calibrations (pA / mV equivalents)
        self.baseline_crawl_current = 8.0
        self.max_reversal_current = 28.0
        self.max_forward_boost = 18.0

    def compute_descending_currents(self, setpoint: CognitiveSetpoint, threat_level: float, scent_delta: float) -> Dict[str, float]:
        """Compute descending synaptic currents to command interneurons."""

        # 1. Forward Crawl (AVB)
        # SPRINT & FORAGE boost AVB; EVADE & STANDBY suppress it
        avb_drive = self.baseline_crawl_current * setpoint.forward_drive
        if setpoint.behavior_mode == "SPRINT":
            avb_drive += self.max_forward_boost
        elif setpoint.behavior_mode == "DWELL":
            avb_drive *= 0.4
        elif setpoint.behavior_mode == "STANDBY":
            avb_drive = 0.0

        # 2. Reversal / Escape (AVA / AVD)
        # Reversals are triggered by high reversal_urgency or EVADE mode or extreme threat
        ava_drive = 0.0
        if setpoint.behavior_mode == "EVADE" or setpoint.reversal_urgency > 0.4:
            ava_drive = self.max_reversal_current * max(setpoint.reversal_urgency, 0.7)
        elif threat_level > 0.6:
            ava_drive = self.max_reversal_current * (threat_level * setpoint.vigilance)

        # 3. Posterior Sprint Interneuron (PVC)
        pvc_drive = 12.0 * setpoint.forward_drive if setpoint.behavior_mode in ["SPRINT", "PATROL"] else 4.0

        # 4. Chemosensory Weighting (ASE)
        ase_current = scent_delta * 15.0 * setpoint.chemo_attraction

        # 5. Neuromodulatory adjustments
        da_boost = 1.0 if setpoint.behavior_mode == "FORAGE" and scent_delta > 0 else 0.0
        ne_boost = 1.5 if setpoint.behavior_mode == "EVADE" or threat_level > 0.5 else 0.0

        return {
            "I_AVB": float(np.clip(avb_drive, 0.0, 30.0)),
            "I_AVA": float(np.clip(ava_drive, 0.0, 35.0)),
            "I_PVC": float(np.clip(pvc_drive, 0.0, 25.0)),
            "I_ASE": float(np.clip(ase_current, -20.0, 20.0)),
            "turn_bias": float(np.clip(setpoint.turn_bias, -1.0, 1.0)),
            "da_boost": da_boost,
            "ne_boost": ne_boost,
        }
