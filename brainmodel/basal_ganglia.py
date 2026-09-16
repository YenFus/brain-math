"""
basal_ganglia.py - action selection through the full nuclear circuit.

The document names every nucleus: striatum (caudate/putamen with D1 and D2 spiny
projection neurons), GPe, GPi, STN, SNr, and the dopaminergic SNc. This module
implements all three canonical pathways as explicit population rates rather than
a single lumped equation.

Pathways
--------
  Direct   (Go,   D1):  cortex -> Str(D1) --| GPi/SNr --| thalamus      (release)
  Indirect (NoGo, D2):  cortex -> Str(D2) --| GPe --| STN --> GPi/SNr   (suppress)
  Hyperdirect:          cortex --------------------> STN --> GPi/SNr     (global STOP)

Output nuclei (GPi/SNr) tonically inhibit the thalamus; an action is selected by
*disinhibition* - lowering GPi for that channel. The hyperdirect pathway gives a
fast, broad brake that buys time before committing.

Dopamine (from SNc) is the gain knob: it excites D1 (facilitates Go) and inhibits
D2 (reduces NoGo). Hence:
  - low DA  (Parkinson's): GPi stays high -> akinesia / bradykinesia
  - high DA (dyskinesia / L-DOPA overshoot): GPi collapses -> involuntary release

The striatum's striosome/matrix split is noted via `limbic_bias`, the value/
reward signal that striosomes carry into the gating decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List
import numpy as np

from .core import Subsystem


@dataclass
class BasalGanglia(Subsystem):
    n_actions: int = 3
    gpi_baseline: float = 1.0      # tonic inhibition the thalamus must overcome
    stn_baseline: float = 0.5      # tonic STN drive onto output nuclei
    gpe_baseline: float = 1.0      # tonic GPe inhibition onto STN
    hyperdirect: float = 0.3       # strength of the cortex->STN global brake
    w_stn: float = 0.3             # STN -> GPi excitation weight
    gate_thresh: float = 0.75      # how low GPi must fall to open a channel
    presynaptic_inhibition: float = 1.0  # 1.0 normal; <1 == loss (spastic CP)
    open_eps: float = 0.05         # a channel counts as "open" above this
    name: str = "basal_ganglia"

    # exposed nucleus activities (last step), for inspection
    nuclei: dict = field(default=None)
    last_gates: np.ndarray = field(default=None)

    def step(self, dt: float, inputs):
        cortex = np.asarray(inputs.get("cortex", np.zeros(self.n_actions)), dtype=float)
        DA = float(inputs.get("dopamine", 1.0))

        # --- striatum: D1 (direct/Go) and D2 (indirect/NoGo) spiny neurons -----
        d1 = cortex * (0.4 + 0.5 * DA)              # facilitated by dopamine
        d2 = cortex * np.maximum(0.0, 1.0 - 0.5 * DA)  # suppressed by dopamine

        # --- GPe: inhibited by D2 (indirect pathway) ---------------------------
        gpe = np.maximum(0.0, self.gpe_baseline - d2)

        # --- STN: disinhibited by low GPe, plus the hyperdirect cortical brake --
        stop = self.hyperdirect * float(np.mean(cortex))   # broad cortical STOP
        stn = np.maximum(0.0, self.stn_baseline + (self.gpe_baseline - gpe) + stop)

        # --- GPi/SNr output: lowered by direct D1, raised by STN ---------------
        gpi = self.gpi_baseline - d1 + self.w_stn * stn

        base = np.maximum(0.0, self.gate_thresh - gpi)
        # Losing presynaptic inhibition (CP) adds an undifferentiated leak that
        # opens channels regardless of GPi -> co-contraction / involuntary overflow.
        leak = max(0.0, 1.0 - self.presynaptic_inhibition) * 0.35
        gate = base + leak

        self.nuclei = {
            "D1": d1, "D2": d2, "GPe": gpe, "STN": stn, "GPi": gpi,
        }
        self.last_gates = gate
        selected = int(np.argmax(gate)) if gate.max() > 0 else -1
        channels_open = int(np.sum(gate > self.open_eps))
        return {
            "gates": gate,
            "selected": selected,
            "gpi": gpi,
            "channels_open": channels_open,   # 0 akinesia, 1 clean, >1 overflow
            # how cleanly one action won (low == tremor/indecision/co-contraction)
            "selection_margin": float(np.sort(gate)[-1] - np.sort(gate)[-2]) if len(gate) > 1 else float(gate.max()),
        }
