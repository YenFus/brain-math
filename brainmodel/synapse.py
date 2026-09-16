"""
synapse.py - synaptic transmission and plasticity (LTP / LTD).

This module implements, as equations, the exact LTP cascade described in the
document:

  glutamate release -> AMPA (Na+ influx, depolarization) -> NMDA Mg2+ block
  cleared by depolarization -> Ca2+ influx -> calcium controls weight change.

NMDA magnesium block (Jahr-Stevens form)
----------------------------------------
The fraction of NMDA channels unblocked at membrane potential V:

    B(V) = 1 / (1 + (Mg / 3.57) * exp(-0.062 * V))

At rest (V hyperpolarized) B ~ 0: the Mg2+ ion plugs the pore. Depolarization
(from AMPA-driven Na+ influx) drives B -> 1, "unlocking" the channel. This single
term is why LTP is *coincidence-detecting* (Hebbian): you need presynaptic
glutamate AND postsynaptic depolarization at the same time.

Calcium-control plasticity (Shouval / BCM-like)
-----------------------------------------------
Postsynaptic calcium is the second messenger. Its level decides the *sign* of
plasticity via two thresholds:

    Ca < theta_d              -> no change
    theta_d < Ca < theta_p    -> LTD  (weight decreases)  - moderate Ca
    Ca > theta_p              -> LTP  (weight increases)   - high Ca

    dw/dt = eta * ( Omega(Ca) - lambda * w )

`Omega` is the +/- learning curve; the `-lambda*w` term is homeostatic decay
(forgetting / normalization). CaMKII-driven AMPA insertion is folded into the
LTP branch as an increase in `w` (AMPA conductance).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from .core import Subsystem


def nmda_unblock(V_mV: float, Mg: float = 1.0) -> float:
    """Fraction of NMDA receptors relieved of the Mg2+ block at potential V."""
    return 1.0 / (1.0 + (Mg / 3.57) * np.exp(-0.062 * V_mV))


@dataclass
class PlasticSynapse(Subsystem):
    """A single glutamatergic synapse with AMPA + NMDA and calcium plasticity."""

    w: float = 0.5            # synaptic weight ~ AMPA conductance (0..w_max)
    w_max: float = 2.0

    # receptor densities (pathologies scale these: e.g. addiction up-regulates NMDA/AMPA)
    ampa_density: float = 1.0
    nmda_density: float = 1.0
    Mg: float = 1.0           # extracellular [Mg2+] (mM); lowering it disinhibits NMDA

    # postsynaptic calcium dynamics
    Ca: float = 0.0
    tau_Ca: float = 20.0      # calcium decay (ms)
    Ca_influx_gain: float = 1.0

    # plasticity thresholds and rates (calcium-control hypothesis)
    theta_d: float = 0.25     # LTD threshold
    theta_p: float = 0.55     # LTP threshold
    eta: float = 0.01         # learning rate
    lam: float = 0.002        # homeostatic decay (forgetting of EARLY-phase LTP)

    # late-phase LTP: synaptic tagging + protein synthesis (CREB / immediate
    # early genes c-fos, Arc, zif268) convert transient potentiation into a
    # stable, decay-resistant weight component.
    w_late: float = 0.0       # consolidated, persistent weight
    tag: float = 0.0          # synaptic tag set by strong LTP-level calcium
    prp: float = 0.0          # plasticity-related proteins (gene expression)
    protein_synthesis: float = 1.0  # 1.0 intact; 0 blocks consolidation (anisomycin / AD)
    eta_late: float = 0.0002  # tag x PRP -> late weight growth rate (slow consolidation)

    # approximate postsynaptic depolarization used for the Mg gate
    V_post: float = -65.0
    tau_V: float = 15.0
    name: str = "synapse"

    @property
    def weight(self) -> float:
        """Total synaptic efficacy = early (labile) + late (consolidated)."""
        return float(np.clip(self.w + self.w_late, 0.0, 2.0 * self.w_max))

    def reset(self) -> None:
        self.Ca = 0.0
        self.V_post = -65.0

    def _omega(self, Ca: float) -> float:
        """LTD/LTP curve: negative for moderate Ca, positive for high Ca."""
        if Ca < self.theta_d:
            return 0.0
        if Ca < self.theta_p:
            # descend into LTD then climb back toward zero
            return -1.0
        return +1.0

    def step(self, dt: float, inputs):
        pre = float(inputs.get("pre", 0.0))     # presynaptic activity / glutamate (0..1)
        post_drive = float(inputs.get("post", 0.0))  # extra postsynaptic depolarizing drive

        # AMPA-mediated depolarization (Na+ influx). Strength scales with weight.
        ampa_current = self.ampa_density * self.w * pre
        # crude postsynaptic membrane: relaxes to rest, pushed up by AMPA + external drive
        V_target = -65.0 + 60.0 * np.tanh(ampa_current + post_drive)
        self.V_post += dt * (-(self.V_post - V_target)) / self.tau_V

        # NMDA conductance requires BOTH glutamate (pre) AND depolarization (Mg unblock)
        B = nmda_unblock(self.V_post, self.Mg)
        nmda_current = self.nmda_density * pre * B

        # calcium influx through NMDA; decays exponentially
        self.Ca += dt * (self.Ca_influx_gain * nmda_current - self.Ca / self.tau_Ca)
        self.Ca = max(self.Ca, 0.0)

        # --- early-phase LTP/LTD: fast, labile weight from calcium -------------
        omega = self._omega(self.Ca)
        dw = self.eta * (omega - self.lam * self.w)
        self.w = float(np.clip(self.w + dw * dt, 0.0, self.w_max))

        # --- late-phase LTP: tagging + protein synthesis -> stable weight ------
        # Strong (LTP-level) calcium sets a synaptic tag and, when sustained,
        # triggers gene expression / protein synthesis (PRP). Where a tag meets
        # available proteins, an enduring late weight is written.
        if omega > 0:
            self.tag += dt * (1.0 - self.tag) * 0.02         # tag rises with LTP
            self.prp += dt * self.protein_synthesis * 0.0004  # IEG / CREB cascade
        else:
            self.tag += dt * (-self.tag) * 0.01              # tag decays
        self.prp += dt * (-self.prp) * 0.0008                # proteins degrade slowly
        self.prp = float(np.clip(self.prp, 0.0, 1.0))        # protein pool saturates
        self.w_late += dt * self.eta_late * self.tag * self.prp
        self.w_late = float(np.clip(self.w_late, 0.0, self.w_max))

        return {
            "w": self.w,            # early/labile component
            "w_late": self.w_late,  # consolidated component
            "weight": self.weight,  # total efficacy
            "tag": self.tag,
            "prp": self.prp,
            "Ca": self.Ca,
            "nmda_unblock": B,
            "V_post": self.V_post,
            "ampa_current": ampa_current,
        }


def excitotoxicity_loss(Ca: float, Ca_tox: float = 3.0, rate: float = 0.05) -> float:
    """Pathological weight/health loss when calcium runs away (Alzheimer's, alcohol
    withdrawal, ischemia). Returns a non-negative damage increment per ms.

    The document's "excess glutamate -> Ca2+ influx -> destroys neurons" is exactly
    a calcium level above a toxicity ceiling driving irreversible loss.
    """
    return rate * max(0.0, Ca - Ca_tox)
