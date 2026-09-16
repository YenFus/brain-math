"""
pathology.py - disease as perturbation.

The central design claim: you do not write special code for each disorder. You
take the healthy `Brain` and change parameters the way the document says the
biology changes, and the pathological behavior emerges from the same equations.

Where the perturbation must act
-------------------------------
Because the brainstem now *manufactures* the neuromodulators each tick (raphe ->
serotonin, VTA/SNc -> dopamine, etc.), a disorder cannot just overwrite a tonic
level. It must act at the correct biological locus:

  * production deficit  -> a brainstem nucleus' max output (e.g. SNc loss)
  * receptor change     -> nm.<transmitter>.sensitivity (up/down-regulation)
  * structural lesion   -> a subsystem's integrity/damage parameter
  * E/I transmitter     -> glutamate/gaba .level (not brainstem-controlled)

This mirrors the difference between losing the cells that make a transmitter and
changing how target cells respond to it.
"""

from __future__ import annotations

from .brain import Brain


def healthy(brain: Brain) -> Brain:
    return brain


# --- movement disorders ------------------------------------------------------

def parkinsons(brain: Brain) -> Brain:
    """Substantia nigra (SNc) dopamine-neuron loss -> reduced dopamine PRODUCTION.
    GPi inhibition stays high -> akinesia / bradykinesia."""
    brain.brainstem.vta_max = 0.35
    return brain


def parkinsons_dyskinesia(brain: Brain) -> Brain:
    """L-DOPA overshoot / hyperdopaminergic swing -> excessive disinhibition ->
    involuntary movement (chorea / dyskinesia)."""
    brain.brainstem.vta_max = 1.9
    return brain


def cerebral_palsy_spastic(brain: Brain) -> Brain:
    """Corticospinal damage -> loss of presynaptic inhibition in motor gating;
    an undifferentiated leak co-activates channels -> spasticity / co-contraction."""
    brain.bg.presynaptic_inhibition = 0.3
    return brain


def cerebral_palsy_ataxic(brain: Brain) -> Brain:
    """Cerebellar damage -> the forward model can no longer cancel motor error ->
    dysmetria (overshoot) and intention tremor; uncoordinated movement."""
    brain.cerebellum.damage = 0.75
    brain.cerebellum.tremor_gain = 0.4
    return brain


# --- neurodegeneration -------------------------------------------------------

def alzheimers(brain: Brain) -> Brain:
    """Cholinergic basal-forebrain loss + glutamate excitotoxicity + failed
    memory consolidation + DMN that won't deactivate.

    - ACh production down (basal forebrain / PPT degeneration).
    - E/I shifted excitotoxic (glutamate up, GABA down).
    - Hippocampus: impaired neurogenesis, noisy retrieval, and BLOCKED late-LTP
      protein synthesis -> memories don't consolidate (anterograde amnesia).
    - Salience no longer suppresses the DMN -> reality-monitoring intrusions."""
    brain.brainstem.ppt_max = 0.4              # acetylcholine production
    brain.nm.glutamate.level = 1.5
    brain.nm.gaba.level = 0.8
    brain.hippocampus.neurogenesis = 0.4
    brain.hippocampus.retrieval_noise = 0.5
    brain.synapse.protein_synthesis = 0.2      # late-LTP / consolidation fails
    brain.synapse.lam = 0.02                   # faster forgetting of early LTP
    brain.networks.w_SD = 0.4
    return brain


# --- addiction ---------------------------------------------------------------

def addiction(brain: Brain) -> Brain:
    """Chronic drug/alcohol neuroadaptation: D2 + GABA_A down-regulation, NMDA/
    AMPA up-regulation, and blunted pleasure from natural rewards."""
    brain.nm.dopamine.sensitivity = 0.85       # D2 down (receptor)
    brain.reward_gain = 0.45                    # natural rewards less pleasurable
    brain.nm.gaba.sensitivity = 0.7             # GABA_A down (alpha1->alpha4 shift)
    brain.nm.glutamate.sensitivity = 1.4        # NMDA/AMPA up
    brain.synapse.nmda_density = 1.5
    brain.synapse.ampa_density = 1.4
    return brain


def withdrawal(brain: Brain) -> Brain:
    """Abrupt cessation after `addiction`: inhibition is gone, excitation remains.
    Hyperglutamatergic + hypoGABAergic (E/I >> 1) -> seizures / excitotoxicity;
    low serotonin production + high CRF -> dysphoria, anxiety, irritability."""
    addiction(brain)
    brain.nm.gaba.level = 0.5
    brain.nm.glutamate.level = 1.6
    brain.brainstem.raphe_max = 0.5            # serotonin production down (dysphoria)
    brain.hypothalamus.crf = 1.5               # extended-amygdala stress
    return brain


# --- psychosis / perception --------------------------------------------------

def hallucination(brain: Brain) -> Brain:
    """Thalamocortical gating failure + dopaminergic / serotonergic hyperactivity.

    - Thalamus reality-monitoring fails -> internally generated noise leaks to
      sensory cortex as phantom percepts.
    - Dopamine production up -> aberrant salience attribution.
    - 5-HT2A receptor hyperactivity (sensitivity up).
    - DMN not suppressed -> internal simulations mistaken for reality.
    - Lower ignition threshold -> false percepts reach consciousness."""
    brain.thalamus.reality_monitoring = 0.2
    brain.brainstem.vta_max = 1.6
    brain.nm.serotonin.sensitivity = 1.5
    brain.networks.w_SD = 0.3
    brain.workspace.bias = 0.45
    return brain


# --- prefrontal / affective --------------------------------------------------

def dysexecutive(brain: Brain) -> Brain:
    """OFC / vmPFC damage: subjective-value computation becomes noisy and
    compressed -> erratic, risky, socially inappropriate choices despite intact
    analytic ability (the document's OFC-lesion picture)."""
    brain.cortex.ofc_integrity = 0.3
    return brain


def reactive_aggression(brain: Brain) -> Brain:
    """Weak PFC->amygdala control: low serotonin production, high testosterone,
    high vasopressin, low oxytocin -> large aggression for modest provocation."""
    brain.brainstem.raphe_max = 0.45
    brain.limbic.testosterone = 1.7
    brain.limbic.pfc_integrity = 0.7
    brain.hypothalamus.vasopressin = 1.6
    brain.hypothalamus.oxytocin = 0.5
    return brain


def chronic_stress(brain: Brain) -> Brain:
    """Sustained HPA-axis activation: elevated CRF/cortisol -> high heart rate,
    threat bias, and (via serotonergic suppression) low mood."""
    brain.hypothalamus.crf = 1.6
    brain.brainstem.raphe_max = 0.6
    return brain


REGISTRY = {
    "healthy": healthy,
    "parkinsons": parkinsons,
    "parkinsons_dyskinesia": parkinsons_dyskinesia,
    "cerebral_palsy_spastic": cerebral_palsy_spastic,
    "cerebral_palsy_ataxic": cerebral_palsy_ataxic,
    "alzheimers": alzheimers,
    "addiction": addiction,
    "withdrawal": withdrawal,
    "hallucination": hallucination,
    "dysexecutive": dysexecutive,
    "reactive_aggression": reactive_aggression,
    "chronic_stress": chronic_stress,
}
