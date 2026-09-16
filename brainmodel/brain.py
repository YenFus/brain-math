"""
brain.py - the whole-brain integration with Emotion, Dual-Process, and Hardware Governance.

Wires all biological subsystems into one steppable organism:
  Homeostasis & Circadian Clock -> Brainstem Neuromodulator Manufacturing
  -> Thalamic Sensory Gating -> Cortical Perception & Value Computation
  -> Limbic Appraisal & Emotion Engine (Lövheim Cube / PAD / Active Inference)
  -> Basal Ganglia Habitual Gating (System 1) vs Prefrontal MCTS Deliberation (System 2)
  -> Cerebellar Movement Refinement -> Phasic Dopamine Learning -> Hippocampal Memory
  -> Hardware Governor guaranteeing >= 10 GB RAM Headroom on Apple Silicon.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np

from .core import Simulation, sigmoid
from .neuromodulators import Neuromodulators
from .brainstem import Brainstem
from .thalamus import Thalamus
from .hypothalamus import Hypothalamus
from .cortex import Cortex
from .hippocampus import Hippocampus
from .cerebellum import Cerebellum
from .basal_ganglia import BasalGanglia
from .decision import ValueLearner, DriftDiffusion
from .networks import TripleNetwork
from .consciousness import GlobalWorkspace
from .synapse import PlasticSynapse
from .emotion import EmotionEngine, SomaticMarker
from .dual_process import DualProcessDecision
from .hardware_governor import HardwareGovernor


@dataclass
class LimbicAppraisal:
    """Amygdala threat vs PFC control, set by neurochemistry and hormones."""

    pfc_integrity: float = 1.0
    testosterone: float = 1.0
    amygdala: float = 0.0
    k_peptide: float = 0.3

    def appraise(
        self,
        threat: float,
        serotonin: float,
        cortisol: float = 0.2,
        vasopressin: float = 1.0,
        oxytocin: float = 1.0,
    ) -> Dict[str, float]:
        drive = (threat + 0.3 * cortisol) * self.testosterone
        self.amygdala = float(sigmoid(drive, gain=4.0, bias=0.5))
        peptide = 1.0 + self.k_peptide * (vasopressin - oxytocin)
        control = serotonin * self.pfc_integrity
        aggression = max(0.0, self.amygdala * peptide - control)
        return {"amygdala": self.amygdala, "pfc_control": control, "aggression": aggression}


@dataclass
class Brain:
    n_actions: int = 3
    dt: float = 1.0
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))

    # --- Subsystems (the full biological atlas) ---
    nm: Neuromodulators = field(default_factory=Neuromodulators)
    hypothalamus: Hypothalamus = field(default_factory=Hypothalamus)
    brainstem: Brainstem = field(default_factory=Brainstem)
    thalamus: Thalamus = None
    cortex: Cortex = None
    hippocampus: Hippocampus = None
    cerebellum: Cerebellum = field(default_factory=Cerebellum)
    networks: TripleNetwork = field(default_factory=TripleNetwork)
    workspace: GlobalWorkspace = field(default_factory=GlobalWorkspace)
    limbic: LimbicAppraisal = field(default_factory=LimbicAppraisal)
    bg: BasalGanglia = None
    value: ValueLearner = None
    synapse: PlasticSynapse = field(default_factory=PlasticSynapse)

    # --- Cognitive & Affective Engines ---
    emotion: EmotionEngine = field(default_factory=EmotionEngine)
    dual_process: DualProcessDecision = field(default_factory=DualProcessDecision)
    governor: HardwareGovernor = field(default_factory=lambda: HardwareGovernor(min_headroom_gb=10.0))

    # --- Whole-brain state ---
    tissue_health: float = 1.0
    reward_gain: float = 1.0
    _attention: float = 0.2
    _experiences: list = None
    _encode_capacity: float = 1.0
    _last_rpe: float = 0.0
    sim: Simulation = None

    def __post_init__(self):
        if self.thalamus is None:
            self.thalamus = Thalamus(rng=self.rng)
        if self.cortex is None:
            self.cortex = Cortex(n_actions=self.n_actions, rng=self.rng)
        if self.hippocampus is None:
            self.hippocampus = Hippocampus(rng=self.rng)
        if self.bg is None:
            self.bg = BasalGanglia(n_actions=self.n_actions)
        if self.value is None:
            self.value = ValueLearner(n_actions=self.n_actions, rng=self.rng)
        if self._experiences is None:
            self._experiences = [self.rng.standard_normal(self.hippocampus.n) for _ in range(6)]
        self.sim = Simulation(dt=self.dt)

    def step(self, world: Dict) -> Dict:
        """Advance the whole brain one tick."""
        dt = self.dt
        nm = self.nm

        sensory = world.get("sensory", {"visual": world.get("saliency", 0.0)})
        external = float(world.get("external", 0.6))
        threat = float(world.get("threat", 0.0))
        stakes = float(world.get("stakes", 0.0))
        true_rewards = np.asarray(world.get("true_rewards", np.zeros(self.n_actions)), dtype=float)

        # 0) HARDWARE GOVERNOR: check RAM headroom (guaranteeing >= 10 GB)
        mem_snap = self.governor.sample_memory()
        # Compute hardware strain if headroom drops toward 10 GB
        hardware_stress = float(max(0.0, (11.0 - mem_snap.available_gb) / 1.0)) if not mem_snap.is_safe else 0.0

        # 1) HYPOTHALAMUS: circadian clock, sleep pressure, HPA axis, metabolic strain
        hyp = self.hypothalamus.step(dt, {
            "light": world.get("light", None),
            "stressor": world.get("stressor", threat),
            "food": world.get("food", 0.0),
            "temp_load": world.get("temp_load", 0.0),
            "hardware_stress": hardware_stress,
        })

        # 2) BRAINSTEM: arousal -> tonic modulators & autonomic heart rate
        bs = self.brainstem.step(dt, {
            "arousal_drive": world.get("arousal_drive", hyp["arousal_drive"]),
            "crf": hyp["crf"],
            "rem": hyp["rem"],
        })
        self.brainstem.apply_to(nm, bs)
        arousal = bs["arousal"]
        asleep = arousal < 0.35

        # 2b) ADVANCE COUPLED NEUROMODULATORS (DA, 5HT, NE, ACh cross-talk)
        nm.step(dt, {
            "rpe": self._last_rpe,
            "threat": threat,
            "surprise": abs(self._last_rpe),
            "crf": hyp["crf"],
            "familiarity": float(np.clip(1.0 - thal_phantom if (thal_phantom := world.get("phantom_leak", 0.0)) else 0.7, 0.0, 1.0)),
        })

        # 3) THALAMUS: sensory gating
        attended = max(sensory, key=sensory.get) if sensory else None
        thal = self.thalamus.step(dt, {
            "sensory": sensory,
            "arousal": arousal,
            "attention": self._attention,
            "attended_channel": attended,
        })
        salient = thal["salient_input"] + thal["phantom_leak"]

        # 4) LIMBIC APPRAISAL & EMOTION ENGINE: compute full affective state
        limb = self.limbic.appraise(
            threat, nm.serotonin.effective, cortisol=hyp["cortisol"],
            vasopressin=hyp["vasopressin"], oxytocin=hyp["oxytocin"],
        )
        somatic_marker = self.emotion.update(
            dopamine=nm.dopamine.effective,
            serotonin=nm.serotonin.effective,
            norepinephrine=nm.norepinephrine.effective,
            prediction_error=self._last_rpe,
            threat=threat,
            cortisol=hyp["cortisol"],
            testosterone=hyp["testosterone"],
            dt=dt,
        )

        # 5) CORTEX: percepts, OFC subjective valuation with somatic markers, ACC conflict
        percepts = self.cortex.sensory(thal["relayed"])
        limbic_bias = -0.2 * threat + 0.1 * (1.0 - hyp["hunger"])
        values = self.cortex.subjective_value(
            self.value.valuation(),
            limbic_bias=limbic_bias,
            somatic_valence=somatic_marker.valence,
            somatic_confidence=somatic_marker.confidence,
        )
        conflict = self.cortex.conflict(values)
        control_gate = self._attention * nm.norepinephrine.effective
        self.cortex.working_memory(values, control_gate, dt)

        # 6) LARGE-SCALE NETWORKS: Salience / TPN / DMN
        net = self.networks.step(dt, {
            "internal": 0.5,
            "external": external * arousal,
            "saliency": salient,
            "acetylcholine": nm.acetylcholine.effective,
        })
        self._attention = float(np.clip(net["TPN"] * net["Salience"] + 0.1, 0, 1))

        # 7) GLOBAL WORKSPACE: conscious ignition
        ws = self.workspace.step(dt, {
            "stimulus": salient * (1.0 if not asleep else 0.0),
            "attention": self._attention,
        })
        conscious = ws["conscious"] and not asleep

        # 8) BASAL GANGLIA: gate the candidate action
        spread = float(np.ptp(values))
        norm = (values - values.min()) / spread if spread > 1e-6 else np.zeros_like(values)
        explore = 0.18 * self.rng.standard_normal(self.n_actions)
        cortex_drive = np.clip(0.35 + 0.55 * norm + explore, 0.0, 1.5)
        gate = self.bg.step(dt, {"cortex": cortex_drive, "dopamine": nm.dopamine.effective})

        # 9) DUAL-PROCESS ARBITRATION: System 1 (reflexive habit) vs System 2 (deliberative search)
        dp_eval = self.dual_process.evaluate_mode(
            heuristic_confidence=1.0 - conflict,
            acc_conflict=conflict,
            norepinephrine=nm.norepinephrine.effective,
            stakes=stakes,
        )
        mode_system = dp_eval["mode"]

        # 10) ACT + LEARN (when conscious and a channel opened)
        delta = 0.0; chosen = -1; reward = 0.0; motor_error = 0.0
        if conscious and gate["selected"] >= 0:
            if not dp_eval["must_deliberate"]:
                # System 1: Softmax over OFC subjective value
                z = self.value.beta * (values - values.max())
                p = np.exp(z) / np.exp(z).sum()
                chosen = int(self.rng.choice(self.n_actions, p=p))
            else:
                # System 2: Deliberative evaluation / rollouts
                chosen, _ = self.dual_process.deliberate_search(
                    candidate_values=values,
                    budget=6,
                )

            raw = float(true_rewards[chosen]) if chosen < len(true_rewards) else 0.0
            reward = raw * self.reward_gain
            self.value.alpha = 0.2 * nm.dopamine.effective
            delta = self.value.learn(chosen, reward)
            self._last_rpe = delta
            nm.dopamine.phasic += delta

            # Cerebellum: refine movement
            command = float(chosen - (self.n_actions - 1) / 2.0)
            desired = 0.4 * command
            _, motor_error = self.cerebellum.refine(command, desired)

            # Store episode in Hippocampus tagged with Somatic Marker
            if len(self.cortex.dlpfc.content) >= self.hippocampus.n:
                pat = self.cortex.dlpfc.content[:self.hippocampus.n]
            else:
                pat = np.pad(self.cortex.dlpfc.content, (0, self.hippocampus.n - len(self.cortex.dlpfc.content)))
            self.hippocampus.store(
                pattern=pat,
                tag=f"action_{chosen}",
                valence=somatic_marker.valence,
                arousal=somatic_marker.arousal,
                salience=1.0 + abs(delta),
            )

        # 11) HIPPOCAMPUS ENCODING & CONSOLIDATION
        self._encode_capacity = float(nm.acetylcholine.effective)
        self.hippocampus.encode_gain = self._encode_capacity
        if hyp["needs_consolidation"]:
            # Offline replay and synaptic pruning
            replayed = self.hippocampus.sleep_replay(n_episodes=3)
            self.hippocampus.consolidate(decay_rate=0.01)

        # 12) PLASTICITY: early + late LTP
        pre = (self._attention * (0.5 + 0.5 * net["TPN"])) if not asleep else 0.6
        syn = self.synapse.step(dt, {"pre": pre, "post": max(0.0, reward) * 0.5})

        # 13) EXCITOTOXICITY & TISSUE HEALTH
        ei = nm.excitation_inhibition_ratio()
        toxic = max(0.0, ei - 1.3) * min(syn["Ca"], 4.0)
        self.tissue_health = float(np.clip(self.tissue_health - dt * 0.00012 * toxic, 0.0, 1.0))

        nm.decay(dt)

        out = {
            "t": self.sim.t,
            "clock_hours": hyp["clock_hours"],
            "arousal": arousal,
            "asleep": asleep,
            "melatonin": hyp["melatonin"],
            "cortisol": hyp["cortisol"],
            "hunger": hyp["hunger"],
            "heart_rate": bs["heart_rate"],
            "serotonin": nm.serotonin.effective,
            "dopamine": nm.dopamine.effective,
            "norepinephrine": nm.norepinephrine.effective,
            "acetylcholine": nm.acetylcholine.effective,
            "EI": ei,
            "DMN": net["DMN"],
            "TPN": net["TPN"],
            "Salience": net["Salience"],
            "mode": net["mode"],
            "mode_system": mode_system,
            "phantom_leak": thal["phantom_leak"],
            "conscious": conscious,
            "workspace": ws["workspace"],
            "conflict": conflict,
            "amygdala": limb["amygdala"],
            "aggression": limb["aggression"],
            "selected": gate["selected"],
            "channels_open": gate["channels_open"],
            "selection_margin": gate["selection_margin"],
            "chosen": chosen,
            "reward": reward,
            "rpe": delta,
            "motor_error": motor_error,
            "synapse_w": syn["w"],
            "synapse_late": syn["w_late"],
            "weight": syn["weight"],
            "Ca": syn["Ca"],
            "tissue_health": self.tissue_health,
            "valence": somatic_marker.valence,
            "emotional_arousal": somatic_marker.arousal,
            "dominance": somatic_marker.dominance,
            "primary_emotion": somatic_marker.primary_emotion,
            "headroom_gb": mem_snap.available_gb,
        }
        self.sim.t += dt
        return out

    def run(self, steps: int, world_fn) -> List[Dict]:
        return [self.step(world_fn(i, self.sim.t)) for i in range(steps)]

    def memory_score(self, corrupt: float = 0.25) -> float:
        h = self.hippocampus
        h.W = np.zeros_like(h.W)
        h.stored = 0
        h.episodes.clear()
        h.memory_matrix = None
        h.encode_gain = self._encode_capacity
        for e in self._experiences:
            h.store(e)
        return float(np.mean([h.recall_accuracy(e, corrupt) for e in self._experiences]))

    def summary(self, log: List[Dict]) -> Dict:
        arr = lambda k: np.array([d[k] for d in log], dtype=float)
        rewards = arr("reward")
        chosen = arr("chosen")
        acted = chosen >= 0
        motor = arr("motor_error")
        return {
            "ticks": len(log),
            "fraction_conscious": float(np.mean([d["conscious"] for d in log])),
            "fraction_asleep": float(np.mean([d["asleep"] for d in log])),
            "fraction_task_mode": float(np.mean([d["mode"] == "task" for d in log])),
            "actions_taken": int(acted.sum()),
            "total_reward": float(rewards.sum()),
            "mean_channels_open": float(arr("channels_open").mean()),
            "peak_aggression": float(arr("aggression").max()),
            "mean_phantom_leak": float(arr("phantom_leak").mean()),
            "final_motor_error": float(np.abs(motor[acted][-50:]).mean()) if acted.sum() > 50 else float("nan"),
            "memory_score": self.memory_score(),
            "final_weight": float(arr("weight")[-1]),
            "final_tissue_health": float(arr("tissue_health")[-1]),
            "mean_heart_rate": float(arr("heart_rate").mean()),
            "final_Q": self.value.Q.tolist(),
            "mean_valence": float(arr("valence").mean()),
            "min_headroom_gb": float(arr("headroom_gb").min()),
        }
