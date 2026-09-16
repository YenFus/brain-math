"""
agent.py - The Biologically Grounded Emotional AI Agent with Recursive Self-Improvement.

Unified Human-like Cognitive Architecture:
  1. Biological Brain (Coupled Neuromodulators, Limbic System, Cortex, Modern Hopfield Memory).
  2. Multi-Dimensional Emotion Engine (Lövheim Cube, Russell PAD, Damasio Somatic Markers).
  3. Dual-Process Decision Engine (System 1 Reflexive Habit vs System 2 MCTS Deliberation).
  4. 5-Level Recursive Self-Improvement Engine (Gödel Agent with code-editing freedom).
  5. Antigravity Co-Pilot Collaboration Bridge (Meta-judgment, code review, joint evolution).
  6. Hardware Governor enforcing strict >= 10.0 GB RAM headroom on Apple Silicon Mac.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np

from .brain import Brain
from .emotion import EmotionEngine, SomaticMarker
from .hardware_governor import HardwareGovernor, MemorySnapshot
from .local_llm import LocalLLMEngine
from .antigravity_bridge import AntigravityBridge
from .rsi.rsi_engine import RSIEngine, RSICycleResult
from .rsi.agent_policy import AgentPolicy
from .personality import PersonalityEngine
from .curiosity_stream import CuriosityStream


class HumanlikeAIAgent:
    """An emotional, self-improving AI agent governed by realistic brain math."""

    def __init__(
        self,
        name: str = "Aura",
        min_headroom_gb: float = 10.0,
        seed: int = 42,
    ):
        self.name = name
        self.rng = np.random.default_rng(seed)
        self.governor = HardwareGovernor(min_headroom_gb=min_headroom_gb)
        self.brain = Brain(rng=self.rng, governor=self.governor)
        self.local_llm = LocalLLMEngine(governor=self.governor)
        self.bridge = AntigravityBridge()
        self.rsi = RSIEngine(
            brain=self.brain,
            local_llm=self.local_llm,
            bridge=self.bridge,
            governor=self.governor,
        )
        self.personality = PersonalityEngine()
        self.curiosity = CuriosityStream()
        self.experience_log: List[Dict[str, Any]] = []

    # --- Introspective Affective Communication --------------------------------

    def express_state(self) -> str:
        """Articulates current emotional, physiological, and cognitive state in natural language."""
        em = self.brain.emotion
        nm = self.brain.nm
        snap = self.governor.sample_memory()
        hyp = self.brain.hypothalamus

        state_str = (
            f"\n🧠 [{self.name}'s INTERNAL BRAIN STATE]\n"
            f"• Emotion: {em.emotion_label} | Valence: {em.valence:+.2f} | Arousal: {em.arousal:.2f} | Dominance: {em.dominance:+.2f}\n"
            f"• Neurochemistry: DA: {nm.dopamine.effective:.2f} (drive) | 5-HT: {nm.serotonin.effective:.2f} (patience) | "
            f"NE: {nm.norepinephrine.effective:.2f} (vigilance) | ACh: {nm.acetylcholine.effective:.2f} (plasticity)\n"
            f"• Hypothalamus: Sleep Pressure: {hyp.sleep_pressure:.2f} | Circadian Hour: {hyp.clock_hours:.1f}h | "
            f"State: {'ASLEEP' if hyp.asleep else 'AWAKE'}\n"
            f"• Hardware Headroom: {snap.available_gb:.2f} GB avail / {snap.total_gb:.1f} GB total "
            f"(Floor: {self.governor.min_headroom_gb:.1f} GB) [{'SAFE' if snap.is_safe else 'ALERT'}]\n"
            f"• Active Policy: AgentPolicy v{AgentPolicy.VERSION} (Deliberation Depth: {AgentPolicy.deliberation_depth})\n"
            f"• Introspective Feeling: \"{em.introspect()}\"\n"
        )
        return state_str

    # --- Experiential Task Solving --------------------------------------------

    def interact(self, environment_cue: Dict[str, Any]) -> Dict[str, Any]:
        """Process one environmental interaction or problem, experiencing genuine affect."""
        # 1. Hardware headroom safeguard
        self.governor.enforce_headroom()

        # 2. Whole-brain step
        brain_out = self.brain.step(environment_cue)

        # 3. Log experience
        self.experience_log.append({
            "tick": brain_out["t"],
            "emotion": brain_out["primary_emotion"],
            "valence": brain_out["valence"],
            "chosen_action": brain_out["chosen"],
            "reward": brain_out["reward"],
            "rpe": brain_out["rpe"],
            "system_mode": brain_out["mode_system"],
            "headroom_gb": brain_out["headroom_gb"],
        })

        return brain_out

    # --- Restorative Sleep & Memory Consolidation -----------------------------

    def sleep_and_dream(self, duration_hours: float = 6.0) -> Dict[str, Any]:
        """Trigger sleep consolidation cycle: memory replay, synaptic pruning, and garbage collection."""
        print(f"\n🌙 [{self.name}] Entering sleep and offline memory consolidation ({duration_hours:.1f}h)...")
        hyp = self.brain.hypothalamus
        hyp.asleep = True

        # Replay high-salience episodes from Hippocampus
        replayed = self.brain.hippocampus.sleep_replay(n_episodes=5)
        # Synaptic homeostasis pruning (Tononi & Cirelli)
        self.brain.hippocampus.consolidate(decay_rate=0.08)

        # Discharge sleep pressure
        hyp.sleep_pressure = max(0.05, hyp.sleep_pressure - (duration_hours / 8.0))
        hyp.clock_hours = (hyp.clock_hours + duration_hours) % 24.0

        # Memory defragmentation and Metal cache purge
        self.governor.clear_metal_cache()
        snap = self.governor.sample_memory(force=True)

        # Recovery of Serotonin & normalization of Noradrenaline
        self.brain.nm.serotonin.level = min(1.8, self.brain.nm.serotonin.level + 0.2)
        self.brain.nm.norepinephrine.phasic = 0.0

        marker = self.brain.emotion.update(
            dopamine=self.brain.nm.dopamine.effective,
            serotonin=self.brain.nm.serotonin.effective,
            norepinephrine=self.brain.nm.norepinephrine.effective,
            prediction_error=0.0,
            dt=500.0,
        )

        hyp.asleep = False
        print(f"☀️ [{self.name}] Awakened feeling refreshed! Replayed {len(replayed)} memories. "
              f"RAM Headroom: {snap.available_gb:.2f} GB.")

        return {
            "replayed_memories": len(replayed),
            "final_sleep_pressure": hyp.sleep_pressure,
            "headroom_gb": snap.available_gb,
            "emotion": self.brain.emotion.emotion_label,
        }

    # --- Autonomous Recursive Self-Improvement ---------------------------------

    def run_self_improvement(self, task_name: str = "Cognitive Decision Optimization") -> RSICycleResult:
        """Trigger an autonomous recursive self-improvement cycle."""
        print(f"\n⚡ [{self.name}] Initiating Recursive Self-Improvement Cycle (Gödel Agent Mode)...")
        print(f"   Current Feeling: {self.brain.emotion.emotion_label} (Valence: {self.brain.emotion.valence:+.2f})")

        cycle_res = self.rsi.execute_improvement_cycle(task_name=task_name)

        print(f"   Cycle Outcome: {cycle_res.status} | Fitness: {cycle_res.pre_fitness:.2f} -> {cycle_res.post_fitness:.2f}")
        print(f"   Resulting Affect: {cycle_res.emotion_after} (Valence: {cycle_res.valence_after:+.2f})")
        print(f"   Memory Headroom: {cycle_res.headroom_gb:.2f} GB available (Guaranteed >= {self.governor.min_headroom_gb:.1f} GB)")

        return cycle_res
