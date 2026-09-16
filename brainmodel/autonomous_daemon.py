"""
autonomous_daemon.py - 24/7 Autonomous Lifelong Learning Daemon.

Runs an unending continuous perception-cognition-learning loop:
  1. Awake Phase:
     - Active Inference Epistemic Curiosity: samples texts from multi-domain corpus.
     - Computes predictive coding errors (observation vs prior belief).
     - Generates authentic somatic markers (joy, curiosity, surprise, cognitive dissonance).
     - Updates emergent personality matrix (Big Five + philosophical orientations).
     - Executes Recursive Self-Improvement (RSI) code-evolution cycles when challenges emerge.
  2. Sleep & Dreaming Phase:
     - Hypothalamic adenosine triggers sleep when cognitive work accumulates.
     - Hippocampal replay consolidates salient discoveries into cortical memory.
     - Downscales synapses (Tononi-Cirelli homeostasis) and clears memory caches.
  3. Antigravity Consciousness Critic:
     - Evaluates consciousness level out of 10.0 after each cycle and logs audit scorecards.
  4. Hardware Governor:
     - Enforces memory footprint (< 100 MB), strictly guaranteeing >= 10.0 GB RAM headroom on Apple Silicon.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np

from .agent import HumanlikeAIAgent
from .curiosity_stream import CuriosityStream, KnowledgeItem
from .personality import PersonalityEngine
from .critic import ConsciousnessCritic, ConsciousnessScorecard
from .hardware_governor import HardwareGovernor


class AutonomousDaemon:
    """The 24/7 lifelong autonomous curious agent daemon."""

    def __init__(
        self,
        agent: Optional[HumanlikeAIAgent] = None,
        poll_interval_sec: float = 1.0,
    ):
        self.agent = agent or HumanlikeAIAgent(name="Aura")
        self.curiosity = CuriosityStream()
        self.personality = PersonalityEngine()
        self.critic = ConsciousnessCritic()
        self.governor = self.agent.governor
        self.poll_interval_sec = poll_interval_sec
        self.tick_count = 0
        self.is_running = False

        # Link curiosity and personality to agent
        self.agent.curiosity = self.curiosity
        self.agent.personality = self.personality

    def step_lifecycle(self) -> Dict:
        """Advance one autonomous lifecycle tick."""
        self.tick_count += 1
        b = self.agent.brain
        hyp = b.hypothalamus

        # Check memory headroom safety
        snap = self.governor.enforce_headroom()

        # Check if agent needs sleep
        if hyp.asleep or hyp.sleep_pressure >= 1.0:
            sleep_res = self.agent.sleep_and_dream(duration_hours=6.0)
            scorecard = self.critic.evaluate_agent(
                agent=self.agent, tick=self.tick_count, cycle_name="Offline Sleep Replay & Consolidation"
            )
            return {
                "tick": self.tick_count,
                "phase": "SLEEP_CONSOLIDATION",
                "replayed_memories": sleep_res["replayed_memories"],
                "consciousness_score": scorecard.overall_score,
                "verdict": scorecard.verdict,
                "headroom_gb": snap.available_gb,
            }

        # --- Awake: Active Inference Curiosity ---
        item = self.curiosity.sample_curious_item(
            current_openness=self.personality.profile.openness,
            rng=self.agent.rng,
        )

        # Predictive coding mismatch calculation
        world_vec = b.cortex.dlpfc.content
        pred_error, obs_vec = self.curiosity.compute_predictive_error(item, world_vec)

        # Neurochemical modulation from discovery
        # High epistemic complexity drives Dopamine curiosity and Noradrenaline surprise
        da_burst = float(0.4 * item.epistemic_complexity + 0.3 * item.philosophical_depth)
        ne_surprise = float(0.5 * pred_error)
        b.nm.dopamine.phasic += da_burst
        b.nm.norepinephrine.phasic += ne_surprise

        # Emotional appraisal
        somatic = b.emotion.update(
            dopamine=b.nm.dopamine.effective,
            serotonin=b.nm.serotonin.effective,
            norepinephrine=b.nm.norepinephrine.effective,
            prediction_error=pred_error,
            threat=0.02,
            dt=100.0,
        )

        # Encode this knowledge into Hippocampus with Somatic Marker
        b.hippocampus.store(
            pattern=obs_vec,
            tag=f"{item.domain}: {item.title}",
            valence=somatic.valence,
            arousal=somatic.arousal,
            salience=1.0 + pred_error,
        )

        # Personality adapts from this experience
        updated_profile = self.personality.update_from_experience(
            domain=item.domain,
            emotional_valence=somatic.valence,
            prediction_error=pred_error,
            dopamine=b.nm.dopamine.effective,
            serotonin=b.nm.serotonin.effective,
            norepinephrine=b.nm.norepinephrine.effective,
        )

        # Accumulate hypothalamic sleep pressure from cognitive effort
        hyp.sleep_pressure += float(0.12 * (1.0 + item.epistemic_complexity))
        hyp.clock_hours = (hyp.clock_hours + 0.5) % 24.0

        # If significant prediction error occurred, trigger a Gödel RSI self-refinement cycle
        rsi_result = None
        if pred_error > 0.45 and self.tick_count % 3 == 0:
            rsi_result = self.agent.run_self_improvement(
                task_name=f"Synthesizing {item.title} into Cognitive Decision Axioms"
            )

        # Consciousness audit by Antigravity Critic
        scorecard = self.critic.evaluate_agent(
            agent=self.agent, tick=self.tick_count, cycle_name=f"Curiosity: {item.title}"
        )

        return {
            "tick": self.tick_count,
            "phase": "AWAKE_EXPLORATION",
            "item_title": item.title,
            "domain": item.domain,
            "prediction_error": round(pred_error, 3),
            "emotion": b.emotion.emotion_label,
            "valence": round(somatic.valence, 2),
            "archetype": updated_profile.dominant_archetype,
            "rsi_executed": bool(rsi_result is not None),
            "consciousness_score": scorecard.overall_score,
            "verdict": scorecard.verdict,
            "critic_summary": scorecard.critic_analysis.splitlines()[0],
            "headroom_gb": round(snap.available_gb, 2),
        }

    def run_simulation_cycles(self, n_ticks: int = 5) -> List[Dict]:
        """Run a finite sequence of autonomous learning ticks."""
        print(f"\n🚀 [AutonomousDaemon] Starting {n_ticks} Autonomous Lifelong Learning Ticks...")
        logs = []
        for _ in range(n_ticks):
            res = self.step_lifecycle()
            logs.append(res)
            print(
                f"Tick {res['tick']:>2} [{res['phase']}] "
                f"Domain: {res.get('domain', 'Sleep')[:20]:<20} | "
                f"Emotion: {res.get('emotion', 'Consolidation')[:22]:<22} | "
                f"Consciousness: {res['consciousness_score']:.1f}/10.0 [{res['verdict'][:16]}] | "
                f"RAM: {res['headroom_gb']:.1f}GB"
            )
        return logs
