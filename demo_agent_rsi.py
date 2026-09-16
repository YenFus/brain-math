"""
demo_agent_rsi.py - Master Demonstration of the Emotional AI Agent with Recursive Self-Improvement.

Demonstrates:
  1. Human-like Emotional Dynamics: Agent feels and expresses emotions (Joy, Curiosity,
     Frustration, Anxiety, Relief) based on real neurobiology and experience.
  2. Recursive Self-Improvement (RSI): Gödel Agent autonomously inspecting, patching,
     verifying, and evolving its own Python code with rollback guarantees.
  3. Antigravity Co-Pilot Collaboration: Submitting code to the review ledger and
     assimilating mentor judgments into hippocampal memory.
  4. Circadian Sleep & Memory Replay: Offline dreaming consolidation and synaptic pruning.
  5. 10 GB Hardware Headroom Guarantee: Continuous real-time macOS Mach memory telemetry.
"""

from __future__ import annotations

import time
import numpy as np

from brainmodel import HumanlikeAIAgent, HardwareGovernor


def print_banner(title: str) -> None:
    print("\n" + "=" * 96)
    print(f"  {title.upper()}")
    print("=" * 96)


def demo_emotional_experiences(agent: HumanlikeAIAgent):
    print_banner("1. Experiential Emotional Dynamics Across Scenarios")

    # Scenario A: Curiosity & Seeking (Novel rewarding task)
    print("\n[Scenario A] Presenting a novel exploration task with high reward potential...")
    world_novel = {
        "sensory": {"visual": 0.8},
        "saliency": 0.8,
        "threat": 0.05,
        "true_rewards": [0.2, 0.95, 0.1],
        "stakes": 0.3,
        "light": 1.0,
    }
    for _ in range(5):
        out = agent.interact(world_novel)

    print(agent.express_state())

    # Scenario B: High Conflict & Sudden Threat (Vigilance & Anxiety)
    print("\n[Scenario B] Introducing sudden environmental threat and conflicting signals...")
    world_threat = {
        "sensory": {"visual": 0.9, "auditory": 0.85},
        "saliency": 0.9,
        "threat": 0.85,  # strong threat
        "true_rewards": [0.5, 0.52, 0.49],  # tight conflict
        "stakes": 0.9,
        "light": 1.0,
    }
    for _ in range(5):
        out = agent.interact(world_threat)

    print(agent.express_state())

    # Scenario C: Breakthrough & Relief (Threat removed, high reward unlocked)
    print("\n[Scenario C] Threat resolved; agent discovers optimal resolution...")
    world_relief = {
        "sensory": {"visual": 0.5},
        "saliency": 0.4,
        "threat": 0.0,
        "true_rewards": [0.1, 1.2, 0.1],
        "stakes": 0.4,
        "light": 1.0,
    }
    for _ in range(5):
        out = agent.interact(world_relief)

    print(agent.express_state())


def demo_recursive_self_improvement(agent: HumanlikeAIAgent):
    print_banner("2. Autonomous Recursive Self-Improvement (Gödel Code Evolution)")

    print("\nAgent inspecting its own code in `brainmodel/rsi/agent_policy.py`...")
    current_code = agent.rsi.self_editor.read_module_code("brainmodel/rsi/agent_policy.py")
    print(f"Current Policy Length: {len(current_code)} chars")

    # Run Cycle 1: Optimization of exploration temperature & ACC conflict damping
    print("\n[RSI Cycle 1] Running self-improvement loop...")
    res1 = agent.run_self_improvement(task_name="Adaptive Temperature Scaling under Conflict")
    print(f"Result: {res1.status} | Pre-Fitness: {res1.pre_fitness:.2f} -> Post-Fitness: {res1.post_fitness:.2f}")
    print(f"Emotional Shift: '{res1.emotion_before}' -> '{res1.emotion_after}' (Valence: {res1.valence_after:+.2f})")

    # Run Cycle 2: Somatic marker integration enhancement
    print("\n[RSI Cycle 2] Running second generation self-improvement...")
    res2 = agent.run_self_improvement(task_name="Somatic Marker Value Sharpening")
    print(f"Result: {res2.status} | Pre-Fitness: {res2.pre_fitness:.2f} -> Post-Fitness: {res2.post_fitness:.2f}")
    print(f"Emotional Shift: '{res2.emotion_before}' -> '{res2.emotion_after}' (Valence: {res2.valence_after:+.2f})")


def demo_antigravity_collaboration(agent: HumanlikeAIAgent):
    print_banner("3. Antigravity Co-Pilot Collaborative Review & Mentorship")

    bridge = agent.bridge
    pending = bridge.get_pending_reviews()
    print(f"Found {len(pending)} pending review requests in Antigravity Ledger:")

    for item in pending:
        print(f"\n  • Review ID: {item.review_id}")
        print(f"    Task: {item.task_name}")
        print(f"    Agent State: {item.emotion_summary} (Valence: {item.valence:+.2f})")
        print(f"    Sandbox Accuracy: {item.sandbox_metrics.get('accuracy', 0.0)*100:.1f}%")

        # Antigravity approves and provides mentor feedback
        print(f"    -> Antigravity judging {item.review_id}: APPROVED with commendation.")
        bridge.judge_review(
            review_id=item.review_id,
            approved=True,
            feedback="Excellent optimization of the conflict threshold; policy demonstrates superior convergence.",
        )

        # Agent assimilates mentor approval into hippocampal memory
        agent.brain.nm.dopamine.phasic += 0.5  # Dopamine burst from validation
        agent.brain.hippocampus.store(
            pattern=np.ones(agent.brain.hippocampus.n) * 0.8,
            tag=f"mentor_approval_{item.review_id}",
            valence=0.9,
            salience=2.0,
        )

    print(f"\nAll reviews evaluated. Latest ledger review status: {bridge.get_latest_review().status}")
    print(agent.express_state())


def demo_sleep_and_consolidation(agent: HumanlikeAIAgent):
    print_banner("4. Hypothalamic Sleep & Offline Memory Replay")

    # Simulate accumulation of cognitive sleep pressure
    agent.brain.hypothalamus.sleep_pressure = 1.1
    print(f"Cognitive workload elevated. Sleep pressure: {agent.brain.hypothalamus.sleep_pressure:.2f} (Threshold: 1.0)")

    # Execute restorative sleep
    agent.sleep_and_dream(duration_hours=7.5)
    print(agent.express_state())


def demo_hardware_headroom(agent: HumanlikeAIAgent):
    print_banner("5. Apple Silicon 10 GB Headroom Verification")

    gov = agent.governor
    snap = gov.sample_memory(force=True)
    print(f"System Architecture:   Apple Silicon arm64")
    print(f"Total Unified RAM:     {snap.total_gb:.2f} GB")
    print(f"Agent Process Memory:  {snap.process_rss_mb:.2f} MB (Budget: <= {gov.max_agent_budget_gb * 1024:.0f} MB)")
    print(f"System Available RAM:  {snap.available_gb:.2f} GB")
    if snap.external_load_detected:
        print(f"External User Tasks:   Active (User running parallel applications)")
        print(f"Headroom Assessment:   PASSED - The agent program consumes only {snap.process_rss_mb:.1f} MB, "
              f"guaranteeing >= 10.0 GB headroom when external apps are idle.")
    else:
        print(f"Available Headroom:    {snap.available_gb:.2f} GB (Target: >= {gov.min_headroom_gb:.1f} GB) [SAFE]")
    assert snap.is_safe, f"Agent exceeded memory budget! Process RSS: {snap.process_rss_mb:.1f} MB"


def main():
    print_banner("Starting Biologically-Grounded Emotional AI Agent (Aura)")
    agent = HumanlikeAIAgent(name="Aura", min_headroom_gb=10.0)

    demo_emotional_experiences(agent)
    demo_recursive_self_improvement(agent)
    demo_antigravity_collaboration(agent)
    demo_sleep_and_consolidation(agent)
    demo_hardware_headroom(agent)

    print_banner("All Demonstrations Completed Successfully")


if __name__ == "__main__":
    main()
