"""
interactive_chat.py - Interactive Terminal Console for Language-Guided Connectome Control.

Allows human users or autonomous agents to type natural language commands:
  - "patrol perimeter"
  - "sprint to north"
  - "evade obstacle"
  - "dwell and feed"
  - "standby"
and observe live descending neural currents and connectome swimming reactions.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from connectome_llm.embodied_agent import LanguageGuidedConnectomeAgent


def run_demo():
    print("=" * 70)
    print("🧠  C. ELEGANS LANGUAGE-GUIDED NEURAL CHAT DEMONSTRATION  🧠")
    print("=" * 70)

    agent = LanguageGuidedConnectomeAgent(seed=42)

    commands = [
        ("patrol the arena perimeter cautiously", 150),
        ("sprint towards the strong nutrient scent in the north", 200),
        ("emergency retreat! evade obstacle immediately", 100),
        ("dwell and feed on the nutrient lawn", 150),
    ]

    for cmd, duration in commands:
        print(f"\n💬 [Human Instruction]: \"{cmd}\"")
        agent.set_goal(cmd)

        for s in range(duration):
            info = agent.step()
            if s % 50 == 0:
                print(f"   Tick {s:>3}: Mode: {info['behavior_mode']:<7} | Pos: ({info['head_x']:>5.1f}, {info['head_y']:>5.1f}) | Vel: {info['velocity']:>5.2f} | Reversing: {str(info['is_reversing']):<5} | Food: {info['food_eaten']}")

    print("\n" + "=" * 70)
    print(f"✅ Mission Finished! Total Pellets: {agent.food_eaten} | Collisions: {agent.collisions} | Total Distance: {agent.total_distance:.1f}m")
    print("=" * 70)


if __name__ == "__main__":
    run_demo()
