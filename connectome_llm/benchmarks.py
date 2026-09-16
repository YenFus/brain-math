"""
benchmarks.py - Benchmark Suite evaluating Language-Guided Connectome Control.

Compares:
  1. Blind Random Walk (Random crawling without goal)
  2. Pure Chemotaxis (Gradient following without language modulation)
  3. Language-Guided Connectome (Our model with dynamic high-level mission scheduling)

Evaluates:
  - Foraging Yield (Pellets Eaten)
  - Collision Avoidance (Fewer wall/obstacle hits)
  - Arena Exploration Coverage (%)
  - Energy Efficiency (Pellets per 100m)
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, List
import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from connectome_llm.embodied_agent import LanguageGuidedConnectomeAgent


class ConnectomeBenchmarkSuite:
    """Runs comparative evaluation across 1000 physical simulation steps."""

    def __init__(self, steps: int = 1000, seed: int = 42):
        self.steps = steps
        self.seed = seed

    def run_trial(self, mode: str) -> Dict:
        agent = LanguageGuidedConnectomeAgent(seed=self.seed)

        # Mission schedule for language-guided mode
        # Step 0-300: "patrol the arena perimeter"
        # Step 300-700: "sprint and forage for food"
        # Step 700-1000: "dwell and feed on nutrient lawn"
        visited_grid = set()

        for step in range(self.steps):
            if mode == "language_guided":
                if step == 0:
                    agent.set_goal("patrol the arena perimeter")
                elif step == 300:
                    agent.set_goal("sprint and forage for food")
                elif step == 700:
                    agent.set_goal("dwell and feed on nutrient lawn")
            elif mode == "pure_chemotaxis":
                agent.set_goal("forage and explore")
            elif mode == "blind_random":
                # Suppress chemo attraction
                agent.policy.active_setpoint.chemo_attraction = 0.0
                agent.policy.active_setpoint.behavior_mode = "FORAGE"

            info = agent.step()

            # Record 5x5m spatial grid cell occupancy for coverage metric
            cell_x = int(info["head_x"] // 10.0)
            cell_y = int(info["head_y"] // 10.0)
            visited_grid.add((cell_x, cell_y))

        coverage_pct = round((len(visited_grid) / 100.0) * 100.0, 1)
        efficiency = round(agent.food_eaten / (agent.total_distance / 100.0 + 1e-4), 2)

        return {
            "mode": mode,
            "food_eaten": agent.food_eaten,
            "total_distance": round(agent.total_distance, 2),
            "collisions": agent.collisions,
            "arena_coverage_pct": coverage_pct,
            "foraging_efficiency": efficiency,
            "final_glucose": round(agent.glucose, 3),
        }

    def run(self) -> Dict:
        print("=" * 75)
        print("🪱  LANGUAGE-GUIDED CONNECTOME BENCHMARK SUITE  🪱")
        print("=" * 75)

        modes = ["blind_random", "pure_chemotaxis", "language_guided"]
        results = {}

        for m in modes:
            print(f"   • Running controller: {m} ({self.steps} steps)...")
            res = self.run_trial(m)
            results[m] = res
            print(f"     -> Food: {res['food_eaten']} | Collisions: {res['collisions']} | Coverage: {res['arena_coverage_pct']}% | Efficiency: {res['foraging_efficiency']} food/100m")

        print("\n" + "-" * 75)
        print("📊  BENCHMARK COMPARISON TABLE")
        print("-" * 75)
        print("  Controller          | Food Eaten | Collisions | Coverage | Efficiency")
        print("  --------------------|------------|------------|----------|-----------")
        for k, v in results.items():
            print(f"  {k:<19} | {v['food_eaten']:>10} | {v['collisions']:>10} | {v['arena_coverage_pct']:>7.1f}% | {v['foraging_efficiency']:>8.2f}")
        print("=" * 75 + "\n")

        # Save to connectome_llm/data/benchmark_results.json
        out_dir = Path(__file__).resolve().parent / "data"
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "benchmark_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        return results


if __name__ == "__main__":
    suite = ConnectomeBenchmarkSuite(steps=1000)
    suite.run()
