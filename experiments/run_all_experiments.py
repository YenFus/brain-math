"""
run_all_experiments.py - Master Benchmark Suite Runner.

Executes all biological neural and embodied experiments:
  1. Embodied Organism in Continuous Fluid Arena (Whole-brain closed-loop)
  2. Bio-Modular Triad vs Monolithic RL (Continuous Inverted Pendulum under violent shocks)
  3. C. elegans 302-Neuron Connectome (Sinusoidal undulation & escape reflex)
  4. Synthesis Hybrid Organism (Connectome + Cerebellar forward model + Neuromodulation)

Generates clean JSON telemetry for the interactive HTML5 visualizer dashboard.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.exp1_embodied_organism import EmbodiedOrganismExperiment
from experiments.exp2_modular_vs_rl import Experiment2Runner
from experiments.exp3_celegans_connectome import Experiment3Runner
from experiments.synthesis_hybrid_organism import SynthesisHybridOrganism


def main():
    print("=" * 75)
    print("🧠  ADVANCED NEURO-EMBODIED & CONNECTOME EXPERIMENTAL SUITE  🧠")
    print("=" * 75)

    start_time = time.time()

    # 1. Experiment 1
    t0 = time.time()
    exp1 = EmbodiedOrganismExperiment(steps=1000)
    res1 = exp1.run()
    t1_dur = round(time.time() - t0, 2)

    # 2. Experiment 2
    t0 = time.time()
    exp2 = Experiment2Runner(steps=1000)
    res2 = exp2.run()
    t2_dur = round(time.time() - t0, 2)

    # 3. Experiment 3
    t0 = time.time()
    exp3 = Experiment3Runner(steps=1000)
    res3 = exp3.run()
    t3_dur = round(time.time() - t0, 2)

    # 4. Synthesis Experiment
    t0 = time.time()
    synthesis = SynthesisHybridOrganism(steps=1000)
    res4 = synthesis.run()
    t4_dur = round(time.time() - t0, 2)

    total_dur = round(time.time() - start_time, 2)

    print("\n" + "=" * 75)
    print("📊  BENCHMARK EXECUTIVE SUMMARY")
    print("=" * 75)
    print(f"Total Execution Time: {total_dur}s (All runs inside Apple Silicon memory safety)")
    print("-" * 75)
    print(f"[Exp 1: Embodied Whole-Brain Organism] ({t1_dur}s)")
    print(f"  • Food Foraged:        {res1['food_eaten']} pellets")
    print(f"  • Foraging Efficiency: {res1['foraging_efficiency']} food/100m")
    print(f"  • Cost of Transport:   {res1['cost_of_transport']} energy/m")
    print(f"  • Collisions Avoided:  {res1['collisions']}")
    print("-" * 75)
    print(f"[Exp 2: Bio-Modular Triad vs Monolithic RL] ({t2_dur}s)")
    print("  Controller          | RMSE Error | Max Overshoot | Drops | Recovery")
    print("  --------------------|------------|---------------|-------|---------")
    for k, v in res2.items():
        print(f"  {k:<19} | {v['mean_rmse_deg']:>7.2f}°  | {v['max_overshoot_deg']:>10.2f}°  | {v['drop_failures']:>5} | {v['shock_recovery_steps']:>7} steps")
    print("-" * 75)
    print(f"[Exp 3: C. elegans 302-Neuron Connectome] ({t3_dur}s)")
    print(f"  • Forward Undulation:  {res3['forward_steps']} steps ({100 - res3['reversal_percentage']}%)")
    print(f"  • Reversal Reflex:     {res3['reversal_steps']} steps ({res3['reversal_percentage']}%)")
    print(f"  • Touch Response Time: {res3['escape_latency_ms']} ms")
    print("-" * 75)
    print(f"[Synthesis: Unified Hybrid Organism] ({t4_dur}s)")
    print(f"  • Worm Distance:       {res4['total_distance']} m")
    print(f"  • Foraging / Pellets:  {res4['food_eaten']}")
    print(f"  • Final Glucose:       {res4['final_glucose']}")
    print("=" * 75)
    print("🌐 Launch visualizer: open visualizer/neuroai_dashboard.html in any browser!")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
