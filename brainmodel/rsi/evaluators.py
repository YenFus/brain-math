"""
evaluators.py - Isolated verification harness and multi-objective fitness evaluator.

Evaluates candidate policies and code mutations against rigorous benchmarks:
  1. Action Selection Accuracy under Ambiguity and High ACC Conflict.
  2. Neuromodulatory Temperature Stability.
  3. Execution Latency and Computational Overhead.
  4. Memory Headroom Compliance (guaranteeing >= 10.0 GB free).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np

from ..hardware_governor import HardwareGovernor


@dataclass
class EvaluationResult:
    score: float
    accuracy: float
    mean_latency_ms: float
    headroom_gb: float
    passed: bool
    num_tests: int
    error_message: Optional[str] = None


class EvaluatorHarness:
    """Benchmark suite for testing and verifying candidate policy mutations."""

    def __init__(self, governor: Optional[HardwareGovernor] = None):
        self.governor = governor or HardwareGovernor(min_headroom_gb=10.0)

    def evaluate_policy(self, policy_class, num_trials: int = 100) -> EvaluationResult:
        """Run standard benchmark suite against the given AgentPolicy class."""
        snap = self.governor.sample_memory()
        if not snap.is_safe:
            return EvaluationResult(
                score=0.0,
                accuracy=0.0,
                mean_latency_ms=0.0,
                headroom_gb=snap.available_gb,
                passed=False,
                num_tests=0,
                error_message="Memory headroom below 10 GB limit before evaluation.",
            )

        rng = np.random.default_rng(42)
        correct_count = 0
        latencies: List[float] = []

        try:
            # Benchmark 1: Decision under high conflict
            for i in range(num_trials):
                true_best = rng.integers(0, 3)
                q_vals = rng.uniform(0.2, 0.8, size=3)
                q_vals[true_best] += 0.4  # ground truth best option

                valence = 0.5 if (i % 2 == 0) else -0.2
                conflict = 0.8 if (i % 3 == 0) else 0.2

                t0 = time.perf_counter()
                scores = [
                    policy_class.heuristic_value(
                        action_idx=a,
                        context_features=np.zeros(4),
                        q_value=q_vals[a],
                        somatic_valence=valence if a == true_best else -valence,
                    )
                    for a in range(3)
                ]
                chosen = policy_class.evaluate_candidate_plans(scores, conflict=conflict)
                latencies.append((time.perf_counter() - t0) * 1000.0)

                if chosen == true_best:
                    correct_count += 1

            # Benchmark 2: Temperature scaling bounds
            for ne, da in [(1.8, 0.6), (0.7, 1.5), (1.0, 1.0)]:
                temp = policy_class.compute_exploration_temperature(
                    dopamine=da, norepinephrine=ne, conflict=0.5
                )
                if not (0.05 <= temp <= 3.0):
                    return EvaluationResult(
                        score=0.0,
                        accuracy=0.0,
                        mean_latency_ms=0.0,
                        headroom_gb=snap.available_gb,
                        passed=False,
                        num_tests=num_trials,
                        error_message=f"Temperature {temp} out of bounds for NE={ne}, DA={da}",
                    )

            accuracy = correct_count / num_trials
            mean_latency = float(np.mean(latencies))

            # Composite fitness score: rewards accuracy, penalizes latency
            fitness = float(accuracy * 100.0 - mean_latency * 2.0)
            passed = accuracy >= 0.75

            return EvaluationResult(
                score=fitness,
                accuracy=accuracy,
                mean_latency_ms=mean_latency,
                headroom_gb=snap.available_gb,
                passed=passed,
                num_tests=num_trials,
            )
        except Exception as e:
            return EvaluationResult(
                score=0.0,
                accuracy=0.0,
                mean_latency_ms=0.0,
                headroom_gb=snap.available_gb,
                passed=False,
                num_tests=num_trials,
                error_message=f"Runtime error in policy evaluation: {e}",
            )
