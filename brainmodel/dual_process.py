"""
dual_process.py - Kahneman System 1 (Fast, Heuristic) vs System 2 (Slow, Deliberative).

Cognitive Formalism:
  System 1: Fast, automated, reflexive habit execution (Basal Ganglia striatal gating).
  System 2: Deliberative, effortful planning, tree search / MCTS (Prefrontal ACC + DLPFC).

Sufficiency Principle (Chaiken 1980 / Dual-Process Theory):
  Decision-makers exert effort only until their actual confidence satisfies their
  desired confidence threshold (sufficiency threshold). High stakes, high ACC conflict,
  or high unexpected uncertainty (Norepinephrine) elevate the sufficiency threshold,
  forcing a switch from System 1 heuristics to System 2 deliberative search.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np

from .core import sigmoid


@dataclass
class DualProcessDecision:
    """Manages the arbitration between System 1 (fast habit) and System 2 (deep deliberation)."""

    conflict_threshold: float = 0.55    # ACC conflict level that recruits System 2
    vigilance_threshold: float = 1.30   # Norepinephrine level that forces deliberation
    base_sufficiency: float = 0.70      # default confidence needed to accept heuristic
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))

    def evaluate_mode(
        self,
        heuristic_confidence: float,
        acc_conflict: float,
        norepinephrine: float = 1.0,
        stakes: float = 0.0,
    ) -> Dict:
        """Determine whether System 1 is sufficient or System 2 must engage."""
        # Sufficiency threshold rises with stakes, conflict, and vigilance (NE)
        required_confidence = float(np.clip(
            self.base_sufficiency + 0.25 * stakes + 0.20 * acc_conflict + 0.15 * max(0.0, norepinephrine - 1.0),
            0.1, 0.98
        ))

        confidence_gap = required_confidence - heuristic_confidence
        must_deliberate = (
            confidence_gap > 0.0
            or acc_conflict >= self.conflict_threshold
            or norepinephrine >= self.vigilance_threshold
        )

        return {
            "mode": "SYSTEM_2_DELIBERATION" if must_deliberate else "SYSTEM_1_HEURISTIC",
            "required_confidence": required_confidence,
            "confidence_gap": float(confidence_gap),
            "must_deliberate": must_deliberate,
        }

    def deliberate_search(
        self,
        candidate_values: np.ndarray,
        rollout_fn: Optional[Callable[[int], float]] = None,
        budget: int = 8,
    ) -> Tuple[int, float]:
        """System 2 Deliberation: Monte Carlo / Tree-search rollout over candidates."""
        n_candidates = len(candidate_values)
        if rollout_fn is None or budget <= 0:
            # Simple analytical reflection: refine noisy initial estimates
            refined_values = np.asarray(candidate_values, dtype=float) + 0.05 * self.rng.standard_normal(n_candidates)
            chosen = int(np.argmax(refined_values))
            return chosen, float(refined_values[chosen])

        # MCTS / rollout sampling
        scores = np.zeros(n_candidates)
        counts = np.zeros(n_candidates)

        for _ in range(budget):
            # Upper Confidence Bound (UCB1) selection
            total_visits = max(1, np.sum(counts))
            ucb = (scores / np.maximum(counts, 1e-6)) + 1.2 * np.sqrt(np.log(total_visits) / np.maximum(counts, 1e-6))
            action = int(np.argmax(ucb))

            outcome = rollout_fn(action)
            counts[action] += 1
            scores[action] += outcome

        final_values = scores / np.maximum(counts, 1)
        best_action = int(np.argmax(final_values))
        return best_action, float(final_values[best_action])


# --- Backwards compatibility for original demo.py -----------------------------

@dataclass
class HeuristicSystematic:
    """Heuristic-Systematic Model (HSM) implementing the sufficiency principle."""

    base_sufficiency: float = 0.7
    dual_proc: DualProcessDecision = field(default_factory=DualProcessDecision)

    def judge(self, cue: float, evid: float, stakes: float = 0.0) -> Dict:
        heuristic_confidence = float(abs(cue))
        eval_res = self.dual_proc.evaluate_mode(
            heuristic_confidence=heuristic_confidence,
            acc_conflict=0.3 if abs(cue - evid) > 0.5 else 0.1,
            stakes=stakes,
        )

        if not eval_res["must_deliberate"]:
            judgment = 1 if cue > 0.5 else 0
            conf = heuristic_confidence
            mode = "heuristic"
        else:
            # Systematic deliberation integrates detailed evidence
            integrated = 0.3 * cue + 0.7 * evid
            judgment = 1 if integrated > 0.0 else 0
            conf = float(np.clip(0.5 + 0.5 * abs(integrated), 0.0, 1.0))
            mode = "systematic"

        return {"mode": mode, "judgment": judgment, "confidence": conf}
