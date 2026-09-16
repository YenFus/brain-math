"""
antigravity_bridge.py - Antigravity Co-Pilot Review Ledger & Collaboration Interface.

Enables seamless co-evolution between the autonomous agent and Antigravity (the AI assistant).
The agent posts proposals, code diffs, and emotional reflections to the Review Ledger.
Antigravity can inspect, critique, judge, approve, or update code alongside the local LLM.
When Antigravity approves or improves a patch, the agent receives high-reward feedback,
triggering a dopamine burst and synaptic consolidation.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class ReviewItem:
    review_id: str
    timestamp: float
    task_name: str
    emotion_summary: str
    valence: float
    arousal: float
    agent_hypothesis: str
    proposed_code_file: str
    code_diff: str
    sandbox_metrics: Dict[str, float]
    status: str = "PENDING_REVIEW"   # PENDING_REVIEW, APPROVED, REJECTED, MODIFIED
    antigravity_feedback: str = ""
    updated_code: Optional[str] = None


class AntigravityBridge:
    """Interface connecting the autonomous agent with Antigravity for meta-review."""

    def __init__(self, ledger_dir: Optional[Path] = None):
        if ledger_dir is None:
            ledger_dir = Path(__file__).resolve().parent.parent / "reviews"
        self.ledger_dir = ledger_dir
        self.ledger_dir.mkdir(parents=True, exist_ok=True)
        self.ledger_file = self.ledger_dir / "review_ledger.json"
        self.markdown_summary_file = self.ledger_dir / "latest_review.md"
        self._reviews: Dict[str, ReviewItem] = self._load_ledger()

    def _load_ledger(self) -> Dict[str, ReviewItem]:
        if not self.ledger_file.exists():
            return {}
        try:
            with open(self.ledger_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {k: ReviewItem(**v) for k, v in data.items()}
        except Exception:
            return {}

    def _save_ledger(self) -> None:
        data = {k: asdict(v) for k, v in self._reviews.items()}
        with open(self.ledger_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def submit_for_review(
        self,
        task_name: str,
        emotion_summary: str,
        valence: float,
        arousal: float,
        agent_hypothesis: str,
        proposed_code_file: str,
        code_diff: str,
        sandbox_metrics: Dict[str, float],
    ) -> str:
        """Agent submits a proposed self-improvement for Antigravity review."""
        review_id = f"REV-{int(time.time() * 1000) % 1000000:06d}"
        item = ReviewItem(
            review_id=review_id,
            timestamp=time.time(),
            task_name=task_name,
            emotion_summary=emotion_summary,
            valence=valence,
            arousal=arousal,
            agent_hypothesis=agent_hypothesis,
            proposed_code_file=proposed_code_file,
            code_diff=code_diff,
            sandbox_metrics=sandbox_metrics,
            status="PENDING_REVIEW",
        )
        self._reviews[review_id] = item
        self._save_ledger()

        # Write human-readable markdown summary for Antigravity / User
        md_content = (
            f"# Antigravity Review Request: {review_id}\n\n"
            f"- **Task**: {task_name}\n"
            f"- **Timestamp**: {time.ctime(item.timestamp)}\n"
            f"- **Agent Emotional State**: {emotion_summary} (Valence: {valence:+.2f}, Arousal: {arousal:.2f})\n"
            f"- **Target File**: `{proposed_code_file}`\n\n"
            f"### Hypothesis & Rationale\n{agent_hypothesis}\n\n"
            f"### Sandbox Metrics\n"
            f"```json\n{json.dumps(sandbox_metrics, indent=2)}\n```\n\n"
            f"### Proposed Code / Diff\n"
            f"```python\n{code_diff}\n```\n\n"
            f"---\n*Status*: `{item.status}`\n"
        )
        with open(self.markdown_summary_file, "w", encoding="utf-8") as f:
            f.write(md_content)

        return review_id

    def judge_review(
        self,
        review_id: str,
        approved: bool,
        feedback: str = "",
        replacement_code: Optional[str] = None,
    ) -> ReviewItem:
        """Antigravity or user provides meta-judgment and updates."""
        if review_id not in self._reviews:
            raise KeyError(f"Review {review_id} not found in ledger.")

        item = self._reviews[review_id]
        if replacement_code is not None:
            item.status = "MODIFIED"
            item.updated_code = replacement_code
        elif approved:
            item.status = "APPROVED"
        else:
            item.status = "REJECTED"

        item.antigravity_feedback = feedback
        self._save_ledger()
        return item

    def get_pending_reviews(self) -> List[ReviewItem]:
        return [item for item in self._reviews.values() if item.status == "PENDING_REVIEW"]

    def get_latest_review(self) -> Optional[ReviewItem]:
        if not self._reviews:
            return None
        return list(self._reviews.values())[-1]
