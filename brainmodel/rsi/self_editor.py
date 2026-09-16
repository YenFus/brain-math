"""
self_editor.py - Gödel Agent Self-Referential Code Modification Engine.

Gives the agent complete freedom to inspect, patch, refactor, and evolve its own
source code with automated syntax parsing, isolated sandbox testing, and atomic rollback guarantees.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .evaluators import EvaluationResult, EvaluatorHarness


@dataclass
class PatchRecord:
    patch_id: str
    timestamp: float
    target_file: str
    description: str
    fitness_before: float
    fitness_after: float
    passed: bool
    status: str      # 'COMMITTED', 'ROLLED_BACK', 'SYNTAX_ERROR'
    error: Optional[str] = None


class SelfEditor:
    """Self-referential code modifier allowing the agent to safely evolve its own algorithms."""

    def __init__(self, workspace_root: Optional[Path] = None):
        if workspace_root is None:
            workspace_root = Path(__file__).resolve().parent.parent.parent
        self.workspace_root = workspace_root
        self.snapshot_dir = self.workspace_root / "brainmodel" / "rsi" / "snapshots"
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.evaluator = EvaluatorHarness()
        self.history: List[PatchRecord] = []

    def read_module_code(self, relative_path: str = "brainmodel/rsi/agent_policy.py") -> str:
        """Inspect the current source code of any module in the workspace."""
        file_path = self.workspace_root / relative_path
        if not file_path.exists():
            raise FileNotFoundError(f"Target module {relative_path} does not exist.")
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    def apply_code_patch(
        self,
        new_code: str,
        relative_path: str = "brainmodel/rsi/agent_policy.py",
        description: str = "Autonomous policy optimization",
    ) -> Dict:
        """Atomically test, evaluate, and commit or rollback a code modification."""
        file_path = self.workspace_root / relative_path
        if not file_path.exists():
            return {"success": False, "status": "FILE_NOT_FOUND", "error": f"File {relative_path} not found"}

        patch_id = f"PATCH-{int(time.time() * 1000) % 1000000:06d}"

        # 1. Static Syntax Validation
        try:
            ast.parse(new_code)
        except SyntaxError as e:
            rec = PatchRecord(
                patch_id=patch_id,
                timestamp=time.time(),
                target_file=relative_path,
                description=description,
                fitness_before=0.0,
                fitness_after=0.0,
                passed=False,
                status="SYNTAX_ERROR",
                error=str(e),
            )
            self.history.append(rec)
            return {"success": False, "status": "SYNTAX_ERROR", "error": f"SyntaxError: {e}"}

        # 2. Benchmark current baseline before change
        current_code = self.read_module_code(relative_path)
        baseline_eval = self._test_code_in_isolation(current_code)
        fitness_before = baseline_eval.score if baseline_eval.passed else 50.0

        # 3. Create snapshot backup
        snapshot_file = self.snapshot_dir / f"{file_path.name}.{patch_id}.bak"
        shutil.copy2(file_path, snapshot_file)

        # 4. Write new code to disk
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(new_code)

            # 5. Evaluate the newly patched code
            candidate_eval = self._test_code_in_isolation(new_code)

            # Acceptance criterion: tests must pass AND fitness must not regress
            improved = candidate_eval.passed and (candidate_eval.score >= (fitness_before - 1e-4))

            if improved:
                # COMMIT the patch
                rec = PatchRecord(
                    patch_id=patch_id,
                    timestamp=time.time(),
                    target_file=relative_path,
                    description=description,
                    fitness_before=fitness_before,
                    fitness_after=candidate_eval.score,
                    passed=True,
                    status="COMMITTED",
                )
                self.history.append(rec)
                return {
                    "success": True,
                    "status": "COMMITTED",
                    "patch_id": patch_id,
                    "fitness_before": fitness_before,
                    "fitness_after": candidate_eval.score,
                    "accuracy": candidate_eval.accuracy,
                    "latency_ms": candidate_eval.mean_latency_ms,
                    "snapshot": str(snapshot_file),
                }
            else:
                # REGRESSION or FAILURE -> Immediate Rollback
                shutil.copy2(snapshot_file, file_path)
                rec = PatchRecord(
                    patch_id=patch_id,
                    timestamp=time.time(),
                    target_file=relative_path,
                    description=description,
                    fitness_before=fitness_before,
                    fitness_after=candidate_eval.score,
                    passed=False,
                    status="ROLLED_BACK",
                    error=candidate_eval.error_message or "Fitness score regressed.",
                )
                self.history.append(rec)
                return {
                    "success": False,
                    "status": "ROLLED_BACK",
                    "patch_id": patch_id,
                    "fitness_before": fitness_before,
                    "candidate_score": candidate_eval.score,
                    "error": rec.error,
                }
        except Exception as e:
            # Emergency rollback
            if snapshot_file.exists():
                shutil.copy2(snapshot_file, file_path)
            return {"success": False, "status": "ERROR_ROLLED_BACK", "error": str(e)}

    def _test_code_in_isolation(self, code_str: str) -> EvaluationResult:
        """Execute code in an isolated local module namespace and run benchmark suite."""
        try:
            mod_namespace = {}
            exec(compile(code_str, "<candidate_policy>", "exec"), mod_namespace)
            if "AgentPolicy" not in mod_namespace:
                return EvaluationResult(
                    score=0.0, accuracy=0.0, mean_latency_ms=0.0,
                    headroom_gb=15.0, passed=False, num_tests=0,
                    error_message="AgentPolicy class not found in code.",
                )
            policy_cls = mod_namespace["AgentPolicy"]
            return self.evaluator.evaluate_policy(policy_cls)
        except Exception as e:
            return EvaluationResult(
                score=0.0, accuracy=0.0, mean_latency_ms=0.0,
                headroom_gb=15.0, passed=False, num_tests=0,
                error_message=f"Isolation test error: {e}",
            )
