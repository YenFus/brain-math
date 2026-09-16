"""
self_improvement_loop.py - Stage 1 Self-Taught Reasoner (STaR) Verification & Curation Engine.

Implements the Stage 1 (Improvement-Execution Autonomy) workflow from recent 2026 research:
1. Candidate Rollout Generation (Qwen2.5-Coder-1.5B)
2. Deterministic Subprocess Sandboxing (assert correctness)
3. Filtering & Curation (zero flawed code permitted into dataset)
4. Training Data Export (`data/verified_training_traces.jsonl`) for MLX LoRA Fine-Tuning
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm_rsi_stage1.benchmark_tasks import BENCHMARK_TASKS, CodingTask
from llm_rsi_stage1.sandbox_verifier import verify_solution, VerificationResult
from llm_rsi_stage1.rollout_generator import QwenRolloutGenerator, MockRolloutGenerator


def run_stage1_loop(
    num_cycles: int = 1,
    samples_per_task: int = 2,
    use_mock: bool = False,
    model_id: str = "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit",
    max_tasks: int = 8
) -> Dict:
    print("=" * 80)
    print("🤖 STAGE 1 LLM SELF-IMPROVEMENT PIPELINE: VERIFIABLE REASONER (STaR / RLVR)")
    print("=" * 80)
    print(f"  • Model Architecture : {model_id if not use_mock else 'Mock (Offline Verification)'}")
    print(f"  • Verification Engine: Isolated Subprocess Sandbox (3.0s Timeout)")
    print(f"  • Benchmark Tasks    : {min(max_tasks, len(BENCHMARK_TASKS))} tasks x {samples_per_task} rollouts/task")
    print(f"  • Scaffolding Level  : Stage 1 (Improvement-Execution Autonomy)")
    print("=" * 80 + "\n")

    if use_mock:
        generator = MockRolloutGenerator()
    else:
        try:
            generator = QwenRolloutGenerator(model_id=model_id)
        except Exception as e:
            print(f"⚠️ Failed to initialize live MLX generator ({e}). Falling back to mock generator.")
            generator = MockRolloutGenerator()
            use_mock = True

    data_dir = Path(__file__).resolve().parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    dataset_file = data_dir / "verified_training_traces.jsonl"

    tasks = BENCHMARK_TASKS[:max_tasks]
    verified_traces = []
    total_rollouts = 0
    total_passed = 0
    failure_counts = {"SYNTAX_ERROR": 0, "ASSERTION_ERROR": 0, "TIMEOUT": 0, "SECURITY_BLOCKED": 0, "OTHER": 0}

    start_time = time.time()

    for task_idx, task in enumerate(tasks, 1):
        print(f"[{task_idx}/{len(tasks)}] Evaluating '{task.name}' ({task.task_id})...")
        task_passed_any = False

        for sample_idx in range(samples_per_task):
            total_rollouts += 1
            temp = 0.2 + (0.3 * sample_idx)  # Anneal temperature for multi-attempt diversity

            if use_mock:
                rollout = generator.generate_candidate(task.task_id, temperature=temp)
            else:
                rollout = generator.generate_candidate(task.prompt, temperature=temp)

            if not rollout.is_code_extracted:
                print(f"    • Sample {sample_idx+1}: ❌ Failed to extract Python code block.")
                failure_counts["SYNTAX_ERROR"] += 1
                continue

            # External Sandboxed Verification
            res: VerificationResult = verify_solution(rollout.extracted_code, task.unit_tests)

            if res.passed:
                total_passed += 1
                task_passed_any = True
                print(f"    • Sample {sample_idx+1}: ✅ PASSED ({res.execution_time_sec:.3f}s) -> Added to Curated Dataset")

                # Store clean training trace (format ready for SFT / LoRA)
                trace_entry = {
                    "task_id": task.task_id,
                    "name": task.name,
                    "prompt": task.prompt,
                    "reasoning_trace": rollout.reasoning_trace,
                    "verified_code": rollout.extracted_code,
                    "execution_time_sec": res.execution_time_sec,
                    "verification_status": "VERIFIED_CORRECT"
                }
                verified_traces.append(trace_entry)
            else:
                err_key = res.error_type if res.error_type in failure_counts else "OTHER"
                failure_counts[err_key] += 1
                print(f"    • Sample {sample_idx+1}: ❌ {res.error_type} ({res.message[:60]}) -> Discarded")

        status_icon = "🟢" if task_passed_any else "🔴"
        print(f"  --> {status_icon} Task '{task.name}' resolution status: {'SOLVED' if task_passed_any else 'UNRESOLVED'}\n")

    elapsed_total = round(time.time() - start_time, 2)
    pass_rate_pct = round((total_passed / max(1, total_rollouts)) * 100.0, 1)

    # Write verified dataset
    with open(dataset_file, "w", encoding="utf-8") as f:
        for trace in verified_traces:
            f.write(json.dumps(trace) + "\n")

    print("=" * 80)
    print("📊 STAGE 1 ITERATION TELEMETRY & DATASET CURATION SUMMARY")
    print("=" * 80)
    print(f"  • Total Candidate Rollouts Generated : {total_rollouts}")
    print(f"  • Verified Correct Solutions (Passed): {total_passed} ({pass_rate_pct}%)")
    print(f"  • Discarded Flawed Rollouts (Failed) : {total_rollouts - total_passed}")
    print(f"    - Assertion Errors (Logical Bugs)  : {failure_counts['ASSERTION_ERROR']}")
    print(f"    - Syntax / Extraction Errors       : {failure_counts['SYNTAX_ERROR']}")
    print(f"    - Execution Timeouts (Infinite)    : {failure_counts['TIMEOUT']}")
    print(f"    - Security Policy Blocks           : {failure_counts['SECURITY_BLOCKED']}")
    print(f"  • Curated Clean Traces Saved         : {len(verified_traces)} samples -> {dataset_file.name}")
    print(f"  • Total Cycle Execution Time         : {elapsed_total}s")
    print("-" * 80)
    print("  VERIFIABLE STAGE 1 VERDICT: DATASET SANITIZED WITH ZERO UNCHECKED CODE")
    print("=" * 80 + "\n")

    summary = {
        "model_id": model_id if not use_mock else "mock_generator",
        "total_rollouts": total_rollouts,
        "total_passed": total_passed,
        "pass_rate_pct": pass_rate_pct,
        "failure_counts": failure_counts,
        "curated_dataset_samples": len(verified_traces),
        "dataset_path": str(dataset_file),
        "execution_time_sec": elapsed_total
    }

    with open(data_dir / "cycle_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Stage 1 LLM Self-Improvement Pipeline.")
    parser.add_argument("--mock", action="store_true", help="Run in mock mode for instant offline validation.")
    parser.add_argument("--model", type=str, default="mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit", help="Model ID.")
    parser.add_argument("--tasks", type=int, default=8, help="Number of tasks to evaluate.")
    parser.add_argument("--samples", type=int, default=2, help="Rollout samples per task.")
    args = parser.parse_args()

    run_stage1_loop(
        num_cycles=1,
        samples_per_task=args.samples,
        use_mock=args.mock,
        model_id=args.model,
        max_tasks=args.tasks
    )
