"""
train_lora.py - Completes the Stage 1 Self-Improvement Cycle via MLX LoRA Fine-Tuning.

1. Ingests verified training traces from `data/verified_training_traces.jsonl`.
2. Converts into standard ChatML format (`train.jsonl` and `valid.jsonl`).
3. Executes a fast, lightweight LoRA fine-tuning session (Low-Rank Adaptation) using `mlx_lm`.
4. Saves adapter checkpoints to `llm_rsi_stage1/adapters/`.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm_rsi_stage1.rollout_generator import SYSTEM_PROMPT


def prepare_lora_dataset(traces_path: Path, output_dir: Path, val_split: float = 0.2):
    """Converts verified traces into ChatDataset JSONL format for MLX LoRA."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if not traces_path.exists():
        raise FileNotFoundError(f"Traces file not found: {traces_path}")

    records = []
    with open(traces_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    if not records:
        raise ValueError("No verified traces available for fine-tuning.")

    formatted_samples = []
    for r in records:
        assistant_content = f"{r.get('reasoning_trace', '').strip()}\n\n```python\n{r.get('verified_code', '').strip()}\n```"
        formatted_samples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT.strip()},
                {"role": "user", "content": r["prompt"].strip()},
                {"role": "assistant", "content": assistant_content.strip()}
            ]
        })

    # Train / Val Split
    n_val = max(1, int(len(formatted_samples) * val_split)) if len(formatted_samples) > 2 else 1
    val_samples = formatted_samples[:n_val]
    train_samples = formatted_samples[n_val:] if len(formatted_samples) > 1 else formatted_samples

    train_file = output_dir / "train.jsonl"
    valid_file = output_dir / "valid.jsonl"

    with open(train_file, "w", encoding="utf-8") as f:
        for s in train_samples:
            f.write(json.dumps(s) + "\n")

    with open(valid_file, "w", encoding="utf-8") as f:
        for s in val_samples:
            f.write(json.dumps(s) + "\n")

    print(f"✅ Prepared LoRA dataset in {output_dir}:")
    print(f"   • Training samples   : {len(train_samples)}")
    print(f"   • Validation samples : {len(val_samples)}")

    return train_file, valid_file


def run_lora_finetuning(
    model_id: str = "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit",
    iters: int = 15,
    batch_size: int = 1,
    learning_rate: float = 1e-4,
    adapter_path: Path = None
):
    """Executes mlx_lm lora command on the curated verified dataset."""
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data" / "lora_data"
    traces_file = base_dir / "data" / "verified_training_traces.jsonl"

    if adapter_path is None:
        adapter_path = base_dir / "adapters"
    adapter_path.mkdir(parents=True, exist_ok=True)

    prepare_lora_dataset(traces_file, data_dir)

    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model", model_id,
        "--train",
        "--data", str(data_dir),
        "--iters", str(iters),
        "--batch-size", str(batch_size),
        "--learning-rate", str(learning_rate),
        "--adapter-path", str(adapter_path),
        "--save-every", str(iters),
        "--steps-per-report", "5"
    ]

    print("\n" + "=" * 80)
    print("🚀 LAUNCHING STAGE 1 LoRA ADAPTER FINE-TUNING VIA MLX METAL")
    print("=" * 80)
    print(f"Command: {' '.join(cmd)}\n")

    res = subprocess.run(cmd, text=True)
    if res.returncode == 0:
        print("\n" + "=" * 80)
        print("✅ LoRA ADAPTER TRAINING COMPLETE!")
        print(f"   • Adapter Checkpoint: {adapter_path / 'adapters.safetensors'}")
        print("   • Cycle Closed: Verified code traces incorporated back into model weights.")
        print("=" * 80 + "\n")
    else:
        print(f"❌ LoRA training exited with code {res.returncode}")

    return res.returncode


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune LoRA on verified traces.")
    parser.add_argument("--model", type=str, default="mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit")
    parser.add_argument("--iters", type=int, default=15, help="Number of LoRA optimization iterations.")
    args = parser.parse_args()

    run_lora_finetuning(model_id=args.model, iters=args.iters)
