# 🤖 Stage 1 LLM Self-Improvement Pipeline (Qwen2.5-Coder-1.5B)

A practical, deterministic implementation of **Stage 1 (Improvement-Execution Autonomy)** for low-billion parameter models on Apple Silicon using `mlx_lm`.

---

## 🔬 Grounding in Recent Research (Sept 2026)

In the September 2026 paper *"The Last AI Built by Humans: Toward Genuine Recursive Self-Improvement"* (Shanghai Jiao Tong University, Tsinghua University, ByteDance), researchers categorized self-improvement into 5 distinct stages:

1. **Stage 1 (Improvement-Execution Autonomy)**: The AI executes a human-designed evaluation and fine-tuning routine (e.g. STaR, RLVR). The human designs the verification harness; the model explores solutions.
2. **Stage 2–4**: Strategy autonomy, dynamic environment discovery, and multi-domain adaptation.
3. **Stage 5 (Genuine Recursive Meta-Improvement)**: The AI autonomously invents and rewires the fundamental mechanisms that create future improvements without human scaffolding.

The researchers proved that **virtually all modern LLMs are strictly at Stage 1 or 2**. This repository implements a robust, real-world Stage 1 pipeline anchored to deterministic unit tests.

---

## ⚙️ How It Works

```
┌─────────────────────────────────┐
│ 1. 15 Benchmark Coding Tasks   │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│ 2. Qwen2.5-Coder-1.5B (MLX)     │  Generates candidate CoT + Python code
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│ 3. Isolated Subprocess Sandbox  │  Tests against strict unit assertions
└────────┬────────────────┬───────┘  (3.0s timeout, memory bounds)
         │ Pass           │ Fail
         ▼                ▼
┌─────────────────┐ ┌─────────────┐
│ 4. Curated JSONL│ │ 5. Discard  │  Zero unverified code permitted
└─────────────────┘ └─────────────┘
```

---

## 🚀 Quick Start

### 1. Run Unit Tests (Sandbox Safety & Isolation)
```bash
python3 -m unittest llm_rsi_stage1/tests/test_verifier.py
```

### 2. Run the Self-Improvement Loop (Offline Mock Mode)
Instant verification without downloading weights:
```bash
python3 llm_rsi_stage1/self_improvement_loop.py --mock
```

### 3. Run with Live Qwen2.5-Coder-1.5B (MLX Apple Silicon)
Runs inference locally via Apple Silicon Metal:
```bash
python3 llm_rsi_stage1/self_improvement_loop.py --tasks 5 --samples 2
```

---

## 📁 Output Artifacts
- `data/verified_training_traces.jsonl`: Curated training dataset where 100% of samples are mathematically and deterministically verified to pass unit tests.
- `data/cycle_summary.json`: Telemetry metrics, pass rates, and failure error breakdowns.
