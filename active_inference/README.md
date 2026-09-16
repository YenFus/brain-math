# 🧠 Active Inference (`active_inference`)

Simulations demonstrating Karl Friston's Free Energy Principle, comparing it against standard navigation techniques.

## How to Run
```bash
python3 active_inference/benchmark_fep_vs_rl.py
```

## What It Shows
The agent navigates a maze with sensory traps (like food scent leaking through walls). Active inference uses expected information gain (epistemic value) to turn away from dead ends and explore open paths.
