# 🔬 Core Embodiment & Connectome Experiments (`experiments`)

This folder contains the foundational simulations combining physical environments with neural controllers.

## How to Run
```bash
# Run all core experiments sequentially
python3 experiments/run_all_experiments.py

# Or run individual experiments:
python3 experiments/exp1_embodied_organism.py
python3 experiments/exp2_modular_vs_rl.py
python3 experiments/exp3_celegans_connectome.py
python3 experiments/synthesis_hybrid_organism.py
```

## What Each Script Does
- `exp1_embodied_organism.py`: An organism foraging in a 2D fluid arena using sensory whiskers and scent gradients.
- `exp2_modular_vs_rl.py`: An inverted pendulum balance task comparing a modular brain model against standard reinforcement learning.
- `exp3_celegans_connectome.py`: A biophysical simulation of the C. elegans worm nervous system generating crawling undulations and escape reflexes.
- `synthesis_hybrid_organism.py`: Combines the worm connectome with higher-level goal-directed navigation.
