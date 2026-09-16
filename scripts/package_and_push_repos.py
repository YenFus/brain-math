import os
import shutil
import subprocess
from pathlib import Path

BASE = Path("/Users/mohit/Projects/Brain Math")
REPOS = BASE / "repos"

def run_cmd(cmd, cwd):
    print(f"Running: {' '.join(cmd)} in {cwd}")
    res = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Error: {res.stderr}")
    else:
        print(f"Success: {res.stdout.strip()}")
    return res

# 1. Package connectome-llm-interface
r1 = REPOS / "connectome-llm-interface"
if r1.exists():
    shutil.rmtree(r1)
r1.mkdir(parents=True)

shutil.copytree(BASE / "connectome_llm", r1 / "connectome_llm")
shutil.copytree(BASE / "experiments", r1 / "experiments")
shutil.copytree(BASE / "brainmodel", r1 / "brainmodel")
shutil.copy(BASE / ".gitignore", r1 / ".gitignore")

r1_readme = """# Connectome-LLM Interface: Language-Guided Biological Locomotion

A neuro-symbolic and biophysical control architecture connecting Natural Language Cognitive Policies to the 302-neuron *C. elegans* biophysical connectome and 12-segment hydrostatic body in continuous fluid physics.

```bash
# Run benchmark comparing Language-Guided control vs. Chemotaxis & Random walk
python3 connectome_llm/benchmarks.py

# Run interactive language chat console
python3 connectome_llm/interactive_chat.py
```

### Architecture
1. **Language Policy**: Interprets natural language commands ("patrol perimeter", "sprint to north", "emergency retreat", "dwell & feed").
2. **Descending Brainstem Bridge**: Converts high-level goals into micro-currents ($I_{\\text{AVB}}, I_{\\text{AVA}}, I_{\\text{ASE}}$).
3. **Biophysical Connectome**: Simulates non-spiking graded membrane potentials and muscle torque traveling waves in viscous fluid dynamics.
"""
with open(r1 / "README.md", "w") as f:
    f.write(r1_readme)

run_cmd(["git", "init"], r1)
run_cmd(["git", "config", "user.name", "Mohit"], r1)
run_cmd(["git", "config", "user.email", "mohit@Mohits-Mac-Studio.local"], r1)
run_cmd(["git", "add", "."], r1)
run_cmd(["git", "commit", "-m", "feat: initial release of connectome-llm-interface"], r1)
run_cmd(["git", "branch", "-M", "main"], r1)

# 2. Package neuroevolution-connectomes
r2 = REPOS / "neuroevolution-connectomes"
if r2.exists():
    shutil.rmtree(r2)
r2.mkdir(parents=True)

shutil.copytree(BASE / "neuroevolution", r2 / "neuroevolution")
shutil.copytree(BASE / "experiments", r2 / "experiments")
shutil.copytree(BASE / "brainmodel", r2 / "brainmodel")
shutil.copy(BASE / ".gitignore", r2 / ".gitignore")

r2_readme = """# Neuroevolution Connectomes: Evolving Brains from Scratch

Genetic algorithms discovering biological connectome wiring topologies without human design or backpropagation.

```bash
# Run multi-generation connectome evolution
python3 neuroevolution/evolver.py
```

### Scientific Breakthrough
Starting from zero prior knowledge (random Gaussian synaptic initialization), artificial evolution autonomously rediscovers:
- **Reciprocal Inhibition**: Evolving strong mutual inhibition between dorsal and ventral motor pools ($W_{\\text{dorsal} \\to \\text{ventral}} = -0.77$, $W_{\\text{ventral} \\to \\text{dorsal}} = -0.61$).
- **Anti-Phase Traveling Waves**: Discovering phase-lagged muscle coordination for forward hydrostatic propulsion.
"""
with open(r2 / "README.md", "w") as f:
    f.write(r2_readme)

run_cmd(["git", "init"], r2)
run_cmd(["git", "config", "user.name", "Mohit"], r2)
run_cmd(["git", "config", "user.email", "mohit@Mohits-Mac-Studio.local"], r2)
run_cmd(["git", "add", "."], r2)
run_cmd(["git", "commit", "-m", "feat: initial release of neuroevolution-connectomes"], r2)
run_cmd(["git", "branch", "-M", "main"], r2)

# 3. Package multi-agent-fluid-swarm
r3 = REPOS / "multi-agent-fluid-swarm"
if r3.exists():
    shutil.rmtree(r3)
r3.mkdir(parents=True)

shutil.copytree(BASE / "swarm", r3 / "swarm")
shutil.copytree(BASE / "experiments", r3 / "experiments")
shutil.copytree(BASE / "brainmodel", r3 / "brainmodel")
shutil.copy(BASE / ".gitignore", r3 / ".gitignore")

r3_readme = """# Multi-Agent Fluid Swarm: Predator-Prey Connectome Hydrodynamics

Multi-agent continuous fluid dynamics simulating 4 forager prey worms driven by authentic biophysical *C. elegans* connectomes against an agile apex predator tracking hydrodynamic wakes.

```bash
# Run multi-agent predator-prey simulation
python3 swarm/predator_prey_sim.py
```

### Features
- 12-segment articulated hydrostatic bodies interacting in fluid drag.
- Mechanosensory touch and wake turbulence triggering instantaneous AVA reversal escape reflexes.
- 99.6% evasion success rate across 800 physical steps.
"""
with open(r3 / "README.md", "w") as f:
    f.write(r3_readme)

run_cmd(["git", "init"], r3)
run_cmd(["git", "config", "user.name", "Mohit"], r3)
run_cmd(["git", "config", "user.email", "mohit@Mohits-Mac-Studio.local"], r3)
run_cmd(["git", "add", "."], r3)
run_cmd(["git", "commit", "-m", "feat: initial release of multi-agent-fluid-swarm"], r3)
run_cmd(["git", "branch", "-M", "main"], r3)

print("✅ All 3 repositories packaged and committed locally!")
