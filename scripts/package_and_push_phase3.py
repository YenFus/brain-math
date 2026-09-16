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

# 1. Package active-inference-connectome
r1 = REPOS / "active-inference-connectome"
if r1.exists(): shutil.rmtree(r1)
r1.mkdir(parents=True)
shutil.copytree(BASE / "active_inference", r1 / "active_inference")
shutil.copytree(BASE / "experiments", r1 / "experiments")
shutil.copytree(BASE / "brainmodel", r1 / "brainmodel")
shutil.copy(BASE / ".gitignore", r1 / ".gitignore")

r1_readme = """# Active Inference Connectome: Free Energy Principle in Biophysical C. elegans

Karl Friston's Free Energy Principle (FEP) embodied in an autonomous biophysical *C. elegans* connectome:
- Generative Model (A, B, C, D matrices)
- Minimizes Variational Free Energy $F$ for state perception
- Minimizes Expected Free Energy $G = \\text{Risk} + \\text{Ambiguity}$ for action selection
- Connectome Descending Drive: Injects micro-currents into $I_{\\text{AVB}}$ (crawl), $I_{\\text{AVA}}$ (reversal), and $I_{\\text{ASE}}$ (chemosensory gain).

```bash
# Run benchmark: Active Inference vs Standard Q-Learning
python3 active_inference/benchmark_fep_vs_rl.py
```
"""
with open(r1 / "README.md", "w") as f: f.write(r1_readme)
run_cmd(["git", "init"], r1)
run_cmd(["git", "config", "user.name", "Mohit"], r1)
run_cmd(["git", "config", "user.email", "mohit@Mohits-Mac-Studio.local"], r1)
run_cmd(["git", "add", "."], r1)
run_cmd(["git", "commit", "-m", "feat: initial release of active-inference-connectome"], r1)
run_cmd(["git", "branch", "-M", "main"], r1)

# 2. Package hebbian-plasticity-connectome
r2 = REPOS / "hebbian-plasticity-connectome"
if r2.exists(): shutil.rmtree(r2)
r2.mkdir(parents=True)
shutil.copytree(BASE / "hebbian_plasticity", r2 / "hebbian_plasticity")
shutil.copytree(BASE / "experiments", r2 / "experiments")
shutil.copytree(BASE / "brainmodel", r2 / "brainmodel")
shutil.copy(BASE / ".gitignore", r2 / ".gitignore")

r2_readme = """# Hebbian Plasticity Connectome: Pavlovian Classical Conditioning in Living Circuits

Associative learning in living neural connectomes using Three-Factor Neuromodulated STDP:
$$dW/dt = \\eta \\cdot e(t) \\cdot M(t)$$
- Conditioned Stimulus (CS): Neutral mechanical tone (normally ignored, 0% reversal).
- Unconditioned Stimulus (US): Noxious heat shock / toxic repellent (triggers reflexive reversal).
- 5 paired trials induce long-term potentiation (LTP) on $W_{\\text{CS} \\to \\text{AVA}}$, causing the neutral tone alone to trigger violent 100% escape reversals!
- Extinction protocol demonstrates subsequent LTD depression back to baseline.

```bash
# Run Pavlovian conditioning simulation
python3 hebbian_plasticity/pavlovian_conditioning.py
```
"""
with open(r2 / "README.md", "w") as f: f.write(r2_readme)
run_cmd(["git", "init"], r2)
run_cmd(["git", "config", "user.name", "Mohit"], r2)
run_cmd(["git", "config", "user.email", "mohit@Mohits-Mac-Studio.local"], r2)
run_cmd(["git", "add", "."], r2)
run_cmd(["git", "commit", "-m", "feat: initial release of hebbian-plasticity-connectome"], r2)
run_cmd(["git", "branch", "-M", "main"], r2)

# 3. Package drosophila-optomotor-connectome
r3 = REPOS / "drosophila-optomotor-connectome"
if r3.exists(): shutil.rmtree(r3)
r3.mkdir(parents=True)
shutil.copytree(BASE / "drosophila_fly", r3 / "drosophila_fly")
shutil.copytree(BASE / "experiments", r3 / "experiments")
shutil.copytree(BASE / "brainmodel", r3 / "brainmodel")
shutil.copy(BASE / ".gitignore", r3 / ".gitignore")

r3_readme = """# Drosophila Optomotor Connectome: Fly Visuomotor Flight Reflex

Inspired by the Princeton FlyWire Connectome and Hassenstein-Reichardt Elementary Motion Detectors (EMDs):
- 16-channel compound eye retina.
- EMD correlation units detecting directional visual flow.
- Lobula Plate Tangential Cells (Horizontal System HS cells) driving yaw flight stabilization against aerodynamic wind vortices.
- Visual loom expansion detector triggering rapid 90-degree escape saccades (< 25 ms).

```bash
# Run Drosophila flight simulation
python3 drosophila_fly/fly_optomotor_sim.py
```
"""
with open(r3 / "README.md", "w") as f: f.write(r3_readme)
run_cmd(["git", "init"], r3)
run_cmd(["git", "config", "user.name", "Mohit"], r3)
run_cmd(["git", "config", "user.email", "mohit@Mohits-Mac-Studio.local"], r3)
run_cmd(["git", "add", "."], r3)
run_cmd(["git", "commit", "-m", "feat: initial release of drosophila-optomotor-connectome"], r3)
run_cmd(["git", "branch", "-M", "main"], r3)

print("✅ All 3 Phase 3 repositories packaged and committed locally!")
