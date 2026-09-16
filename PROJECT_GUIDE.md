# 🧭 Brain Math: Project & Directory Guide

Welcome! This guide is designed to make this codebase clear, human-readable, and easy to explore.

Everything in this folder is a **computational simulation** exploring different ideas from biology, neuroscience, and embodied AI.

---

## 🌟 The Easiest Way to Explore: The Web Visualizer

If you just want to see everything running interactively without touching the command line:

```bash
open visualizer/neuroai_dashboard.html
```

This opens an offline, self-contained dashboard in your browser with **13 interactive tabs**:
1. **Embodied Organism**: A simulated creature foraging in a 2D fluid arena.
2. **Modular Pendulum**: Balances an inverted cart-pole under sudden wind gusts.
3. **C. elegans Connectome**: A 302-neuron biological circuit producing crawling waves.
4. **Synthesis**: Combines crawling with scent-seeking navigation.
5. **Language Connectome**: Type words like 'sprint' or 'evade' to guide the worm.
6. **Neuroevolution**: Shows how random neural connections evolve into alternating rhythms.
7. **Predator-Prey Swarm**: Four forager worms evading a larger predator.
8. **Active Inference**: Compares exploration paths in a maze with dead ends.
9. **Conditioning**: Ring a tone or deliver a shock to see associative learning in action.
10. **Fly Optomotor**: A fruit fly eye stabilizing against wind turbulence.
11. **Mirror Self-Recognition**: A creature using sensory predictions to recognize its reflection.
12. **Theory of Mind**: The Sally-Anne scenario demonstrating perspective-taking.
13. **Proto-Grammar**: Two simulated agents coordinating via sound signals.

---

## 📁 Directory Map

Here is what each folder in this project contains:

| Folder | What It Is | Key File to Run |
|---|---|---|
| `visualizer/` | The interactive browser dashboard | `open visualizer/neuroai_dashboard.html` |
| `brainmodel/` | Core Python classes for biological brain regions | (Imported as a library) |
| `experiments/` | Foundational embodiment & connectome benchmarks | `python3 experiments/run_all_experiments.py` |
| `active_inference/` | Free Energy Principle vs standard navigation | `python3 active_inference/benchmark_fep_vs_rl.py` |
| `connectome_llm/` | Steering the worm using English text commands | `python3 connectome_llm/connectome_llm_sim.py` |
| `drosophila_fly/` | Fruit fly visual tracking and flight saccades | `python3 drosophila_fly/fly_optomotor_sim.py` |
| `emergent_communication/`| Cooperative acoustic signaling between two agents | `python3 emergent_communication/proto_language_sim.py` |
| `hebbian_plasticity/` | Pavlovian classical conditioning (sound + shock) | `python3 hebbian_plasticity/pavlovian_conditioning.py` |
| `mirror_self_recognition/`| Efference copy cancelling self-movement in a mirror | `python3 mirror_self_recognition/corollary_discharge.py` |
| `neuroevolution/` | Evolving neural circuits from scratch | `python3 neuroevolution/evolve_connectome.py` |
| `swarm/` | Predator and prey swimming in a shared fluid | `python3 swarm/swarm_predator_prey_sim.py` |
| `theory_of_mind/` | The Sally-Anne false-belief perspective task | `python3 theory_of_mind/sally_anne_experiment.py` |

---

## 🗄️ Supporting Folders

- `repos/`: Standalone packaged copies of projects that have been published to GitHub.
- `scripts/`: Helper scripts used for maintenance and data exports.
- `critic_reports/` & `reviews/`: Archived notes and internal logs from earlier runs.

---

## 💡 Quick Tips
- **Every subfolder now has its own `README.md`**: If you navigate into any subfolder, there is a short, friendly explanation of what the files inside do.
- **Low Resource Usage**: All scripts run using pure standard Python and NumPy, taking only ~50 MB of memory and leaving plenty of headroom on your Mac.
