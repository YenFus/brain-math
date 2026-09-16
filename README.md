# brainmodel — Bio-Mathematical Brain & Embodied Connectome Suite

A multi-scale, dependency-light (pure NumPy + Apple Silicon Metal) computational neuroscience framework, autonomous self-improving cognitive agent, and physical connectome embodiment suite.

Every brain structure is a class holding its governing mathematical equations; they are wired into a coherent biological `Brain` running end-to-end; every neurological disorder is a parameter perturbation of the healthy model; and the system features an autonomous embodied loop with an interactive 60 FPS visual dashboard.

```bash
# 1. Run the foundational whole-brain & pathology validation (12 disorders)
python3 demo.py

# 2. Run the lifelong learning agent with Recursive Self-Improvement (RSI)
python3 demo_agent_rsi.py

# 3. Run the objective consciousness evaluation suite
python3 test_consciousness.py

# 4. Run the physical embodiment & connectome benchmark suite
python3 experiments/run_all_experiments.py

# 5. Open the interactive visual dashboard in any browser
open visualizer/neuroai_dashboard.html
```

---

## 🔬 Experimental Suites & Benchmarks

### 1. Embodied Closed-Loop Organism in 2D Fluid Arena (`experiments/exp1_embodied_organism.py`)
- Continuous 2D fluid hydrodynamics with viscous drag, boundary walls, obstacles, and diffusing chemical scent fields.
- Whole-brain closed-loop integration:
  - **Chemotaxis & Whiskers**: Bilateral scent antennae and 3 proximity raycast whiskers.
  - **Hypothalamus**: Glucose tracker where starvation amplifies foraging drive.
  - **Basal Ganglia**: Dopamine RPE selects locomotive actions (straight, gradient steer, avoidance).
  - **Cerebellum**: Real-time forward model cancels fluid drag and eliminates intention tremor.

### 2. Doya's Bio-Modular Triad vs Monolithic RL (`experiments/exp2_modular_vs_rl.py`)
- Empirical benchmark of Kenji Doya's Triad architecture against standard Actor-Critic TD Reinforcement Learning on continuous Inverted Pendulum / Cart-Pole balance.
- Subjected to three violent perturbation regimes:
  - **Step 300**: Sudden lateral wind shock (+18 N impulse).
  - **Step 600**: Sudden physical parameter drift (pole mass increased 250%).
  - **Step 800**: Stochastic aerodynamic turbulence storm.
- **Key Results**:
  | Controller | RMSE Angle Error | Max Overshoot | Drop Failures | Recovery Time |
  |---|---|---|---|---|
  | **Monolithic RL** | 3.24° | 12.87° | 1 | 22 steps |
  | **Bio-Modular Triad** | **2.76°** | 16.31° | **0** | 25 steps |
  | **Cerebellar Ataxia (Damaged)** | 2.94° | 18.38° | **0** | 25 steps |

### 3. *C. elegans* 302-Neuron Connectome & Escape Reflex (`experiments/exp3_celegans_connectome.py`)
- Non-spiking graded potential biophysics (White et al. 1986 / Wicks et al. 1996 / OpenWorm).
- Sensory mechanosensory neurons (`ALM` head touch, `PLM` tail touch) driving command interneurons:
  - **Forward crawling**: `AVB` + `PVC` drive B-class motor neurons (`DB`, `VB`).
  - **Reversal escape**: `AVA` + `AVD` drive A-class motor neurons (`DA`, `VA`).
  - **GABAergic cross-inhibition**: D-class neurons (`DD`, `VD`) enforce dorsal-ventral muscular phase opposition.
- 12-segment articulated hydrostatic body with stretch-receptor proprioceptive wave propagation.
- **Reflex Latency**: ~54.0 ms from head tap to reverse wave propagation.

### 4. Synthesis Hybrid Organism (`experiments/synthesis_hybrid_organism.py`)
- Combines the 12-segment *C. elegans* connectome with cerebellar fluid drag compensation and prefrontal/OFC value seeking in a continuous foraging arena.

---

## 🖥️ Interactive HTML5 Visualizer Dashboard

Open `visualizer/neuroai_dashboard.html` directly in any web browser. Zero server setup or external dependencies required.

Features:
- **Tab 1**: Live 2D Fluid Arena with scent gradients, food pellets, obstacles, and dopamine/glucose telemetry.
- **Tab 2**: Side-by-side Inverted Pendulum comparison (Monolithic RL vs Bio-Modular Triad) with interactive shock triggers.
- **Tab 3**: Nematode connectome oscilloscope showing membrane potentials ($V_m$) and live muscle torque waves with touch tap triggers.
- **Tab 4**: Synthesis hybrid foraging organism.

---

## 🧠 Brain Architecture & Neural Atlas

| File | Region / layer | Equations / computation |
|---|---|---|
| `brainmodel/neuron.py` | neuron | Leaky integrate-and-fire; mean-field rate population |
| `brainmodel/synapse.py` | synapse | NMDA Mg²⁺-block LTP/LTD; early→late LTP (CREB protein synthesis); excitotoxicity |
| `brainmodel/neuromodulators.py` | neurochemistry | Coupled Lotka-Volterra dynamics (DA, 5-HT, NE, ACh, Glutamate, GABA); E/I balance |
| `brainmodel/emotion.py` | affective neuroscience | Lövheim's Cube of Emotion; Russell's Circumplex (PAD); Damasio Somatic Markers; Free Energy velocity ($-\frac{d\mathcal{F}}{dt}$) |
| `brainmodel/hippocampus.py` | episodic memory | Modern Continuous Hopfield Networks ($C \sim 2^{d/2}$ capacity); DG pattern separation; sharp-wave ripple sleep replay |
| `brainmodel/brainstem.py` | brainstem | Raphe/VTA/LC/PPT source nuclei → tonic 5-HT/DA/NE/ACh; autonomic heart rate & respiration |
| `brainmodel/thalamus.py` | thalamus | LGN/MGN/VPL sensory relay; arousal + attention gating; reality-monitoring |
| `brainmodel/hypothalamus.py` | hypothalamus + pineal | Two-process sleep (adenosine + SCN circadian clock), HPA stress axis (CRF→cortisol) |
| `brainmodel/cerebellum.py` | cerebellum | Forward model; parallel-fiber→Purkinje LTD supervised learning; tremor cancellation |
| `brainmodel/cortex.py` | cerebral cortex | Sensory cortices; OFC subjective value; DLPFC working memory; ACC conflict monitoring |
| `brainmodel/basal_ganglia.py` | basal ganglia | Full nuclei (D1/D2 striatum, GPe, STN, GPi/SNr), direct/indirect/hyperdirect pathways |
| `brainmodel/dual_process.py` | decision engine | System 1 (fast basal ganglia gating) vs System 2 (prefrontal MCTS deliberation) |
| `brainmodel/hardware_governor.py`| system supervisor | Real-time macOS Mach kernel memory monitor guaranteeing $\ge 10\text{ GB}$ headroom |
| `brainmodel/rsi/` | recursive self-improvement | Gödel-style self-referential AST code modifier with atomic verification & rollback |

---

## 🛡️ Apple Silicon Memory Safety & Headroom Guarantee

The entire system is optimized for Apple Silicon Macs:
- Native process memory footprint: **$\sim 48\text{ MB}$** resident memory.
- Hardware Governor strictly enforces $\ge 10.0\text{ GB}$ free RAM headroom on the host system.
- Zero risk of memory exhaustion, system freezing, or unmonitored allocation spikes.
