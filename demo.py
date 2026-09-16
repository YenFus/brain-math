"""
demo.py - exercise the whole integrated brain.

Sections:
  1. Whole-brain task: healthy vs every disorder on a 3-armed bandit, now also
     reporting memory, motor coordination, autonomic and affective signatures.
  2. Circadian sleep-wake: the hypothalamus + brainstem clock over a simulated day.
  3. Drift Diffusion Model: speed-accuracy trade-off.
  4. Heuristic-Systematic Model: when deliberate processing engages.
  5. Integrated information (toy Phi).

    python3 demo.py
"""

from __future__ import annotations

import warnings
import numpy as np

from brainmodel.brain import Brain
from brainmodel import pathology
from brainmodel.consciousness import integrated_information
from brainmodel.decision import DriftDiffusion
from brainmodel.dual_process import HeuristicSystematic
from brainmodel.hypothalamus import Hypothalamus
from brainmodel.brainstem import Brainstem

warnings.filterwarnings("ignore")  # silence spurious macOS Accelerate BLAS matmul warning

TRUE_REWARDS = np.array([0.1, 1.0, 0.2])   # arm #1 is the good one


def world_fn(i, t):
    """3-armed bandit during daytime/awake: periodic salient cue, sporadic threat."""
    salient = 1.0 if (i % 20) < 6 else 0.05
    threat = 0.8 if (i % 137) < 5 else 0.05
    return {
        "sensory": {"visual": salient},
        "saliency": salient,
        "external": 0.6,
        "threat": threat,
        "true_rewards": TRUE_REWARDS,
        "arousal_drive": 0.95,   # keep awake for the task
        "light": 1.0,
    }


def run_condition(name, perturb, steps=1500, seed=0):
    brain = Brain(n_actions=3, rng=np.random.default_rng(seed))
    perturb(brain)
    log = brain.run(steps, world_fn)
    s = brain.summary(log)
    s["learned_correct_arm"] = int(np.argmax(s["final_Q"])) == int(np.argmax(TRUE_REWARDS))
    return s


def section_whole_brain():
    print("=" * 96)
    print("1. WHOLE-BRAIN TASK  -  3-armed bandit (good arm = #1). Each disorder is")
    print("   ONLY a parameter perturbation of the same brain; all signatures emerge.")
    print("=" * 96)
    header = (f"{'condition':<24}{'act':>5}{'rew':>6}{'chan':>5}{'mem':>6}"
              f"{'motor':>7}{'phant':>7}{'aggr*':>7}{'health':>7}{'HR':>5}{'arm?':>6}")
    print(header); print("-" * len(header))
    for name, fn in pathology.REGISTRY.items():
        s = run_condition(name, fn)
        motor = s["final_motor_error"]
        motor_s = f"{motor:>7.3f}" if motor == motor else f"{'  --':>7}"   # nan check
        print(f"{name:<24}{s['actions_taken']:>5}{s['total_reward']:>6.0f}"
              f"{s['mean_channels_open']:>5.1f}{s['memory_score']:>6.2f}{motor_s}"
              f"{s['mean_phantom_leak']:>7.3f}{s['peak_aggression']:>7.2f}"
              f"{s['final_tissue_health']:>7.2f}{s['mean_heart_rate']:>5.0f}"
              f"{'  yes' if s['learned_correct_arm'] else '   no':>6}")
    print("-" * len(header))
    print("act=actions executed  rew=total reward  chan=motor channels (0 akinesia, ~1 clean,")
    print(">1 overflow)  mem=hippocampal recall (0.5 chance)  motor=cerebellar error (-- if no")
    print("actions)  phant=phantom percept leak  aggr*=peak aggression  health=surviving tissue")
    print("HR=mean heart rate (bpm)  arm?=learned the good arm")


def section_circadian():
    print("\n" + "=" * 96)
    print("2. CIRCADIAN SLEEP-WAKE  -  hypothalamic clock + brainstem over a simulated day")
    print("=" * 96)
    hyp = Hypothalamus(clock_hours=7.0)
    bs = Brainstem()
    dt = 60000.0  # 1-minute steps
    print(f"  {'time':>6}{'light':>7}{'melatonin':>11}{'arousal':>9}{'state':>9}{'HR':>6}")
    marks = {6, 9, 13, 18, 22, 1, 3}
    seen = set()
    for minute in range(34 * 60):
        hr = (hyp.clock_hours) % 24
        light = 1.0 if 6.5 < hr < 20.5 else 0.0
        o = hyp.step(dt, {"light": light})
        bo = bs.step(dt, {"arousal_drive": o["arousal_drive"], "crf": o["crf"], "rem": o["rem"]})
        hyp.asleep = bo["arousal"] < 0.35
        h_int = int(hr)
        if h_int in marks and h_int not in seen and abs(hr - h_int) < 0.02:
            seen.add(h_int)
            state = "ASLEEP" if hyp.asleep else "awake"
            print(f"  {h_int:>4}h {light:>6.0f}{o['melatonin']:>11.2f}"
                  f"{bo['arousal']:>9.2f}{state:>9}{bo['heart_rate']:>6.0f}")
    print("  (melatonin peaks in biological night; arousal collapses -> sleep)")


def section_ddm():
    print("\n" + "=" * 96)
    print("3. DRIFT DIFFUSION MODEL  -  reaction time vs evidence quality / caution")
    print("=" * 96)
    for v in (0.05, 0.2, 0.4):
        for a in (0.8, 1.6):
            rts, correct = [], 0
            for k in range(300):
                ddm = DriftDiffusion(v=v, a=a, z=0.5, t0=120, rng=np.random.default_rng(k))
                choice, rt = ddm.simulate_choice()
                if rt is not None:
                    rts.append(rt); correct += (choice == 1)
            print(f"  drift v={v:<4} threshold a={a:<4} mean RT={np.mean(rts):6.0f}ms "
                  f"accuracy={correct/300:5.1%}")
    print("  (higher drift -> faster & accurate; higher threshold -> slower but accurate)")


def section_hsm():
    print("\n" + "=" * 96)
    print("4. HEURISTIC-SYSTEMATIC MODEL  -  when does effortful processing engage?")
    print("=" * 96)
    hsm = HeuristicSystematic()
    cases = [
        ("credible source, low stakes", 0.9, -0.8, 0.0),
        ("weak source, low stakes",     0.4, -0.8, 0.0),
        ("weak source, HIGH stakes",    0.4, -0.8, 1.0),
        ("credible source, HIGH stakes",0.85, -0.8, 1.0),
    ]
    print(f"  {'scenario':<32}{'mode':>11}{'judgment':>10}{'confidence':>12}")
    for label, cue, evid, stakes in cases:
        r = hsm.judge(cue, evid, stakes=stakes)
        print(f"  {label:<32}{r['mode']:>11}{r['judgment']:>10}{r['confidence']:>12.2f}")
    print("  (sufficiency principle: heuristic alone unless a confidence gap remains)")


def section_phi():
    print("\n" + "=" * 96)
    print("5. INTEGRATED INFORMATION (toy Phi)  -  integrated vs disconnected system")
    print("=" * 96)
    n = 3
    states = [tuple((i >> b) & 1 for b in range(n)) for i in range(2 ** n)]
    T_int = np.array([[0.9 if sum(s) >= 2 else 0.1 for _ in range(n)] for s in states])
    T_ind = np.array([[0.5 for _ in range(n)] for _ in states])
    print(f"  Phi (integrated, units vote on the whole) = {integrated_information(T_int):.4f}")
    print(f"  Phi (independent, units ignore each other) = {integrated_information(T_ind):.4f}")
    print("  (IIT: more irreducible cause-effect structure -> more consciousness)")


def main():
    section_whole_brain()
    section_circadian()
    section_ddm()
    section_hsm()
    section_phi()


if __name__ == "__main__":
    main()
