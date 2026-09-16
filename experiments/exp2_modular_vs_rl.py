"""
exp2_modular_vs_rl.py - Experiment 2: Bio-Modular Triad (Doya's Architecture) vs Monolithic RL
on Continuous Inverted Pendulum Stabilization under Sudden Shocks.

Kenji Doya's Triad Principle (1999/2000):
  - Basal Ganglia: Reinforcement Learning (Dopamine RPE) for macro goal-directed policy.
  - Cerebellum: Supervised Learning (Climbing Fiber error) for rapid forward/inverse dynamic compensation.
  - Locus Coeruleus: Norepinephrine (NE) tracking unexpected uncertainty & shock amplification.

Test Environment:
  Continuous Cart-Pole / Inverted Pendulum:
    s = [x, x_dot, theta, theta_dot] (theta=0 is upright)
  Perturbations:
    - Step 300: Sudden violent lateral wind impulse shock (+18 N)
    - Step 600: Physical parameter drift (Pole mass doubled: m -> 2.5m)
    - Step 800: Stochastic turbulence storm (random wind gusts)
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from brainmodel.cerebellum import Cerebellum
from brainmodel.basal_ganglia import BasalGanglia
from brainmodel.neuromodulators import Neuromodulators
from brainmodel.hardware_governor import HardwareGovernor


class CartPolePhysics:
    """Nonlinear Cart-Pole / Inverted Pendulum dynamic equations."""

    def __init__(self, cart_mass: float = 1.0, pole_mass: float = 0.1, length: float = 0.5, dt: float = 0.02):
        self.M = cart_mass
        self.m = pole_mass
        self.l = length
        self.g = 9.81
        self.dt = dt

        # State: [x, x_dot, theta, theta_dot]
        self.state = np.zeros(4, dtype=np.float64)
        self.external_force = 0.0

    def reset(self, initial_theta: float = 0.05):
        self.state = np.array([0.0, 0.0, initial_theta, 0.0], dtype=np.float64)
        self.external_force = 0.0
        return self.state.copy()

    def step(self, force: float) -> Tuple[np.ndarray, float, bool]:
        """Physics step using 4th-order Runge-Kutta / Euler-Cromer integration with external perturbations."""
        total_force = np.clip(force, -35.0, 35.0) + self.external_force

        x, x_dot, theta, theta_dot = self.state

        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        temp = (total_force + self.m * self.l * (theta_dot ** 2) * sintheta) / (self.M + self.m)
        theta_acc = (self.g * sintheta - costheta * temp) / (self.l * (4.0 / 3.0 - self.m * (costheta ** 2) / (self.M + self.m)))
        x_acc = temp - self.m * self.l * theta_acc * costheta / (self.M + self.m)

        # Semi-implicit Euler integration
        x_dot += x_acc * self.dt
        x += x_dot * self.dt
        theta_dot += theta_acc * self.dt
        theta += theta_dot * self.dt

        # Normalize theta to [-pi, pi]
        theta = (theta + math.pi) % (2 * math.pi) - math.pi

        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float64)

        # Failure condition: angle > 60 degrees (1.047 rad) or cart off track (|x| > 4.0)
        failed = abs(theta) > 1.05 or abs(x) > 4.0

        # Reward: 1.0 for upright, penalized by angle and velocity
        reward = math.cos(theta) - 0.1 * (theta ** 2) - 0.01 * (x ** 2) - 0.005 * (force ** 2)
        if failed:
            reward -= 50.0

        return self.state.copy(), reward, failed


class MonolithicRLController:
    """Standard Reinforcement Learning Controller (Actor-Critic TD learning without forward models)."""

    def __init__(self, state_dim: int = 4, lr_actor: float = 0.02, lr_critic: float = 0.05, gamma: float = 0.95):
        # Initial policy weights capable of nominal balance
        self.w_actor = np.array([1.5, 2.0, 35.0, 8.0])
        self.w_critic = np.zeros(state_dim)
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = gamma
        self.prev_state = None
        self.prev_action = 0.0

    def act(self, state: np.ndarray, learn: bool = True) -> float:
        action = float(np.dot(self.w_actor, state))
        self.prev_state = state.copy()
        self.prev_action = action
        return float(np.clip(action, -35.0, 35.0))

    def update(self, next_state: np.ndarray, reward: float, done: bool):
        if self.prev_state is None:
            return
        v_current = float(np.dot(self.w_critic, self.prev_state))
        v_next = 0.0 if done else float(np.dot(self.w_critic, next_state))
        td_error = reward + self.gamma * v_next - v_current

        # Delayed TD error updates
        self.w_critic += self.lr_critic * td_error * self.prev_state
        self.w_actor += self.lr_actor * td_error * self.prev_state


class BioModularTriadController:
    """Doya's Bio-Modular Triad: Basal Ganglia RL + Cerebellar Fast Forward Model + Locus Coeruleus NE."""

    def __init__(self, state_dim: int = 4, damage_cerebellum: float = 0.0):
        # 1. Basal Ganglia for goal setpoint policy
        self.bg_weights = np.array([1.2, 1.8, 30.0, 6.0])
        self.bg_lr = 0.01

        # 2. Cerebellum for fast inverse dynamic compensation and jitter damping
        tremor = 0.6 if damage_cerebellum > 0.0 else 0.0
        self.cerebellum = Cerebellum(n_inputs=16, lr=0.08, damage=damage_cerebellum, tremor_gain=tremor)

        # 3. Neuromodulators for NE surge during sudden shocks
        self.nm = Neuromodulators()

        self.prev_state = None
        self.prev_force = 0.0
        self.expected_theta_acc = 0.0

    def act(self, state: np.ndarray) -> Tuple[float, float, float]:
        """Computes motor command composed of:
           u_total = u_BasalGanglia + Cerebellar_correction * (1 + NE_gain)
        """
        x, x_dot, theta, theta_dot = state

        # 1. Basal Ganglia setpoint command
        u_bg = float(np.dot(self.bg_weights, state))

        # 2. Dynamic state prediction error (acceleration residual)
        actual_acc = theta_dot - (self.prev_state[3] if self.prev_state is not None else 0.0)
        prediction_error = abs(actual_acc - self.expected_theta_acc)

        # 3. Locus Coeruleus Norepinephrine phasic burst on unexpected dynamic shock
        ne_burst = float(np.clip(prediction_error * 3.5, 0.0, 1.5))
        self.nm.norepinephrine.phasic = ne_burst
        ne_level = self.nm.norepinephrine.effective

        # 4. Cerebellar inverse compensation for angular velocity and inertia
        desired_correction = -theta_dot * (2.2 + ne_level * 0.8)
        u_refined, cer_error = self.cerebellum.refine(
            command=u_bg,
            desired_correction=desired_correction,
            learn=True
        )

        total_force = float(np.clip(u_refined, -35.0, 35.0))
        self.prev_state = state.copy()
        self.prev_force = total_force
        self.expected_theta_acc = actual_acc * 0.7

        return total_force, float(cer_error), float(ne_level)


class Experiment2Runner:
    """Runs comparative evaluation between Monolithic RL and Bio-Modular Triad."""

    def __init__(self, steps: int = 1000):
        self.steps = steps

    def run_trial(self, controller_type: str) -> Dict:
        env = CartPolePhysics(cart_mass=1.0, pole_mass=0.1, length=0.5, dt=0.02)
        state = env.reset(initial_theta=0.04)

        if controller_type == "monolithic_rl":
            agent = MonolithicRLController()
        elif controller_type == "bio_modular":
            agent = BioModularTriadController(damage_cerebellum=0.0)
        elif controller_type == "cerebellar_ataxia":
            agent = BioModularTriadController(damage_cerebellum=0.7)
        else:
            raise ValueError(f"Unknown controller: {controller_type}")

        telemetry = []
        failures = 0
        total_squared_angle_error = 0.0
        max_overshoot = 0.0
        recovery_time_steps = 0
        shock_active = False

        for step in range(self.steps):
            # Apply Perturbation Schedule
            # Shock 1: Violent wind gust at step 300 for 6 steps (+18.0 N)
            if 300 <= step < 306:
                env.external_force = 18.0
                shock_active = True
            # Mass doubling at step 600: Pole mass 0.1 -> 0.25 (250% increase)
            elif step == 600:
                env.m = 0.25
                env.external_force = 0.0
            # Turbulence storm at step 800-900: Random fluctuating wind
            elif 800 <= step < 900:
                env.external_force = float(np.sin(step * 0.4) * 8.0 + np.random.normal(0, 3.0))
            else:
                env.external_force = 0.0

            # Step agent
            ne_val = 0.0
            cer_err = 0.0
            if controller_type == "monolithic_rl":
                force = agent.act(state)
            else:
                force, cer_err, ne_val = agent.act(state)

            # Step physics
            next_state, reward, failed = env.step(force)

            if controller_type == "monolithic_rl":
                agent.update(next_state, reward, failed)

            # Track metrics
            theta = state[2]
            theta_deg = math.degrees(theta)
            total_squared_angle_error += (theta ** 2)
            max_overshoot = max(max_overshoot, abs(theta_deg))

            if shock_active:
                if abs(theta_deg) > 4.0:
                    recovery_time_steps += 1
                elif step > 310:
                    shock_active = False

            if failed:
                failures += 1
                state = env.reset(initial_theta=0.02)
            else:
                state = next_state

            if step % 2 == 0:
                telemetry.append({
                    "step": step,
                    "x": round(float(state[0]), 3),
                    "theta_deg": round(theta_deg, 2),
                    "force": round(force, 2),
                    "wind_force": round(env.external_force, 2),
                    "pole_mass": round(env.m, 2),
                    "ne_level": round(ne_val, 2),
                    "cerebellar_err": round(cer_err, 3),
                    "failed": bool(failed),
                })

        mean_rmse_deg = math.degrees(math.sqrt(total_squared_angle_error / self.steps))

        return {
            "controller": controller_type,
            "mean_rmse_deg": round(mean_rmse_deg, 3),
            "max_overshoot_deg": round(max_overshoot, 2),
            "drop_failures": failures,
            "shock_recovery_steps": recovery_time_steps,
            "telemetry": telemetry,
        }

    def run(self) -> Dict:
        print(f"\n⚖️  [Experiment 2] Launching Bio-Modular vs Monolithic RL Comparative Benchmark ({self.steps} steps)...")

        controllers = ["monolithic_rl", "bio_modular", "cerebellar_ataxia"]
        results = {}

        for c_type in controllers:
            print(f"   • Running controller: {c_type}...")
            res = self.run_trial(c_type)
            results[c_type] = res
            print(f"     -> RMSE: {res['mean_rmse_deg']}° | Max Overshoot: {res['max_overshoot_deg']}° | Drops: {res['drop_failures']} | Recovery: {res['shock_recovery_steps']} steps")

        # Save to experiments/data/exp2_results.json
        data_dir = Path(__file__).resolve().parent / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        with open(data_dir / "exp2_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        print("\n✅ [Experiment 2 Complete] Benchmark comparison saved to experiments/data/exp2_results.json")
        return results


if __name__ == "__main__":
    exp2 = Experiment2Runner(steps=1000)
    exp2.run()
