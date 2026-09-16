"""
hypothalamus.py - homeostasis, circadian rhythm, endocrine axes, and sleep consolidation.

Biological Functions:
  1. Circadian Clock (Process C): SCN phase oscillator driving a 24h diurnal rhythm
     and pineal melatonin secretion.
  2. Two-Process Sleep Regulation (Process S): Sleep pressure (adenosine) accumulates
     with cognitive wake activity and hardware compute load; discharges during sleep.
     When sleep pressure exceeds critical threshold, offline dreaming & consolidation is triggered.
  3. HPA Stress Axis: Stressor -> CRF -> ACTH -> Cortisol -> negative feedback loop.
  4. Metabolic & Hardware Stress Watchdog: High hardware load (low memory headroom)
     generates homeostatic strain, accelerating sleep pressure to preserve system integrity.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class Hypothalamus:
    # Circadian
    clock_hours: float = 8.0          # current biological time of day (0..24)
    sleep_pressure: float = 0.0       # Process S (adenosine); rises awake, discharges asleep
    asleep: bool = False
    sleep_threshold: float = 1.2      # threshold where sleep/consolidation becomes urgent

    # Homeostatic setpoints
    glucose: float = 1.0              # 1.0 sated; falls -> hunger
    hydration: float = 1.0
    temperature: float = 37.0         # core body temp (C)
    hardware_stress: float = 0.0      # metabolic strain from hardware memory limits

    # Endocrine
    crf: float = 0.0                  # corticotropin-releasing factor
    cortisol: float = 0.2             # circulating stress hormone
    testosterone: float = 1.0
    oxytocin: float = 1.0
    vasopressin: float = 1.0
    name: str = "hypothalamus"

    def _circadian_wake(self) -> float:
        """Process C: circadian wake propensity, peaks mid-day (~14h), troughs at night."""
        return 0.5 + 0.5 * np.cos((self.clock_hours - 14.0) / 24.0 * 2 * np.pi)

    def melatonin(self, light: float) -> float:
        """Pineal output: high in biological night (peak ~3am), suppressed by light."""
        night = 0.5 + 0.5 * np.cos((self.clock_hours - 3.0) / 24.0 * 2 * np.pi)
        return float(np.clip(night * (1.0 - 0.8 * light), 0.0, 1.0))

    def step(self, dt_ms: float, inputs: dict) -> dict:
        # Advance clock (dt is ms; convert to hours)
        self.clock_hours = (self.clock_hours + dt_ms / 3.6e6) % 24.0
        light = inputs.get("light", None)
        if light is None:
            light = 1.0 if 6 < self.clock_hours < 20 else 0.0
        light = float(light)
        stressor = float(inputs.get("stressor", 0.0))
        metabolic = float(inputs.get("metabolic_cost", 0.0002))
        self.hardware_stress = float(inputs.get("hardware_stress", 0.0))

        # --- Two-Process Sleep Regulation ------------------------------------
        wake_c = self._circadian_wake()
        if self.asleep:
            # Discharge sleep pressure during sleep (~8h time constant)
            tau_discharge = 8.0 * 3.6e6
            self.sleep_pressure = float(max(0.0, self.sleep_pressure - dt_ms / tau_discharge))
            # Wake up if sleep pressure is low and circadian wake propensity rises
            if self.sleep_pressure < 0.15 and wake_c > 0.4:
                self.asleep = False
        else:
            # Accumulate sleep pressure awake; accelerated by cognitive & hardware strain
            tau_accum = 16.0 * 3.6e6
            strain_factor = 1.0 + 2.0 * self.hardware_stress
            self.sleep_pressure = float(self.sleep_pressure + strain_factor * (dt_ms / tau_accum))
            # Fall asleep if sleep pressure is exhausted or night arrives
            if self.sleep_pressure > self.sleep_threshold:
                self.asleep = True

        mel = self.melatonin(light)
        # Net arousal drive: circadian wake + (1 - sleep_pressure) - melatonin
        arousal_drive = float(np.clip(wake_c + 0.4 * (1.0 - self.sleep_pressure) - 0.5 * mel, 0.05, 1.0))

        # --- HPA Stress Axis -------------------------------------------------
        # Stressors & hardware strain drive CRF; Cortisol provides negative feedback
        combined_stress = stressor + 0.5 * self.hardware_stress
        d_crf = combined_stress - 0.3 * self.crf - 0.4 * self.cortisol
        self.crf = float(np.clip(self.crf + dt_ms * 0.0005 * d_crf, 0.0, 3.0))

        d_cortisol = 0.4 * self.crf - 0.2 * (self.cortisol - 0.2)
        self.cortisol = float(np.clip(self.cortisol + dt_ms * 0.0002 * d_cortisol, 0.05, 2.5))

        # Needs consolidation when sleep pressure is high
        needs_consolidation = bool(self.sleep_pressure >= 0.85 or self.asleep)

        return {
            "clock_hours": self.clock_hours,
            "sleep_pressure": self.sleep_pressure,
            "asleep": self.asleep,
            "needs_consolidation": needs_consolidation,
            "melatonin": mel,
            "arousal_drive": arousal_drive,
            "crf": self.crf,
            "cortisol": self.cortisol,
            "testosterone": self.testosterone,
            "oxytocin": self.oxytocin,
            "vasopressin": self.vasopressin,
            "hunger": float(1.0 - self.glucose),
            "hardware_stress": self.hardware_stress,
            "rem": float(1.0 if self.asleep and self.sleep_pressure < 0.5 else 0.0),
        }
