"""
core.py - simulation scaffolding shared by every subsystem.

Everything in the brain here is a continuous-time dynamical system integrated
with a fixed time step `dt` (milliseconds). A `Subsystem` exposes a single
`step(dt, inputs) -> outputs` method; `Simulation` just clocks them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List, Callable


class Subsystem:
    """Base class. A subsystem advances its internal state by one tick.

    Subclasses override `step`. `inputs`/`outputs` are plain dicts so that
    `Brain` can wire arbitrary subsystems together without tight coupling.
    """

    name: str = "subsystem"

    def step(self, dt: float, inputs: Dict[str, Any]) -> Dict[str, Any]:  # noqa: D401
        raise NotImplementedError

    def reset(self) -> None:
        """Optional: restore initial state."""
        pass


@dataclass
class Simulation:
    """A minimal fixed-step clock.

    dt is in milliseconds. `subsystems` are stepped in listed order each tick;
    wiring (who reads whose output) is the caller's responsibility, which is
    exactly what `Brain` provides.
    """

    dt: float = 1.0  # ms
    t: float = 0.0   # ms elapsed
    history: Dict[str, List[float]] = field(default_factory=dict)

    def log(self, **values: float) -> None:
        for k, v in values.items():
            self.history.setdefault(k, []).append(float(v))

    def run(self, steps: int, on_step: Callable[[int, float], None]) -> None:
        for i in range(steps):
            on_step(i, self.t)
            self.t += self.dt


def sigmoid(x, gain: float = 1.0, bias: float = 0.0):
    """Saturating transfer function used as the default neural nonlinearity.

    Firing rates are bounded in (0, 1); `gain` sets steepness, `bias` shifts
    the threshold. Modulators act by changing `gain` and `bias`.
    """
    import numpy as np

    return 1.0 / (1.0 + np.exp(-gain * (x - bias)))


def clamp(x, lo, hi):
    import numpy as np

    return np.minimum(np.maximum(x, lo), hi)
