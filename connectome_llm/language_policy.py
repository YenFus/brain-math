"""
language_policy.py - Natural Language Semantic Policy Generator.

Translates human natural language instructions into dynamic CognitiveSetpoints
for descending motor execution by the connectome.
"""

from __future__ import annotations

import math
import re
from typing import Dict, List, Optional
import numpy as np
from connectome_llm.brainstem_bridge import CognitiveSetpoint


class LanguageConnectomePolicy:
    """Interprets natural language commands and generates continuous cognitive setpoints."""

    def __init__(self):
        # Semantic intent dictionary mapping keywords to behavioral postures
        self.intent_lexicon = {
            "sprint": {"mode": "SPRINT", "forward": 1.8, "urgency": 0.0, "vigilance": 0.4, "chemo": 1.2},
            "rush": {"mode": "SPRINT", "forward": 1.8, "urgency": 0.0, "vigilance": 0.4, "chemo": 1.2},
            "fast": {"mode": "SPRINT", "forward": 1.6, "urgency": 0.0, "vigilance": 0.5, "chemo": 1.1},
            "patrol": {"mode": "PATROL", "forward": 1.1, "urgency": 0.0, "vigilance": 0.8, "chemo": 0.6},
            "perimeter": {"mode": "PATROL", "forward": 1.0, "urgency": 0.0, "vigilance": 0.8, "chemo": 0.4},
            "explore": {"mode": "FORAGE", "forward": 1.2, "urgency": 0.0, "vigilance": 0.6, "chemo": 1.0},
            "forage": {"mode": "FORAGE", "forward": 1.2, "urgency": 0.0, "vigilance": 0.5, "chemo": 1.5},
            "feed": {"mode": "DWELL", "forward": 0.4, "urgency": 0.0, "vigilance": 0.3, "chemo": 1.8},
            "eat": {"mode": "DWELL", "forward": 0.4, "urgency": 0.0, "vigilance": 0.3, "chemo": 1.8},
            "dwell": {"mode": "DWELL", "forward": 0.3, "urgency": 0.0, "vigilance": 0.3, "chemo": 1.5},
            "evade": {"mode": "EVADE", "forward": 0.2, "urgency": 0.9, "vigilance": 1.0, "chemo": 0.2},
            "retreat": {"mode": "EVADE", "forward": 0.2, "urgency": 0.9, "vigilance": 1.0, "chemo": 0.2},
            "escape": {"mode": "EVADE", "forward": 0.3, "urgency": 1.0, "vigilance": 1.0, "chemo": 0.1},
            "back": {"mode": "EVADE", "forward": 0.2, "urgency": 0.8, "vigilance": 0.9, "chemo": 0.2},
            "stop": {"mode": "STANDBY", "forward": 0.0, "urgency": 0.0, "vigilance": 0.5, "chemo": 0.0},
            "standby": {"mode": "STANDBY", "forward": 0.0, "urgency": 0.0, "vigilance": 0.2, "chemo": 0.0},
            "pause": {"mode": "STANDBY", "forward": 0.0, "urgency": 0.0, "vigilance": 0.2, "chemo": 0.0},
        }

        self.current_instruction = "explore and forage for nutrients"
        self.active_setpoint = CognitiveSetpoint(behavior_mode="FORAGE", forward_drive=1.0)

    def set_instruction(self, instruction: str):
        """Set a new natural language mission/instruction."""
        self.current_instruction = instruction.lower().strip()
        self._parse_instruction()

    def _parse_instruction(self):
        """Parse natural language command into base behavioral setpoints."""
        tokens = re.findall(r'\w+', self.current_instruction)

        # Default settings
        mode = "FORAGE"
        fwd = 1.0
        urg = 0.0
        vig = 0.5
        chemo = 1.0

        for tok in tokens:
            if tok in self.intent_lexicon:
                lex = self.intent_lexicon[tok]
                mode = lex["mode"]
                fwd = lex["forward"]
                urg = lex["urgency"]
                vig = lex["vigilance"]
                chemo = lex["chemo"]
                break

        self.active_setpoint = CognitiveSetpoint(
            behavior_mode=mode,
            forward_drive=fwd,
            turn_bias=0.0,
            reversal_urgency=urg,
            vigilance=vig,
            chemo_attraction=chemo,
        )

    def evaluate(self, state: Dict) -> CognitiveSetpoint:
        """Continuously modulates setpoint based on environmental telemetry and instruction goals."""
        sp = CognitiveSetpoint(
            behavior_mode=self.active_setpoint.behavior_mode,
            forward_drive=self.active_setpoint.forward_drive,
            turn_bias=0.0,
            reversal_urgency=self.active_setpoint.reversal_urgency,
            vigilance=self.active_setpoint.vigilance,
            chemo_attraction=self.active_setpoint.chemo_attraction,
        )

        hx = state.get("head_x", 50.0)
        hy = state.get("head_y", 50.0)
        heading = state.get("heading", 0.0)
        min_whisker = state.get("min_whisker", 10.0)
        arena_size = state.get("arena_size", 100.0)

        # Context-dependent behavioral adaptation:
        # 1. If PATROL/PERIMETER: steer towards the arena boundary, then follow perimeter
        if sp.behavior_mode == "PATROL":
            dist_to_edge = min(hx, hy, arena_size - hx, arena_size - hy)
            if dist_to_edge > 20.0:
                # Steer towards nearest wall
                desired_heading = math.atan2(50.0 - hy, 50.0 - hx) + math.pi
                heading_err = (desired_heading - heading + math.pi) % (2 * math.pi) - math.pi
                sp.turn_bias = float(np.clip(heading_err * 0.5, -0.8, 0.8))
            else:
                # Follow wall by turning tangent
                sp.turn_bias = 0.25

        # 2. Obstacle Proximity Avoidance
        if min_whisker < 5.0:
            if sp.behavior_mode != "EVADE":
                # Escalate reversal urgency if collision is imminent
                sp.reversal_urgency = max(sp.reversal_urgency, float(np.clip((5.0 - min_whisker) / 4.0, 0.0, 1.0)))

        # 3. DWELL / FEEDING: If food is close, slow down to feed (Sawin et al. dopamine slowing)
        if state.get("food_near", False):
            if sp.behavior_mode in ["FORAGE", "DWELL"]:
                sp.forward_drive = 0.35
                sp.turn_bias = 0.0

        return sp
