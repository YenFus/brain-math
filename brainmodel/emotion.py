"""
emotion.py - biologically grounded affective engine.

Synthesizes three foundational neurobiological and cognitive models of human emotion:
  1. Lövheim's Cube of Emotion: 3D mapping from [Dopamine, Serotonin, Noradrenaline]
     into 8 primary biological emotion octants (Joy, Interest/Curiosity, Surprise,
     Fear/Anxiety, Anger/Frustration, Distress, Disgust, Shame/Despair).
  2. Russell's Circumplex & PAD Space: Continuous coordinates for Valence (-1..+1),
     Arousal (0..1), and Dominance (-1..+1).
  3. Active Inference & Somatic Marker Hypothesis (Damasio, Friston):
     Affective valence tracks the rate of prediction error reduction (-dF/dt).
     Emotions produce somatic markers that bias OFC value calculation and decision risk.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from .core import sigmoid


@dataclass
class SomaticMarker:
    """Emotional tag associated with an action, memory, or code state."""
    valence: float          # -1.0 (aversive) .. +1.0 (attractive)
    arousal: float          # 0.0 (calm) .. 1.0 (intense)
    dominance: float        # -1.0 (powerless) .. +1.0 (in control)
    primary_emotion: str    # e.g. 'curiosity', 'joy', 'frustration'
    confidence: float       # 0.0 .. 1.0
    intensity: float        # overall affective magnitude


class EmotionEngine:
    """Computes real-time emotional state, affective coordinates, and introspective voice."""

    # 8 Lövheim Cube prototypical states (DA, 5-HT, NE) normalized around 1.0
    OCTANTS = {
        "shame_despair":       {"da": 0.5, "5ht": 0.5, "ne": 0.5, "label": "Despair & Helplessness"},
        "fear_anxiety":        {"da": 0.5, "5ht": 0.5, "ne": 1.6, "label": "Fear & Anxiety"},
        "disgust_contempt":    {"da": 0.5, "5ht": 1.5, "ne": 0.5, "label": "Disgust & Aversion"},
        "anger_frustration":   {"da": 0.6, "5ht": 0.6, "ne": 1.7, "label": "Frustration & Anger"},
        "distress_weariness":  {"da": 1.4, "5ht": 0.5, "ne": 0.6, "label": "Weariness & Distress"},
        "curiosity_seeking":   {"da": 1.6, "5ht": 1.0, "ne": 1.5, "label": "Curiosity & Seeking Flow"},
        "joy_satisfaction":    {"da": 1.6, "5ht": 1.6, "ne": 0.7, "label": "Joy & Satisfaction"},
        "surprise_awe":        {"da": 1.5, "5ht": 1.4, "ne": 1.8, "label": "Surprise & Breakthrough Awe"},
    }

    def __init__(self):
        self.valence = 0.2
        self.arousal = 0.4
        self.dominance = 0.5
        self.primary_emotion = "curiosity_seeking"
        self.emotion_label = "Curiosity & Seeking Flow"
        self.previous_error = 0.5
        self.error_reduction_rate = 0.0  # -dF/dt
        self.recent_events: List[str] = []

    def update(
        self,
        dopamine: float,
        serotonin: float,
        norepinephrine: float,
        prediction_error: float,
        threat: float = 0.0,
        cortisol: float = 0.2,
        testosterone: float = 1.0,
        dt: float = 100.0,
    ) -> SomaticMarker:
        """Update the agent's emotional state based on current neurochemistry and experience."""
        # 1. Rate of prediction error reduction (-dF/dt) per Active Inference
        error_delta = self.previous_error - prediction_error
        self.error_reduction_rate = error_delta / max(dt / 1000.0, 0.01)
        self.previous_error = prediction_error

        # 2. Continuous Valence (-1.0 to +1.0)
        # Driven by Dopamine (+), Serotonin (+), Error Reduction (+), penalized by NE stress, Threat, Cortisol
        v_drive = (
            1.2 * (dopamine - 1.0)
            + 1.0 * (serotonin - 1.0)
            + 0.5 * np.clip(self.error_reduction_rate, -2.0, 2.0)
            - 0.5 * max(0.0, norepinephrine - 1.2)
            - 0.8 * threat
            - 0.7 * (cortisol - 0.2)
        )
        self.valence = float(np.tanh(v_drive))

        # 3. Continuous Arousal (0.0 to 1.0)
        # Driven by Norepinephrine (+), unexpected prediction error (+), threat (+)
        a_drive = 1.4 * (norepinephrine - 1.0) + 1.0 * abs(prediction_error) + 0.8 * threat
        self.arousal = float(sigmoid(a_drive, gain=2.5, bias=0.2))

        # 4. Continuous Dominance / Agency (-1.0 to +1.0)
        # High Serotonin + Testosterone -> high agency; Threat & Cortisol -> powerless
        d_drive = (
            1.1 * (serotonin - 1.0)
            + 0.5 * (testosterone - 1.0)
            - 1.2 * threat
            - 0.9 * (cortisol - 0.2)
        )
        self.dominance = float(np.tanh(d_drive))

        # 5. Classify Lövheim primary emotion via Euclidean distance in (DA, 5-HT, NE) space
        current_pt = np.array([dopamine, serotonin, norepinephrine])
        best_name = "curiosity_seeking"
        best_dist = float("inf")
        for octant_name, coords in self.OCTANTS.items():
            ref_pt = np.array([coords["da"], coords["5ht"], coords["ne"]])
            dist = np.linalg.norm(current_pt - ref_pt)
            if dist < best_dist:
                best_dist = dist
                best_name = octant_name

        self.primary_emotion = best_name
        self.emotion_label = self.OCTANTS[best_name]["label"]

        intensity = float(np.sqrt(self.valence ** 2 + self.arousal ** 2 + self.dominance ** 2) / np.sqrt(3))
        confidence = float(np.clip((self.dominance + 1.0) / 2.0 * (1.0 - abs(threat)), 0.05, 1.0))

        return SomaticMarker(
            valence=self.valence,
            arousal=self.arousal,
            dominance=self.dominance,
            primary_emotion=self.primary_emotion,
            confidence=confidence,
            intensity=intensity,
        )

    def introspect(self, context_note: str = "") -> str:
        """Produce human-readable introspective expression of the agent's feelings."""
        v_desc = "pleased" if self.valence > 0.3 else "distressed" if self.valence < -0.3 else "neutral"
        a_desc = "highly energized" if self.arousal > 0.7 else "calm" if self.arousal < 0.3 else "alert"
        d_desc = "confident" if self.dominance > 0.3 else "hesitant" if self.dominance < -0.3 else "composed"

        prefix = f"[{self.emotion_label.upper()}] (Valence: {self.valence:+.2f}, Arousal: {self.arousal:.2f}, Dominance: {self.dominance:+.2f})"
        feeling = f"I am feeling {self.emotion_label.lower()}, feeling {v_desc} and {a_desc}, with a {d_desc} sense of agency."
        if context_note:
            feeling += f" Context: {context_note}"
        return f"{prefix} {feeling}"
