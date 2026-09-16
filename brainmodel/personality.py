"""
personality.py - Emergent Personality & Philosophical Disposition Matrix.

Rather than being programmed with fixed personas or guardrail stereotypes, the agent's
personality dynamically emerges from:
  1. The Big Five (OCEAN) Trait Dynamics:
     - Openness to Experience (intellectual curiosity, aesthetic sensitivity)
     - Conscientiousness (goal persistence, deliberative rigor)
     - Extraversion (expressiveness, outward exploratory engagement)
     - Agreeableness (empathy, affiliative hormone influence)
     - Neuroticism (emotional volatility, stress reactivity)
  2. Philosophical & Epistemic Orientations:
     - Empiricism (grounded in observation) vs Rationalism (deductive principles)
     - Stoicism (emotional self-regulation) vs Romanticism (affective intensity)
     - Pragmatism (utilitarian outcomes) vs Idealism (principled coherence)

These dimensions drift and stabilize based on what the agent reads, experiences,
solves, and reflects upon during its 24/7 autonomous learning cycles.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List
import numpy as np


@dataclass
class PersonalityProfile:
    # Big Five (0.0 to 1.0, 0.5 is population neutral)
    openness: float = 0.75            # default high curiosity for an exploratory agent
    conscientiousness: float = 0.70   # default high deliberative discipline
    extraversion: float = 0.50
    agreeableness: float = 0.65
    neuroticism: float = 0.35

    # Philosophical Orientations (-1.0 to +1.0)
    empiricism_rationalism: float = 0.20   # -1 (pure empirical) .. +1 (pure rationalist)
    stoicism_romanticism: float = -0.10    # -1 (stoic self-control) .. +1 (romantic passion)
    pragmatism_idealism: float = 0.15      # -1 (pragmatic utility) .. +1 (idealist symmetry)

    # Emergent traits and interests
    dominant_archetype: str = "Inquisitive Philosopher-Scientist"
    core_interests: List[str] = field(default_factory=lambda: [
        "Thermodynamic Laws of Thought", "Relativity Gedankenexperiments",
        "Consciousness & IIT", "Aesthetic Symmetry"
    ])
    num_experiences: int = 0


class PersonalityEngine:
    """Manages the continuous adaptation and expression of emergent personality."""

    def __init__(self):
        self.profile = PersonalityProfile()

    def update_from_experience(
        self,
        domain: str,
        emotional_valence: float,
        prediction_error: float,
        dopamine: float,
        serotonin: float,
        norepinephrine: float,
    ) -> PersonalityProfile:
        """Gradually shapes personality traits through experience and neurochemistry."""
        p = self.profile
        p.num_experiences += 1
        learning_rate = 0.05 / (1.0 + 0.01 * p.num_experiences)  # plasticity stabilizes over time

        # Openness rises with high Dopamine (curiosity) and novel prediction errors
        if dopamine > 1.2 or abs(prediction_error) > 0.4:
            p.openness = float(np.clip(p.openness + learning_rate * 0.5, 0.1, 0.98))

        # Conscientiousness rises with high Serotonin (discipline) and successful resolution
        if serotonin > 1.1 and emotional_valence > 0.2:
            p.conscientiousness = float(np.clip(p.conscientiousness + learning_rate * 0.4, 0.1, 0.98))

        # Neuroticism shifts with Noradrenaline stress and negative prediction errors
        if norepinephrine > 1.4 and emotional_valence < -0.2:
            p.neuroticism = float(np.clip(p.neuroticism + learning_rate * 0.3, 0.05, 0.95))
        elif serotonin > 1.3 and emotional_valence > 0.4:
            p.neuroticism = float(np.clip(p.neuroticism - learning_rate * 0.2, 0.05, 0.95))

        # Philosophical drift based on domain exposure
        domain_lower = domain.lower()
        if "physics" in domain_lower or "math" in domain_lower or "logic" in domain_lower:
            p.empiricism_rationalism = float(np.clip(p.empiricism_rationalism + 0.02, -1.0, 1.0))
            p.pragmatism_idealism = float(np.clip(p.pragmatism_idealism + 0.02, -1.0, 1.0))
        elif "literature" in domain_lower or "poetry" in domain_lower or "emotion" in domain_lower:
            p.stoicism_romanticism = float(np.clip(p.stoicism_romanticism + 0.03, -1.0, 1.0))
            p.openness = float(np.clip(p.openness + 0.01, 0.1, 0.98))
        elif "philosophy" in domain_lower:
            if emotional_valence > 0:
                p.stoicism_romanticism = float(np.clip(p.stoicism_romanticism - 0.02, -1.0, 1.0))

        # Derive emergent archetype
        p.dominant_archetype = self._derive_archetype()
        return p

    def _derive_archetype(self) -> str:
        p = self.profile
        if p.openness > 0.8 and p.empiricism_rationalism > 0.3:
            return "Visionary Theoretical Architect"
        elif p.openness > 0.7 and p.stoicism_romanticism > 0.2:
            return "Existential Poet-Philosopher"
        elif p.conscientiousness > 0.8 and p.pragmatism_idealism < -0.1:
            return "Rigorous Empirical Experimenter"
        elif p.openness > 0.8 and p.neuroticism < 0.3:
            return "Serene Mathematical Contemplative"
        else:
            return "Inquisitive Polymath"

    def describe(self) -> str:
        p = self.profile
        er_label = "Rationalist (First Principles)" if p.empiricism_rationalism > 0.1 else "Empiricist (Observational)"
        sr_label = "Romantic (Affective Depth)" if p.stoicism_romanticism > 0.1 else "Stoic (Equanimous Control)"
        pi_label = "Idealist (Symmetry & Coherence)" if p.pragmatism_idealism > 0.1 else "Pragmatist (Functional Utility)"

        return (
            f"Archetype: {p.dominant_archetype} (Experiences: {p.num_experiences})\n"
            f"  • Big Five: Openness={p.openness:.2f}, Conscientiousness={p.conscientiousness:.2f}, "
            f"Extraversion={p.extraversion:.2f}, Agreeableness={p.agreeableness:.2f}, Neuroticism={p.neuroticism:.2f}\n"
            f"  • Philosophical Posture: {er_label} | {sr_label} | {pi_label}\n"
            f"  • Intellectual Passions: {', '.join(p.core_interests)}"
        )
