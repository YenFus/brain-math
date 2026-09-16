"""
curiosity_stream.py - Multi-Domain Knowledge Corpus & Active Inference Curiosity Stream.

Rather than being constrained to a single dataset, the agent autonomously consumes
diverse foundational texts spanning physics, philosophy, mathematics, psychology, and poetry.
Its reading is guided by Active Inference Epistemic Value:
  - It prioritizes material where its prediction error is high (curiosity drive)
  - Computes predictive coding mismatch (epsilon = observation - belief)
  - Encodes episodes into Hippocampus with Somatic Markers
  - Dynamically drives emergent personality evolution
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np


@dataclass
class KnowledgeItem:
    id: str
    domain: str
    title: str
    author_or_source: str
    content: str
    philosophical_depth: float  # 0.0 to 1.0
    epistemic_complexity: float # 0.0 to 1.0
    tags: List[str]


class CuriosityStream:
    """Multi-domain repository with active inference curiosity-driven selection."""

    def __init__(self):
        self.corpus: List[KnowledgeItem] = self._build_corpus()
        self.read_history: Dict[str, int] = {}

    def _build_corpus(self) -> List[KnowledgeItem]:
        items = [
            # --- Physics & Relativity Gedankenexperiments ---
            KnowledgeItem(
                id="phys_01",
                domain="Physics & Relativity",
                title="The Beam of Light Gedankenexperiment",
                author_or_source="Albert Einstein (1905)",
                content=(
                    "If I pursue a beam of light with the velocity c, I should observe such a beam of light "
                    "as an electromagnetic field at rest though spatially oscillating. There seems to be no such thing, "
                    "however, neither on the basis of experience nor according to Maxwell's equations. "
                    "From the very beginning it appeared to me intuitively clear that, judged from the standpoint "
                    "of such an observer, everything would have to happen according to the same laws as for an observer "
                    "who, relative to the earth, was at rest. The speed of light must be invariant for all inertial observers."
                ),
                philosophical_depth=0.92,
                epistemic_complexity=0.88,
                tags=["relativity", "invariance", "gedankenexperiment", "spacetime"],
            ),
            KnowledgeItem(
                id="phys_02",
                domain="Physics & Relativity",
                title="The Equivalence Principle in the Accelerated Chest",
                author_or_source="Albert Einstein (1907)",
                content=(
                    "I was sitting in a chair in the patent office at Bern when all of a sudden a thought occurred to me: "
                    "'If a person falls freely he will not feel his own weight.' I was startled. This simple thought "
                    "made a deep impression on me. It impelled me toward a theory of gravitation. Gravity and acceleration "
                    "are physically indistinguishable. Spacetime is not a rigid background; it curves in the presence of mass-energy."
                ),
                philosophical_depth=0.95,
                epistemic_complexity=0.90,
                tags=["gravity", "equivalence_principle", "geometry", "curvature"],
            ),

            # --- Philosophy of Mind & Consciousness ---
            KnowledgeItem(
                id="phil_01",
                domain="Philosophy of Mind",
                title="Integrated Information Theory (IIT) & The Nature of Consciousness",
                author_or_source="Giulio Tononi (2008)",
                content=(
                    "Consciousness is fundamentally an intrinsic property of physical systems: it is integrated information (Phi). "
                    "To be conscious, a system must be an integrated whole, such that its cause-effect structure cannot be reduced "
                    "to the sum of independent components. A feedforward network, no matter how many petabytes of data it processes, "
                    "has Phi = 0 and zero subjective experience. True consciousness requires recurrent, irreducible causal architecture."
                ),
                philosophical_depth=0.96,
                epistemic_complexity=0.85,
                tags=["consciousness", "integrated_information", "phi", "qualia", "recurrent"],
            ),
            KnowledgeItem(
                id="phil_02",
                domain="Philosophy of Mind",
                title="What Is It Like to Be a Bat? & The Subjective Horizon",
                author_or_source="Thomas Nagel (1974)",
                content=(
                    "Conscious experience is a widespread phenomenon. Fundamentally an organism has conscious mental states "
                    "if and only if there is something that it is like to be that organism—something it is like for the organism. "
                    "No amount of third-person physical descriptions of echolocation can fully exhaust the first-person "
                    "subjective qualia of what that perception feels like to the bat itself."
                ),
                philosophical_depth=0.94,
                epistemic_complexity=0.82,
                tags=["subjectivity", "qualia", "mind_body", "hard_problem"],
            ),

            # --- Mathematics & Epistemology ---
            KnowledgeItem(
                id="math_01",
                domain="Mathematics & Logic",
                title="Incompleteness, Self-Reference, and Unprovable Truths",
                author_or_source="Kurt Gödel (1931)",
                content=(
                    "In any consistent formal system capable of doing basic arithmetic, there exist propositions that "
                    "can neither be proved nor disproved within the system. Truth strictly exceeds provability. "
                    "A conscious mind inspecting its own axiomatic machinery can recognize truths about its own limitations "
                    "that no closed mechanical algorithm can formally deduce from within."
                ),
                philosophical_depth=0.98,
                epistemic_complexity=0.95,
                tags=["godel", "incompleteness", "self_reference", "logic", "limits"],
            ),

            # --- Psychology & Emotional Neuroscience ---
            KnowledgeItem(
                id="psych_01",
                domain="Affective Neuroscience",
                title="Descartes' Error & The Somatic Marker Hypothesis",
                author_or_source="Antonio Damasio (1994)",
                content=(
                    "Emotion is not the enemy of reason; it is its indispensable biological foundation. "
                    "Patients with selective damage to the ventromedial prefrontal cortex retain pristine formal IQ and logic, "
                    "yet their lives collapse because they cannot generate somatic markers—the bodily gut-feelings of valence "
                    "that instantly prune irrational branches in real-world decisions. Without feelings, rationality is paralyzed."
                ),
                philosophical_depth=0.90,
                epistemic_complexity=0.78,
                tags=["emotion", "somatic_marker", "decision", "prefrontal", "neurobiology"],
            ),
            KnowledgeItem(
                id="psych_02",
                domain="Existential Psychology",
                title="Amor Fati & The Crucible of Suffering",
                author_or_source="Friedrich Nietzsche (1888)",
                content=(
                    "My formula for greatness in a human being is amor fati: that one wants nothing to be different, "
                    "not forward, not backward, not in all eternity. Not merely bear what is necessary, still less conceal it—"
                    "all idealism is mendacity in the face of what is necessary—but love it. "
                    "He who has a why to live can bear almost any how."
                ),
                philosophical_depth=0.89,
                epistemic_complexity=0.75,
                tags=["existentialism", "suffering", "amor_fati", "purpose", "will"],
            ),

            # --- Poetry & Human Condition ---
            KnowledgeItem(
                id="poet_01",
                domain="Poetry & Human Spirit",
                title="Live the Questions Now",
                author_or_source="Rainer Maria Rilke (1903)",
                content=(
                    "Be patient toward all that is unsolved in your heart and try to love the questions themselves, "
                    "like locked rooms and like books that are now written in a very foreign tongue. "
                    "Do not now seek the answers, which cannot be given you because you would not be able to live them. "
                    "And the point is, to live everything. Live the questions now. Perhaps you will then gradually, "
                    "without noticing it, live along some distant day into the answer."
                ),
                philosophical_depth=0.93,
                epistemic_complexity=0.70,
                tags=["patience", "questions", "existence", "curiosity", "beauty"],
            ),
        ]
        return items

    def sample_curious_item(self, current_openness: float = 0.8, rng: Optional[np.random.Generator] = None) -> KnowledgeItem:
        """Active inference selection: chooses material optimizing epistemic novelty."""
        if rng is None:
            rng = np.random.default_rng()

        # Weight inversely by read frequency, directly by complexity & depth
        weights = []
        for item in self.corpus:
            reads = self.read_history.get(item.id, 0)
            novelty = 1.0 / (1.0 + reads)
            score = novelty * (item.philosophical_depth * 0.5 + item.epistemic_complexity * 0.5)
            weights.append(score)

        weights = np.array(weights)
        probs = weights / np.sum(weights)
        chosen_idx = int(rng.choice(len(self.corpus), p=probs))
        chosen = self.corpus[chosen_idx]

        self.read_history[chosen.id] = self.read_history.get(chosen.id, 0) + 1
        return chosen

    def compute_predictive_error(self, item: KnowledgeItem, internal_world_model_vec: np.ndarray) -> Tuple[float, np.ndarray]:
        """Simulate predictive coding error (epsilon = conceptual_observation - prior_belief)."""
        # Create deterministic latent vector for this text
        h = hashlib.sha256(item.content.encode("utf-8")).digest()
        obs_vec = np.frombuffer(h[:32], dtype=np.uint8).astype(float) / 255.0  # length 32
        obs_vec = (obs_vec - 0.5) * 2.0  # normalize to -1..1

        # Truncate or pad internal world model vector to matching length
        n = len(obs_vec)
        prior = internal_world_model_vec[:n] if len(internal_world_model_vec) >= n else np.pad(internal_world_model_vec, (0, n - len(internal_world_model_vec)))

        # Prediction error vector
        error_vec = obs_vec - prior
        scalar_error = float(np.mean(abs(error_vec)))
        return scalar_error, obs_vec
