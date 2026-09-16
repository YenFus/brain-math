"""
hippocampus.py - Modern Continuous Hopfield Network & Episodic Memory Consolidation.

Upgraded from classical toy Hopfield models to Modern Continuous Hopfield Networks
(Demircigil et al. 2017, Ramsauer et al. 2020):
  - Continuous energy function: E(x) = -1/beta * log(sum(exp(beta * x^T * xi_i))) + 1/2 * ||x||^2
  - Global update rule with exponential retrieval capacity (C ~ 2^(d/2)):
        x^(t+1) = Xi * softmax(beta * Xi^T * x^t)
  - Trisynaptic Circuit Architecture:
      Entorhinal Cortex -> Dentate Gyrus (DG pattern separation via sparsification)
      -> CA3 (Modern Hopfield recurrent autoassociative attractor network)
      -> CA1 (comparator & readout to cortex)
  - ACh-gated memory write: High acetylcholine enables plastic encoding of novel episodes.
  - Emotional Somatic Marker Tagging: Stores emotional valence, prediction error, and context.
  - Sleep Consolidation & Replay: Replays salient experiences during slow-wave/REM sleep.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class Episode:
    """An episodic memory unit stored in hippocampus."""
    pattern: np.ndarray             # dense feature representation
    tag: str                        # semantic or task label
    somatic_valence: float = 0.0    # emotional valence when experienced
    arousal: float = 0.5            # emotional arousal
    timestamp: float = 0.0          # chronological step
    salience: float = 1.0           # consolidation priority


@dataclass
class Hippocampus:
    n: int = 64                    # dimension of pattern representations
    neurogenesis: float = 1.0      # DG pattern separation fidelity (0.1..1.5)
    encode_gain: float = 1.0       # ACh-gated write permeability
    beta: float = 4.0              # Modern Hopfield inverse temperature
    max_episodes: int = 512        # capacity
    retrieval_noise: float = 0.15  # baseline noise during retrieval
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))
    name: str = "hippocampus"

    # Memory stores
    episodes: List[Episode] = field(default_factory=list)
    memory_matrix: Optional[np.ndarray] = None  # shape (n, M) for Hopfield attention
    stored: int = 0

    # Backwards compatibility legacy weight matrix
    W: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.W is None:
            self.W = np.zeros((self.n, self.n))

    # --- Dentate Gyrus (DG): Pattern Separation ------------------------------

    def dentate_gyrus_separate(self, pattern: np.ndarray) -> np.ndarray:
        """Sparsifies and orthogonalizes input to prevent catastrophic interference."""
        p = np.asarray(pattern, dtype=float).flatten()
        if len(p) != self.n:
            # Resize or pad/crop if necessary
            p = np.pad(p, (0, max(0, self.n - len(p))))[:self.n]
        norm = np.linalg.norm(p)
        if norm > 1e-9:
            p = p / norm

        # Non-linear expansion & k-winners-take-all sparsification
        k = int(max(4, int(self.n * 0.25 * self.neurogenesis)))
        top_indices = np.argpartition(abs(p), -k)[-k:]
        sparse = np.zeros_like(p)
        sparse[top_indices] = p[top_indices]
        sparse_norm = np.linalg.norm(sparse)
        return sparse / max(sparse_norm, 1e-9)

    # --- CA3: Modern Continuous Hopfield Autoassociative Network -------------

    def store(
        self,
        pattern: np.ndarray,
        tag: str = "",
        valence: float = 0.0,
        arousal: float = 0.5,
        salience: float = 1.0,
    ) -> None:
        """Write an episodic memory into the Modern Hopfield CA3 attractor store."""
        p = self.dentate_gyrus_separate(pattern)

        # ACh-gated encoding: if ACh is suppressed, encoding is blunted
        effective_salience = salience * max(0.1, self.encode_gain)

        ep = Episode(
            pattern=p,
            tag=tag,
            somatic_valence=valence,
            arousal=arousal,
            timestamp=float(self.stored),
            salience=effective_salience,
        )

        if len(self.episodes) >= self.max_episodes:
            # Evict the lowest-salience memory
            min_idx = int(np.argmin([e.salience for e in self.episodes]))
            self.episodes[min_idx] = ep
        else:
            self.episodes.append(ep)

        # Update CA3 memory matrix Xi of shape (n, M)
        self.memory_matrix = np.column_stack([e.pattern for e in self.episodes])
        self.stored += 1

        # Also update legacy W matrix for backwards-compatibility tests
        p_code = np.sign(p + 1e-9)
        self.W += self.encode_gain * np.outer(p_code, p_code) / self.n
        np.fill_diagonal(self.W, 0.0)

    def recall(self, cue: np.ndarray, iters: int = 4) -> np.ndarray:
        """Modern Hopfield softmax attention retrieval from partial/noisy cue."""
        if self.memory_matrix is None or len(self.episodes) == 0:
            return np.asarray(cue, dtype=float)

        x = self.dentate_gyrus_separate(cue)
        Xi = self.memory_matrix  # (n, M)

        for _ in range(iters):
            # Compute inner products x^T * Xi
            scores = self.beta * (Xi.T @ x)
            # Numerically stable softmax
            max_s = np.max(scores)
            weights = np.exp(scores - max_s)
            weights /= np.sum(weights)

            # Continuous state update: x = Xi * weights
            x = Xi @ weights
            # Thermal/retrieval noise
            noise = (self.retrieval_noise / max(self.neurogenesis, 0.2)) * self.rng.standard_normal(self.n) * 0.05
            x = x + noise
            norm = np.linalg.norm(x)
            if norm > 1e-9:
                x /= norm

        return x

    def recall_accuracy(self, original: np.ndarray, corrupt: float = 0.3) -> float:
        """Measure pattern completion overlap when given a corrupted cue."""
        target = self.dentate_gyrus_separate(original)
        cue = target.copy()
        flip = self.rng.random(len(cue)) < corrupt
        cue[flip] += self.rng.standard_normal(np.sum(flip)) * 0.5
        recovered = self.recall(cue)

        # Cosine similarity between recovered and target
        similarity = float(np.dot(recovered, target) / (np.linalg.norm(recovered) * np.linalg.norm(target) + 1e-9))
        return float(np.clip(0.5 + 0.5 * similarity, 0.0, 1.0))

    # --- Sleep Replay & Offline Consolidation ---------------------------------

    def sleep_replay(self, n_episodes: int = 5) -> List[Episode]:
        """Sample high-salience episodes for cortical replay and weight consolidation."""
        if not self.episodes:
            return []
        # Probability weighted by salience and emotional arousal
        priorities = np.array([e.salience * (1.0 + abs(e.somatic_valence)) for e in self.episodes])
        priorities /= np.sum(priorities)
        count = min(n_episodes, len(self.episodes))
        chosen_indices = self.rng.choice(len(self.episodes), size=count, replace=False, p=priorities)
        return [self.episodes[i] for i in chosen_indices]

    def consolidate(self, decay_rate: float = 0.05) -> None:
        """Synaptic pruning during sleep: slightly decays low-salience memories."""
        for ep in self.episodes:
            ep.salience = max(0.05, ep.salience * (1.0 - decay_rate))
