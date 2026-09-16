"""
brainmodel
==========

A multi-scale, runnable mathematical model of the human brain with
Human-like Emotion Dynamics, Dual-Process Decision Deliberation,
and 5-Level Recursive Self-Improvement (RSI).
"""

from .core import Simulation, Subsystem
from .neuron import LIFNeuron, RatePopulation
from .synapse import PlasticSynapse
from .neuromodulators import Neuromodulators, Modulator
from .brainstem import Brainstem
from .thalamus import Thalamus
from .hypothalamus import Hypothalamus
from .cortex import Cortex, WorkingMemory
from .hippocampus import Hippocampus, Episode
from .cerebellum import Cerebellum
from .basal_ganglia import BasalGanglia
from .decision import DriftDiffusion, ValueLearner
from .dual_process import HeuristicSystematic, DualProcessDecision
from .networks import TripleNetwork
from .consciousness import GlobalWorkspace, integrated_information
from .brain import Brain, LimbicAppraisal
from .emotion import EmotionEngine, SomaticMarker
from .hardware_governor import HardwareGovernor, MemorySnapshot
from .local_llm import LocalLLMEngine
from .antigravity_bridge import AntigravityBridge, ReviewItem
from .agent import HumanlikeAIAgent
from .rsi import AgentPolicy, EvaluatorHarness, EvaluationResult, SelfEditor, RSIEngine, RSICycleResult

__all__ = [
    "Simulation",
    "Subsystem",
    "LIFNeuron",
    "RatePopulation",
    "PlasticSynapse",
    "Neuromodulators",
    "Modulator",
    "Brainstem",
    "Thalamus",
    "Hypothalamus",
    "Cortex",
    "WorkingMemory",
    "Hippocampus",
    "Episode",
    "Cerebellum",
    "BasalGanglia",
    "DriftDiffusion",
    "ValueLearner",
    "HeuristicSystematic",
    "DualProcessDecision",
    "TripleNetwork",
    "GlobalWorkspace",
    "integrated_information",
    "Brain",
    "LimbicAppraisal",
    "EmotionEngine",
    "SomaticMarker",
    "HardwareGovernor",
    "MemorySnapshot",
    "LocalLLMEngine",
    "AntigravityBridge",
    "ReviewItem",
    "HumanlikeAIAgent",
    "AgentPolicy",
    "EvaluatorHarness",
    "EvaluationResult",
    "SelfEditor",
    "RSIEngine",
    "RSICycleResult",
]

__version__ = "0.2.0"
