"""
HPO Module Initialization.

This module provides hyperparameter optimization algorithms for the RL pipeline.

Attributes:
    BaseHPO: Abstract base class for HPO algorithms.
    DifferentialEvolutionHyperband: DEHB HPO algorithm.
    HypRLHPO: HypRL HPO algorithm.
    OptunaHPO: Optuna HPO algorithm.
    RayTuneHPO: Ray Tune HPO algorithm.

Example:
    >>> from logic.src.pipeline.learning.hpo import OptunaHPO
    >>> optuna_hpo = OptunaHPO()
    >>> optuna_hpo
    OptunaHPO(n_trials=100, direction='maximize', study_name=None, storage=None)
"""

from logic.src.pipeline.learning.hpo.base import BaseHPO
from logic.src.pipeline.learning.hpo.dehb import DifferentialEvolutionHyperband
from logic.src.pipeline.learning.hpo.hyp_rl import HypRLHPO
from logic.src.pipeline.learning.hpo.optuna_hpo import OptunaHPO
from logic.src.pipeline.learning.hpo.ray_tune_hpo import RayTuneHPO

__all__ = ["BaseHPO", "DifferentialEvolutionHyperband", "HypRLHPO", "OptunaHPO", "RayTuneHPO"]
