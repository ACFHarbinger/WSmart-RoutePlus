"""
Base classes for simulation policy Hyperparameter Optimization (HPO).

Includes an abstract base class for defining HPO strategies and a concrete
implementation for finding the best performing policy from the pool.

Attributes:
    PolicyHPOBase: Abstract base class for HPO strategies.

Example:
    >>> # from logic.src.policies.helpers.hpo import PolicyHPOBase
    >>> # class MyHPO(PolicyHPOBase):
    >>> #     def run(self, n_trials=50): ...
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import optuna
import torch
from optuna.integration import BoTorchSampler

from logic.src.configs import Config


class PolicyHPOBase(ABC):
    """Abstract base class for simulation policy Hyperparameter Optimization.

    Attributes:
        cfg (Config): Root application configuration.
        best_params (Optional[Dict[str, Any]]): Best parameters found during HPO.
        best_value (float): Best objective value achieved (scalar mode).
        best_values (Optional[List[float]]): Best objective values achieved (multi-objective mode).
    """

    def __init__(self, cfg: Config, search_space: Optional[Dict[str, Any]] = None):
        """Initialize PolicyHPO.

        Args:
            cfg: Root application configuration.
            search_space: Search space for this policy.
        """
        self.cfg = cfg
        self.search_space = search_space or {}
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_value: float = float("-inf")
        self.best_values: Optional[List[float]] = None

    @abstractmethod
    def run(self, cfg: Config) -> Union[float, List[float]]:
        """Run a full simulation trial.

        Args:
            cfg: Config object with sampled parameters already applied.

        Returns:
            Union[float, List[float]]: Metric value(s).
        """
        pass

    def run_iterative(self, cfg: Config, max_steps: int) -> Any:
        """Generator for multi-fidelity simulation reporting.

        Yields intermediate metrics after each step (e.g. after each simulation day).
        By default, this simply runs the full simulation and yields the final result.

        Args:
            cfg: Config with sampled parameters.
            max_steps: Maximum resource budget.

        Yields:
            Union[float, List[float]]: Intermediate metrics.
        """
        yield self.run(cfg)

    def suggest_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest all parameters in the search space for a trial.

        Args:
            trial: The Optuna trial object.

        Returns:
            Dict[str, Any]: Mapping of (possibly dot-separated) config paths to values.
        """
        return {name: self.suggest_param(trial, name, spec) for name, spec in self.search_space.items()}

    def _apply_params(self, cfg: Config, params: Dict[str, Any]) -> None:
        """Apply sampled parameters to the config.

        Supports dot-separated paths for nested keys, including list indices
        (e.g., 'sim.full_policies.0.alns.max_iterations').

        Args:
            cfg: Config object to mutate with new parameters.
            params: Dictionary mapping dot-separated paths to values.

        Raises:
            AttributeError: If a segment of the config path cannot be resolved.
        """
        for key, value in params.items():
            parts = key.split(".")
            target = cfg
            for part in parts[:-1]:
                if hasattr(target, part):
                    target = getattr(target, part)
                elif isinstance(target, dict) and part in target:
                    target = target[part]
                elif isinstance(target, list) and part.isdigit():
                    target = target[int(part)]
                else:
                    try:
                        target = target[part] if not part.isdigit() else target[int(part)]  # type: ignore[index]
                    except (KeyError, IndexError, TypeError) as err:
                        raise AttributeError(f"Cannot find config path: {key}") from err

            last_part = parts[-1]
            try:
                if hasattr(target, last_part):
                    setattr(target, last_part, value)
                elif isinstance(target, list) and last_part.isdigit():
                    target[int(last_part)] = value
                else:
                    target[last_part] = value  # type: ignore[index]
            except (KeyError, IndexError, TypeError, AttributeError) as err:
                raise AttributeError(f"Cannot find config path: {key}") from err

    @staticmethod
    def validate_search_space(space: Dict[str, Any], policy_name: str) -> None:
        """Validate a search space specification before any trial is run.

        Checks that all parameters have the required keys for their type
        (e.g., 'low'/'high' for int/float, 'choices' for categorical).
        Raises a descriptive ValueError on the first invalid parameter so that
        misconfigured JSON files are caught immediately rather than mid-trial.

        Args:
            space: The search space dict mapping parameter names
                to their specification dicts.
            policy_name: Policy name used in error messages for context.

        Raises:
            ValueError: If any parameter specification is missing required keys
                or contains an unrecognised type.
        """
        valid_types = {"float", "int", "categorical"}
        for name, spec in space.items():
            p_type = spec.get("type")
            if p_type not in valid_types:
                raise ValueError(
                    f"[{policy_name}] Parameter '{name}' has unknown type '{p_type}'. "
                    f"Must be one of: {sorted(valid_types)}."
                )
            if p_type in ("float", "int"):
                missing = [k for k in ("low", "high") if k not in spec]
                if missing:
                    raise ValueError(
                        f"[{policy_name}] Parameter '{name}' (type='{p_type}') is missing "
                        f"required key(s): {missing}. "
                        f"Edit the corresponding JSON in hpo/routing/{policy_name}.json."
                    )
                if spec["low"] >= spec["high"]:
                    raise ValueError(
                        f"[{policy_name}] Parameter '{name}': 'low' ({spec['low']}) must be "
                        f"strictly less than 'high' ({spec['high']})."
                    )
            elif p_type == "categorical":
                choices = spec.get("choices")
                if not choices:
                    raise ValueError(
                        f"[{policy_name}] Parameter '{name}' (type='categorical') has an "
                        f"empty or missing 'choices' list."
                    )

    @staticmethod
    def suggest_param(trial: optuna.Trial, name: str, spec: Dict[str, Any]) -> Any:
        """Suggest a parameter value for an Optuna trial based on its specification.

        Handles the hashability requirement for categorical parameters: list
        choices are converted to tuples before sampling and converted back
        afterward, so callers always receive the original type.

        Args:
            trial: Optuna trial object for parameter sampling.
            name: Short name of the parameter as registered with Optuna.
                This must be consistent across all calls for the same parameter
                so that Optuna's internal bookkeeping and study resumption work
                correctly.
            spec: Specification dictionary with keys:
                - type (str): 'float', 'int', or 'categorical'.
                - low (numeric): Lower bound (float/int only).
                - high (numeric): Upper bound (float/int only).
                - step (numeric, optional): Discrete step size.
                - log (bool, optional): Sample on a log scale (default False).
                - choices (list): Candidate values (categorical only).

        Returns:
            Any: The suggested parameter value of the appropriate Python type.

        Raises:
            ValueError: If spec['type'] is not a recognised type string.
        """
        p_type = spec.get("type")
        if p_type == "float":
            return trial.suggest_float(
                name,
                float(spec["low"]),
                float(spec["high"]),
                step=spec.get("step"),
                log=spec.get("log", False),
            )
        elif p_type == "int":
            return trial.suggest_int(
                name,
                int(spec["low"]),
                int(spec["high"]),
                step=int(spec.get("step", 1)),
                log=spec.get("log", False),
            )
        elif p_type == "categorical":
            choices = spec["choices"]
            # Optuna requires hashable choices; convert lists → tuples for sampling.
            hashable_choices = [tuple(c) if isinstance(c, list) else c for c in choices]
            sampled = trial.suggest_categorical(name, hashable_choices)
            # Restore original type for the caller.
            return list(sampled) if isinstance(sampled, tuple) else sampled
        else:
            raise ValueError(
                f"Unknown parameter type '{p_type}' for '{name}'. Expected one of: 'float', 'int', 'categorical'."
            )

    @staticmethod
    def build_sampler(
        method: str, seed: int, search_space: Optional[Dict[str, Any]] = None
    ) -> optuna.samplers.BaseSampler:
        """Construct the Optuna sampler corresponding to the configured method.

        Args:
            method: Sampler identifier. Supported values:
                - 'tpe'      – Tree-structured Parzen Estimator (default).
                - 'random'   – Uniform random search.
                - 'grid'     – Exhaustive grid search (requires finite, bounded space).
                - 'cmaes'    – CMA-ES (best for continuous, correlated ALNS-style spaces).
                - 'nsgaii'   – NSGA-II (required for multi-objective studies).
                - 'botorch'  – Bayesian optimisation with GP surrogate (expensive trials).
            seed: Random seed for reproducibility.
            search_space: Full search space spec; required
                when method='grid' to enumerate the grid.

        Returns:
            optuna.samplers.BaseSampler: Configured sampler instance.

        Raises:
            ValueError: If method='grid' and search_space is None or contains
                non-categorical / unbounded parameters.
            ImportError: If method='botorch' and optuna-integration is not installed.
        """
        method = method.lower()

        if method == "tpe":
            return optuna.samplers.TPESampler(seed=seed)

        if method == "random":
            return optuna.samplers.RandomSampler(seed=seed)

        if method == "cmaes":
            return optuna.samplers.CmaEsSampler(seed=seed)

        if method == "nsgaii":
            return optuna.samplers.NSGAIISampler(seed=seed)

        if method == "grid":
            if search_space is None:
                raise ValueError(
                    "method='grid' requires the search_space to enumerate the grid. "
                    "Pass search_space=hpo_sim.search_space."
                )
            grid = _build_grid_from_search_space(search_space)
            return optuna.samplers.GridSampler(grid)

        if method == "botorch":
            try:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                return BoTorchSampler(
                    seed=seed,
                    n_startup_trials=20,  # Burn-in phase (LHS)
                    n_ei_candidates=24,  # Candidates for acquisition maximization
                    device=device,
                )
            except ImportError as exc:
                raise ImportError(
                    "method='botorch' requires the 'optuna-integration[botorch]' package. "
                    "Install it with: pip install optuna-integration[botorch]"
                ) from exc

        raise ValueError(
            f"Unknown sampler method '{method}'. Supported: 'tpe', 'random', 'grid', 'cmaes', 'nsgaii', 'botorch'."
        )


def _build_grid_from_search_space(search_space: Dict[str, Any]) -> Dict[str, List[Any]]:
    """Convert a search space spec into the dict format required by GridSampler.

    Only categorical parameters and int parameters with an explicit 'choices'
    list are supported. Float parameters and unbounded int ranges cannot be
    enumerated and will raise a ValueError.

    Args:
        search_space: Search space specification.

    Returns:
        Dict[str, List[Any]]: Grid mapping parameter names to candidate lists.

    Raises:
        ValueError: If a non-enumerable parameter is encountered.
    """
    grid: Dict[str, List[Any]] = {}
    for name, spec in search_space.items():
        p_type = spec.get("type")
        if p_type == "categorical":
            grid[name] = list(spec["choices"])
        elif p_type == "int" and "choices" in spec:
            grid[name] = [int(c) for c in spec["choices"]]
        elif p_type == "int" and "step" in spec:
            grid[name] = list(range(int(spec["low"]), int(spec["high"]) + 1, int(spec["step"])))
        else:
            raise ValueError(
                f"Grid search cannot enumerate parameter '{name}' (type='{p_type}'). "
                f"Provide explicit 'choices' or use a different sampler method."
            )
    return grid
