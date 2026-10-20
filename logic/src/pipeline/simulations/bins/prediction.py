"""
Statistical prediction and scenario generation for waste bins.

This module provides tools for predicting future bin fill levels and
generating scenario trees for stochastic optimization.

Attributes:
    ScenarioTreeNode: Node in a scenario tree.
    ScenarioTree: Tree structure of nodes.
    ScenarioGenerator: Factory for creating scenario trees.

Example:
    >>> # generator = ScenarioGenerator(method="stochastic", horizon=7)
    >>> # tree = generator.generate(current_wastes, bin_stats)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import gamma  # pyrefly: ignore[missing-module-attribute]


@dataclass
class ScenarioTreeNode:
    """A node in the scenario tree representing a state at a specific day.

    Attributes:
        day: Current day index in the horizon.
        wastes: Numpy array of bin fill levels.
        probability: Probability of this state occurring.
        children: List of child nodes representing future states.
        metadata: Additional diagnostic information.
    """

    day: int
    wastes: np.ndarray  # Bin fill levels [0-100]
    probability: float
    children: List["ScenarioTreeNode"] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScenarioTree:
    """A collection of scenarios forming a tree structure.

    Attributes:
        root: The starting state of the tree.
        horizon: Prediction horizon in days.
        num_bins: Number of bins in each state vector.
    """

    root: ScenarioTreeNode
    horizon: int
    num_bins: int

    def get_scenarios_at_day(self, day: int) -> List[ScenarioTreeNode]:
        """Flatten the tree at a specific depth to get a set of scenarios.

        Args:
            day: Depth in the tree to retrieve nodes from.

        Returns:
            List of ScenarioTreeNode objects at the specified depth.
        """
        result = []

        def traverse(node: ScenarioTreeNode):
            """
            Recursively traverse the tree.

            Args:
                node: Current scenario tree node to traverse.
            """
            if node.day == day:
                result.append(node)
            else:
                for child in node.children:
                    traverse(child)

        traverse(self.root)
        return result


class ScenarioGenerator:
    """
    Generates scenario trees for multi-period stochastic optimization.

    Attributes:
        method: The generation method (e.g., "stochastic", "perfect_oracle").
        horizon: Simulation horizon.
        seed: Random seed for stochastic generation.
        distribution: Statistical distribution to use.
        dist_kwargs: Parameters for the distribution.
    """

    def __init__(
        self,
        method: str = "stochastic",
        horizon: int = 7,
        seed: int = 42,
        distribution: str = "mean",
        dist_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initializes the scenario generator.

        Args:
            method: The generation method to use.
            horizon: Simulation horizon.
            seed: Random seed.
            distribution: Statistical distribution to use for waste levels.
            dist_kwargs: Additional arguments for the distribution.
        """
        self.method = method
        self.horizon = horizon
        self.seed = seed
        self.distribution = distribution
        self.dist_kwargs = dist_kwargs or {}
        # Initialize RNG for stochastic paths
        self.rng = np.random.default_rng(seed)
        self._dist_instance = None

        if self.distribution not in (
            "mean",
            "gamma",
            "norm",
            "poisson",
            "compound_poisson_gamma",
            "bernoulli_gamma_mixture",
        ):
            from logic.src.data.distributions import DISTRIBUTION_REGISTRY

            if self.distribution in DISTRIBUTION_REGISTRY:
                dist_class = DISTRIBUTION_REGISTRY[self.distribution]
                self._dist_instance = dist_class(**self.dist_kwargs)
                self._dist_instance.set_sampling_method("sample_array")
            else:
                raise ValueError(f"Unknown scenario distribution: {self.distribution}")

    def generate(
        self,
        current_wastes: np.ndarray,
        bin_stats: Optional[Dict[str, np.ndarray]] = None,
        truth_generator: Optional[Any] = None,
    ) -> ScenarioTree:
        """
        Generate a scenario tree.

        Args:
            current_wastes: Current fill levels for bins.
            bin_stats: Dictionary with 'means' and 'stds' of fill rates.
            truth_generator: Optional object that provides future true values (Oracle).

        Returns:
            ScenarioTree: The generated scenario tree.
        """
        root = ScenarioTreeNode(day=0, wastes=current_wastes.copy(), probability=1.0)

        if self.method == "perfect_oracle" and truth_generator:
            self._generate_oracle_path(root, truth_generator)
        else:
            self._generate_stochastic_tree(root, bin_stats)

        return ScenarioTree(root=root, horizon=self.horizon, num_bins=len(current_wastes))

    def _generate_oracle_path(self, root: ScenarioTreeNode, truth_generator: Any) -> None:
        """Generate a single deterministic path using known future values.

        Args:
            root: Root node to build upon.
            truth_generator: Object providing future ground truth values.
        """
        current = root
        for t in range(1, self.horizon + 1):
            future_wastes = truth_generator.get_future_wastes(t)
            child = ScenarioTreeNode(day=t, wastes=future_wastes, probability=1.0)
            current.children.append(child)
            current = child

    def _generate_stochastic_tree(self, root: ScenarioTreeNode, bin_stats: Optional[Dict[str, np.ndarray]]) -> None:
        """Generate branches based on statistical distributions.

        Args:
            root: Root node to build upon.
            bin_stats: Dictionary containing 'means' and 'stds' of fill rates.
        """
        if bin_stats is None:
            # Fallback: static levels
            current = root
            for t in range(1, self.horizon + 1):
                child = ScenarioTreeNode(day=t, wastes=current.wastes.copy(), probability=1.0)
                current.children.append(child)
                current = child
            return

        means = bin_stats["means"]
        stds = bin_stats.get("stds", np.zeros_like(means))
        # Prevent division by zero mathematically
        safe_means = np.where(means == 0, 1e-9, means)
        safe_stds = np.where(stds == 0, 1e-9, stds)
        safe_vars = safe_stds**2

        current = root
        for t in range(1, self.horizon + 1):
            if self.distribution == "mean":
                step_wastes = means
            elif self._dist_instance is not None:
                # Use custom distribution from registry
                step_wastes = self._dist_instance.sample(means.shape, rng=self.rng)
            elif self.distribution == "gamma":
                # Method of moments fallback for gamma
                k = safe_means**2 / safe_vars
                th = safe_vars / safe_means
                step_wastes = self.rng.gamma(shape=k, scale=th)
            elif self.distribution == "norm":
                step_wastes = self.rng.normal(means, stds)
            elif self.distribution == "poisson":
                step_wastes = self.rng.poisson(means)
            elif self.distribution == "compound_poisson_gamma":
                # Method of Moments mapping matching 'means' and 'vars' assuming exponential jumps (alpha=1)
                lam = 2.0 * safe_means**2 / safe_vars
                th = safe_vars / (2.0 * safe_means)
                from logic.src.data.distributions import DISTRIBUTION_REGISTRY

                dist_class = DISTRIBUTION_REGISTRY["compound_poisson_gamma"]
                inst = dist_class(lam=lam, alpha=1.0, theta=th)
                inst.set_sampling_method("sample_array")
                step_wastes = inst.sample(means.shape, rng=self.rng)
            elif self.distribution == "bernoulli_gamma_mixture":
                # Method of Moments mapping matching 'means' and 'vars'
                v = safe_vars / (safe_means**2)
                p = 2.0 / (v + 2.0)
                alpha = (v + 2.0) / v
                theta = safe_vars / (2.0 * safe_means)
                from logic.src.data.distributions import DISTRIBUTION_REGISTRY

                dist_class = DISTRIBUTION_REGISTRY["bernoulli_gamma_mixture"]
                inst = dist_class(p=p, alpha=alpha, theta=theta)
                inst.set_sampling_method("sample_array")
                step_wastes = inst.sample(means.shape, rng=self.rng)
            else:
                step_wastes = means

            new_wastes = np.clip(current.wastes + step_wastes, 0.0, 100.0)
            child = ScenarioTreeNode(day=t, wastes=new_wastes, probability=1.0)
            current.children.append(child)
            current = child


def predict_days_to_overflow(ui: np.ndarray, vi: np.ndarray, f: np.ndarray, cl: float) -> np.ndarray:
    """
    Internal math for predicting days until a bin overflows.

    Uses the Gamma distribution CDF to estimate the probability of
    reaching 100% capacity within a 31-day window.

    Args:
        ui: Mean fill rate.
        vi: Variance of fill rate.
        f: Current fill level.
        cl: Confidence level (0-1).

    Returns:
        np.ndarray: Predicted days to overflow per bin (clipped at 31).
    """
    n = np.zeros(ui.shape[0]) + 31
    # Loop-invariant guards and headroom, hoisted out of the day loop.
    safe_vi = np.where(vi == 0, 1e-9, vi)
    safe_ui = np.where(ui == 0, 1e-9, ui)
    headroom = 100 - f
    for ii in np.arange(1, 31, 1):
        k = ii * safe_ui**2 / safe_vi
        th = safe_vi / safe_ui
        aux = np.zeros(ui.shape[0]) + 31

        # Calculate CDF
        p = 1 - gamma.cdf(headroom, k, scale=th)

        aux[np.nonzero(p > cl)[0]] = ii
        n = np.minimum(n, aux)
        if np.all(p > cl):
            return n
    return n


def calculate_frequency_and_level(ui: float, vi: float, cf: float) -> Tuple[int, float]:
    """
    Calculates the recommended visit frequency and target overflow level.

    Args:
        ui: Mean daily fill rate.
        vi: Variance of daily fill rate.
        cf: Target confidence level (e.g., 0.9 for 90% service level).

    Returns:
        Tuple[int, float]: (Optimal days between visits, Target level at visit).
    """
    ov = 80.0  # Default fallback
    for n in range(1, 50):
        # Prevent division by zero
        safe_vi = 1e-9 if vi == 0 else vi
        safe_ui = 1e-9 if ui == 0 else ui

        k = n * safe_ui**2 / safe_vi
        th = safe_vi / safe_ui

        if n == 1:
            ov = 100 - gamma.ppf(1 - cf, k, scale=th)

        v = gamma.ppf(1 - cf, k, scale=th)
        if v > 100:
            return n, float(ov)
    return 49, float(ov)
