"""
Parameter dataclasses for the Concurrent Adaptive Lagrangian Matheuristic (CALM).

This module defines all configuration structures used by CALM, providing
a single point of control for tuning and ablating its various components.

Attributes:
    LookaheadParams: Controls lookahead prize valuation.
    LagrangianParams: Controls the Lagrangian relaxation of x^K = x^R.
    DualBoundParams: Dual-bound tracking strategy.
    BanditParams: LinUCB contextual bandit for engine + cut-strategy selection.
    RegretParams: Adaptive regret-based preprocessing (soft -> hard escalation).
    CALMParams: Top-level parameters; composes all subsystems.

Example:
    >>> CALMParams(
    ...     lookahead=LookaheadParams(
    ...         horizon=7,
    ...         n_scenarios=1,
    ...     ),
    ...     lagrangian=LagrangianParams(
    ...         max_outer_iterations=12,
    ...         stagnation_patience=3,
    ...     ),
    ...     dual_bound=DualBoundParams(
    ...         strategy="ema",
    ...         ema_alpha=0.3,
    ...         ema_quality_threshold=1.05,
    ...     ),
    ...     bandit=BanditParams(
    ...         n_arms=2,
    ...         n_features=10,
    ...         delta=0.3,
    ...         nu=2.0,
    ...         lambda_reg=1.0,
    ...     ),
    ...     regret=RegretParams(
    ...         adaptive_budget=0.3,
    ...         soft_phase_boost=5.0,
    ...     ),
    ... )
"""

from __future__ import annotations

from dataclasses import MISSING, dataclass, field, fields
from typing import Any, Dict, List, Optional

from logic.src.configs.policies.routing.matheuristics.tpks import TpksConfig

# ---------------------------------------------------------------------------
# Lookahead (scenario tree + per-bin DP)
# ---------------------------------------------------------------------------


@dataclass
class LookaheadParams:
    """
    Controls lookahead prize valuation.

    Attributes:
        horizon: Horizon length.
        scenario_method: Method for scenario generation.
        scenario_distribution: Distribution for scenario generation.
        scenario_dist_kwargs: Keyword arguments for scenario distribution.
        n_scenarios: Number of scenarios.
        seed: Random seed.
        capacity_cap: Capacity cap.
        volume: Volume.
        density: Density.
        revenue_per_kg: Revenue per kg.
    """

    # Scenario-tree inputs (forwarded to ScenarioGenerator)
    horizon: int = 7
    scenario_method: str = "stochastic"  # "stochastic" | "perfect_oracle"
    scenario_distribution: str = "gamma"
    scenario_dist_kwargs: Dict[str, Any] = field(default_factory=dict)
    # Monte-Carlo: how many independent scenario paths to average V[i,t] over.
    # Set to 1 for the "mean" distribution; >=8 recommended for "gamma".
    n_scenarios: int = 8
    seed: int = 42

    # Overflow-saturation cap (matches MAX_CAPACITY_PERCENT in the simulator).
    capacity_cap: float = 100.0

    # Revenue conversion:  collected_kg = (fill% / 100) * volume * density.
    # The policy reads volume/density/revenue_per_kg from the area params
    # loaded upstream, but they may be overridden here for ablations.
    volume: Optional[float] = None
    density: Optional[float] = None
    revenue_per_kg: Optional[float] = None


# ---------------------------------------------------------------------------
# Lagrangian coordination layer
# ---------------------------------------------------------------------------


@dataclass
class LagrangianParams:
    """
    Controls the Lagrangian relaxation of x^K = x^R.

    Attributes:
        max_outer_iterations: Maximum number of outer iterations.
        stagnation_patience: Number of iterations to wait before escalating to hard phase.
        dual_bound_tolerance: Tolerance for dual bound.
        asynchronous_updates: Whether to use asynchronous updates.
        polyak_mu_default: Default step size for Polyak rule.
        polyak_mu_floor: Floor value for step size.
        polyak_mu_ceil: Ceiling value for step size.
        lambda_min: Minimum value for dual variables.
        lambda_max: Maximum value for dual variables.
        gamma_init: Initial value for gamma.
        gamma_decay: Decay rate for gamma.
    """

    # Outer-loop control
    max_outer_iterations: int = 12
    stagnation_patience: int = 3  # stop if primal unchanged for N iters
    dual_bound_tolerance: float = 1e-3

    # Async vs. sync multiplier updates.  True => option 1b (per-period,
    # on RS-worker completion).  False => synchronous at end of outer iter.
    asynchronous_updates: bool = True

    # Stepsize (Polyak rule) - used only when the dual_bound strategy is "ema".
    # The bundle strategy ignores these since it solves a QP for the step.
    polyak_mu_default: float = 0.15
    polyak_mu_floor: float = 0.01
    polyak_mu_ceil: float = 0.6

    # Multiplier clipping (avoids runaway on degenerate instances).
    lambda_min: float = -100.0
    lambda_max: float = 100.0

    # How strongly the Master trusts insertion-cost oracle vs raw prizes.
    # Annealed down as real dual information accumulates.
    gamma_init: float = 1.0
    gamma_decay: float = 0.85


# ---------------------------------------------------------------------------
# Dual-bound tracker
# ---------------------------------------------------------------------------


@dataclass
class DualBoundParams:
    """
    Dual-bound tracking strategy.

    strategy = "ema"    ->  best-so-far Lagrangian value, with a per-node EMA of
                            dual contributions that is only updated when the
                            routing subproblem returns a tour within
                            `ema_quality_threshold` of the period incumbent.
                            Prevents the lookahead from being poisoned by
                            temporary suboptimal routes.

    strategy = "bundle" ->  proximal bundle method.  Maintains a bundle of
                            subgradients and solves a small QP at each update
                            to choose a stabilising step.  Stronger dual,
                            more expensive.

    Attributes:
        strategy: Strategy to use for dual-bound tracking.
        ema_alpha: Smoothing factor for EMA variant.
        ema_quality_threshold: Threshold for accepting new subgradients in EMA.
        bundle_size: Maximum number of subgradients to store in bundle.
        bundle_proximal_weight: Weight for proximal term in bundle optimization.
        bundle_descent_threshold: Threshold for serious step in bundle method.
        bundle_weight_increase: Factor to increase proximal weight on null step.
        bundle_weight_decrease: Factor to decrease proximal weight on serious step.
    """

    strategy: str = "ema"  # "ema" | "bundle"

    # --- EMA variant ---
    ema_alpha: float = 0.3  # smoothing (higher = more reactive)
    # A new subgradient is accepted into the EMA only if the RS-returned tour
    # cost is within `ema_quality_threshold` (fractional) of the period's
    # incumbent.  1.05 => accept tours up to 5% worse than incumbent.
    ema_quality_threshold: float = 1.05

    # --- Bundle variant ---
    bundle_size: int = 50  # max bundle entries (oldest evicted FIFO)
    bundle_proximal_weight: float = 1.0  # u in min g(lambda) + u/2 ||lambda - lambda_hat||^2
    bundle_descent_threshold: float = 0.1  # m_L in standard bundle terminology
    bundle_weight_increase: float = 2.0  # u <- u * this on null step
    bundle_weight_decrease: float = 0.5  # u <- u * this on serious step


# ---------------------------------------------------------------------------
# RL: Contextual bandit for algorithmic-choice selection
# ---------------------------------------------------------------------------


@dataclass
class BanditParams:
    """
    LinUCB contextual bandit for engine + cut-strategy selection.

    Attributes:
        enabled: Whether to enable the bandit.
        alpha: Exploration width (UCB coefficient).
        ridge_lambda: LinUCB regulariser.
        feature_dim: Context vector dimension.
        engines: List of engines to choose from.
        cut_strategies: List of cut strategies to choose from.
        reward_scale: Multiplicative scale for reward shaping.
        reward_clip: Clip value for reward shaping.
    """

    enabled: bool = True
    alpha: float = 1.0  # exploration width (UCB coefficient)
    ridge_lambda: float = 1.0  # LinUCB regulariser
    feature_dim: int = 8  # context vector dimension (see coordinator)

    # Action arms.  Each arm is a (engine, cut_strategy) pair.
    engines: List[str] = field(default_factory=lambda: ["tpks", "tpks_warm", "greedy"])
    cut_strategies: List[str] = field(default_factory=lambda: ["plain", "lifted", "pareto"])

    # Reward shaping: primal improvement per second of RS + Master wallclock.
    reward_scale: float = 1.0  # multiplicative scale (for numerical conditioning)
    reward_clip: float = 10.0  # clip abs reward after scaling


# ---------------------------------------------------------------------------
# Regret-based preprocessing
# ---------------------------------------------------------------------------


@dataclass
class RegretParams:
    """
    Adaptive regret-based preprocessing (soft -> hard escalation).

    Attributes:
        enabled: Whether to enable regret-based preprocessing.
        soft_bias_coefficient: Coefficient for biasing objective of high-regret bins.
        escalation_patience: Number of iterations to wait before escalating to hard phase.
        hard_fix_top_fraction: Fraction of top-regret bins to fix in hard phase.
        hard_fix_max_periods: Maximum number of periods to fix top-regret bins.
    """

    enabled: bool = True

    # Soft phase: bias objective coefficients of high-regret bins upward.
    soft_bias_coefficient: float = 0.5  # added to V[i,t] for top-decile regret bins

    # Escalation trigger: primal stagnation (no improvement) for this many
    # outer iterations flips soft->hard.
    escalation_patience: int = 2

    # Hard phase: force top-decile regret bins to visit in periods 1-2.
    hard_fix_top_fraction: float = 0.10  # top 10% by cumulative early regret
    hard_fix_max_periods: int = 2  # force visit within [0, this)


# ---------------------------------------------------------------------------
# Top-level params
# ---------------------------------------------------------------------------


@dataclass
class CALMParams:
    """
    Top-level parameters; composes all subsystems.

    Attributes:
        lookahead: Lookahead parameters.
        lagrangian: Lagrangian parameters.
        dual_bound: Dual bound parameters.
        bandit: Bandit parameters.
        regret: Regret parameters.
        tpks: TPKS parameters.
        time_limit: Global time budget (seconds).
        seed: Random seed.
        verbose: Whether to print verbose output.
        stockout_penalty: Overflow (stockout) penalty.
    """

    lookahead: LookaheadParams = field(default_factory=LookaheadParams)
    lagrangian: LagrangianParams = field(default_factory=LagrangianParams)
    dual_bound: DualBoundParams = field(default_factory=DualBoundParams)
    bandit: BanditParams = field(default_factory=BanditParams)
    regret: RegretParams = field(default_factory=RegretParams)
    tpks: TpksConfig = field(default_factory=TpksConfig)

    # Global time budget (seconds).  Distributed across outer iterations.
    time_limit: float = 600.0
    seed: int = 42
    verbose: bool = False

    # Overflow (stockout) penalty, forwarded from BaseMultiPeriodRoutingPolicy.
    stockout_penalty: float = 500.0

    @classmethod
    def from_config(cls, config: Any) -> "CALMParams":
        """
        Build params from an OmegaConf / dataclass-like config object.

        Nested subsystems are flattened using the convention
        ``config.<subsystem>.<field>`` (e.g. ``config.lookahead.horizon``).
        Missing values fall back to dataclass defaults.

        Args:
            config: The configuration object.

        Returns:
            CALMParams: The parameters as a CALMParams object.
        """

        def _hydrate(sub_cls, sub_cfg: Any):
            kwargs: Dict[str, Any] = {}
            for f in fields(sub_cls):
                val = getattr(sub_cfg, f.name, MISSING)
                if val is MISSING or str(val) == "MISSING":
                    continue
                kwargs[f.name] = val
            return sub_cls(**kwargs)

        top_kwargs: Dict[str, Any] = {}
        for f in fields(cls):
            sub_cfg = getattr(config, f.name, None)
            if sub_cfg is None:
                continue
            if f.type in (
                "LookaheadParams",
                "LagrangianParams",
                "DualBoundParams",
                "BanditParams",
                "RegretParams",
                "TpksConfig",
            ):
                top_kwargs[f.name] = _hydrate(f.default_factory, sub_cfg)  # type: ignore[arg-type]
            else:
                val = getattr(config, f.name, MISSING)
                if val is MISSING or str(val) == "MISSING":
                    continue
                top_kwargs[f.name] = val
        return cls(**top_kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the parameters to a dictionary.

        Returns:
            Dict[str, Any]: The parameters as a dictionary.
        """
        out: Dict[str, Any] = {}
        for f in fields(self):
            val = getattr(self, f.name)
            if hasattr(val, "__dataclass_fields__"):
                out[f.name] = {g.name: getattr(val, g.name) for g in fields(val)}
            else:
                out[f.name] = val
        return out
