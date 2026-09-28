r"""Configuration parameters for the TCF → ALNS → BPC → SP-merge pipeline.

The pipeline is controlled by a single quality/speed dial ``alpha ∈ [0, 1]``
that proportionally allocates the total time budget across the four stages:

    alpha = 0.0  →  TCF + tiny ALNS only  (fastest, no BPC)
    alpha = 0.5  →  balanced              (default)
    alpha = 1.0  →  full BPC + large ALNS (highest quality, slowest)

Stage budgets are computed as:
    τ_TCF  = T * max(0.05, 0.15 − 0.10 * α)
    τ_ALNS = T * (0.20  + 0.15 * α)
    τ_BPC  = T * (0.00  + 0.65 * α)
    τ_SP   = min(30, T * 0.05)              # always small

All four values are then renormalized to sum to T.

Attributes:
    ExactGuidedHeuristicParams: Dataclass for pipeline solver configuration.

Example:
    >>> params = ExactGuidedHeuristicParams(alpha=0.5, time_limit=120.0)
    >>> budgets = params.stage_budgets()   # (τ_tcf, τ_alns, τ_bpc, τ_sp)
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict, Optional, Tuple


@dataclass
class ExactGuidedHeuristicParams:
    """Configuration parameters for the TCF → ALNS → BPC → SP-merge pipeline.

    Attributes:
        alpha: Quality/speed dial ∈ [0, 1].
            0.0 = TCF + tiny ALNS only (no BPC stage).
            0.5 = balanced (default).
            1.0 = full BPC + large ALNS (highest quality, slowest).
        time_limit: Total wall-clock budget in seconds.
        seed: Global RNG seed for reproducibility.

        # ── ALNS overrides ─────────────────────────────────────────────────
        alns_max_iterations: Hard iteration cap for ALNS (overrides the
            alpha-derived default when set to a positive integer).
            Default 0 = derive from alpha: max(500, 2000 + 18000 * alpha).
        alns_segment_size: Weight-update segment size (Ropke & Pisinger 2006).
        alns_reaction_factor: Learning rate r for weight updates.
        alns_cooling_rate: SA temperature decay factor per iteration.
        alns_start_temp_control: 'w' parameter — accept a solution 'w*100 %'
            worse than current with probability 0.5 at start temperature.
        alns_sigma_1: Score awarded for a new global-best solution.
        alns_sigma_2: Score awarded for a better-than-current new solution.
        alns_sigma_3: Score awarded for an accepted worse new solution.
        alns_xi: Fraction of n for the removal upper-bound cap.
        alns_min_removal: Minimum nodes removed per destroy step.
        alns_noise_factor: Noise amplitude η for noisy repair operators.
        alns_worst_removal_randomness: Randomness exponent p ≥ 1 for worst removal.
        alns_shaw_randomization: Shaw randomisation factor p_shaw.
        alns_regret_pool: Which regret variants to use
            ('regret2', 'regret234', 'regretAll').
        alns_extended_operators: If True, add string/cluster/neighbor destroy
            operators (3 → 6 operators).
        alns_profit_aware_operators: If True, use profit-aware operator variants.
        alns_vrpp: If True, allow ALNS repair operators to insert nodes from the
            full candidate pool (not only the just-removed set).
        alns_engine: Which ALNS backend to use ('custom', 'package', 'ortools').

        # ── BPC overrides ──────────────────────────────────────────────────
        bpc_ng_size_min: Minimum ng-neighborhood size (used when alpha=0).
        bpc_ng_size_max: Maximum ng-neighborhood size (used when alpha=1).
            Effective ng_size = bpc_ng_size_min + int(alpha *
                (bpc_ng_size_max − bpc_ng_size_min)).
        bpc_max_bb_nodes_min: Minimum B&B node cap (alpha=0).
        bpc_max_bb_nodes_max: Maximum B&B node cap (alpha=1).
        bpc_cutting_planes: Cut family for BPC ('rcc', 'saturated_arc_lci',
            'all', etc.).
        bpc_branching_strategy: Branching rule ('divergence', 'ryan_foster',
            'edge').
        skip_bpc: Force-skip the BPC stage regardless of alpha or time budget.
            Useful for very large instances or pure-heuristic runs.

        # ── SP-merge overrides ─────────────────────────────────────────────
        sp_pool_cap: Maximum number of routes kept in the SP-merge MIP.
            If the collected pool exceeds this cap, the top-half by profit and
            a random sample of the rest are retained.
        sp_mip_gap: Relative gap at which the SP MIP is considered solved.
    """

    # ── Quality / speed dial ───────────────────────────────────────────────
    alpha: float = 0.5
    time_limit: float = 120.0
    seed: Optional[int] = None

    # ── ALNS ──────────────────────────────────────────────────────────────
    alns_max_iterations: int = 0  # 0 = derived from alpha
    alns_segment_size: int = 100
    alns_reaction_factor: float = 0.1
    alns_cooling_rate: float = 0.995
    alns_start_temp_control: float = 0.05
    alns_sigma_1: float = 33.0
    alns_sigma_2: float = 9.0
    alns_sigma_3: float = 13.0
    alns_xi: float = 0.4
    alns_min_removal: int = 4
    alns_noise_factor: float = 0.025
    alns_worst_removal_randomness: float = 3.0
    alns_shaw_randomization: float = 6.0
    alns_regret_pool: str = "regret234"
    alns_extended_operators: bool = False
    alns_profit_aware_operators: bool = True
    alns_vrpp: bool = True
    alns_engine: str = "custom"

    # ── BPC ───────────────────────────────────────────────────────────────
    bpc_ng_size_min: int = 8
    bpc_ng_size_max: int = 16
    bpc_max_bb_nodes_min: int = 200
    bpc_max_bb_nodes_max: int = 1000
    bpc_cutting_planes: str = "rcc"
    bpc_branching_strategy: str = "divergence"
    skip_bpc: bool = False

    # ── SP merge ──────────────────────────────────────────────────────────
    sp_pool_cap: int = 50_000
    sp_mip_gap: float = 1e-4

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------

    def stage_budgets(self) -> Tuple[float, float, float, float]:
        """Compute per-stage time budgets from alpha and total time_limit.

        Returns:
            (tau_tcf, tau_alns, tau_bpc, tau_sp) — guaranteed to sum to
            self.time_limit.
        """
        T = self.time_limit
        a = max(0.0, min(1.0, self.alpha))  # clamp to [0, 1]
        tau_tcf = T * max(0.05, 0.15 - 0.10 * a)
        tau_alns = T * (0.20 + 0.15 * a)
        tau_bpc = T * (0.00 + 0.65 * a)
        tau_sp = min(30.0, T * 0.05)
        total = tau_tcf + tau_alns + tau_bpc + tau_sp
        scale = T / total if total > 0 else 1.0
        return (
            tau_tcf * scale,
            tau_alns * scale,
            tau_bpc * scale,
            tau_sp * scale,
        )

    def alns_iterations(self) -> int:
        """Effective ALNS iteration count (respects explicit override).

        Args:
            None

        Returns:
            Effective ALNS iteration count.
        """
        if self.alns_max_iterations > 0:
            return self.alns_max_iterations
        return max(500, int(2_000 + 18_000 * max(0.0, min(1.0, self.alpha))))

    def bpc_ng_size(self) -> int:
        """Effective ng-neighborhood size, linearly interpolated by alpha.

        Args:
            None

        Returns:
            Effective ng-neighborhood size.
        """
        a = max(0.0, min(1.0, self.alpha))
        return self.bpc_ng_size_min + int(a * (self.bpc_ng_size_max - self.bpc_ng_size_min))

    def bpc_max_bb_nodes(self) -> int:
        """Effective B&B node cap, linearly interpolated by alpha.

        Args:
            None

        Returns:
            Effective B&B node cap.
        """
        a = max(0.0, min(1.0, self.alpha))
        return self.bpc_max_bb_nodes_min + int(a * (self.bpc_max_bb_nodes_max - self.bpc_max_bb_nodes_min))

    def as_alns_values_dict(self) -> Dict[str, Any]:
        """Build the ``values`` dict expected by the existing ALNS dispatcher.

        The ALNS dispatcher (``run_alns``) accepts a flat dict whose keys mirror
        ``ALNSParams`` field names.  This method projects the pipeline params
        onto that dict so the existing ALNS code can be called without changes.

        Args:
            None

        Returns:
            Dict with ALNS parameters.
        """
        return {
            "engine": self.alns_engine,
            "time_limit": 0.0,  # managed by pipeline
            "max_iterations": self.alns_iterations(),
            "start_temp": 0.0,  # dynamic calculation
            "cooling_rate": self.alns_cooling_rate,
            "reaction_factor": self.alns_reaction_factor,
            "min_removal": self.alns_min_removal,
            "start_temp_control": self.alns_start_temp_control,
            "xi": self.alns_xi,
            "segment_size": self.alns_segment_size,
            "noise_factor": self.alns_noise_factor,
            "worst_removal_randomness": self.alns_worst_removal_randomness,
            "shaw_randomization": self.alns_shaw_randomization,
            "max_removal_cap": 100,
            "regret_pool": self.alns_regret_pool,
            "sigma_1": self.alns_sigma_1,
            "sigma_2": self.alns_sigma_2,
            "sigma_3": self.alns_sigma_3,
            "vrpp": self.alns_vrpp,
            "profit_aware_operators": self.alns_profit_aware_operators,
            "extended_operators": self.alns_extended_operators,
            "seed": self.seed,
        }

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: Any) -> "ExactGuidedHeuristicParams":
        """Create ExactGuidedHeuristicParams from a dict or config object.

        Silently ignores unknown keys so the YAML can contain extra fields
        without causing hard failures.

        Args:
            config: dict or any object whose attributes mirror ExactGuidedHeuristicParams
                field names.

        Returns:
            ExactGuidedHeuristicParams: Initialized parameter object.
        """
        if config is None:
            return cls()

        valid_fields = {f.name for f in fields(cls)}
        raw: Dict[str, Any] = {}

        if isinstance(config, dict):
            raw = {k: v for k, v in config.items() if k in valid_fields}
        else:
            for f in fields(cls):
                if hasattr(config, f.name):
                    raw[f.name] = getattr(config, f.name)

        return cls(**raw)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize all fields to a plain dict.

        Returns:
            Dict[str, Any]: Mapping of field names to their current values.
        """
        return {f.name: getattr(self, f.name) for f in fields(self)}
