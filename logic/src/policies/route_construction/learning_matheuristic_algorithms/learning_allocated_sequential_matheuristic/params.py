"""Runtime parameters for the Learning Allocated Sequential Matheuristic (LASM).

The config dataclass ``LASMPipelineConfig`` in
``logic/src/configs/policies/lasm.py`` is the single field authority.
``LASMPipelineParams`` is a thin subclass: it inherits every field and
re-declares only ``lbbd_cut_families`` and ``rl_state_features`` whose
constructor default is ``None``; the ``__post_init__`` below then
materializes the pipeline's built-in lists exactly as before, so
constructed instances are iterable exactly as before. The derived runtime
members (``stage_budgets``, ``alns_iterations``, ``bpc_ng_size``,
``bpc_max_bb_nodes``, ``as_alns_values_dict``) and the
``from_config``/``to_dict`` helpers are kept verbatim from the original
class.

Example:
    >>> params = LASMPipelineParams(alpha=0.5, time_limit=120.0)
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict, List, Optional, Tuple

from logic.src.configs.policies import LASMPipelineConfig


@dataclass
class LASMPipelineParams(LASMPipelineConfig):
    """Runtime parameters for the LASM pipeline.

    Thin subclass of :class:`LASMPipelineConfig`. The two re-declared
    fields keep the historical ``None`` constructor defaults; the
    ``__post_init__`` below materializes the built-in family lists, so
    constructed instances are iterable exactly as before.
    """

    lbbd_cut_families: Optional[List[str]] = None
    rl_state_features: Optional[List[str]] = None

    def __post_init__(self) -> None:
        """Post-initialization hook.

        Sets default values for optional parameters.

        Args:
            self: The configuration object.

        Returns:
            None
        """
        if self.lbbd_cut_families is None:
            self.lbbd_cut_families = ["nogood", "optimality", "pareto"]
        if self.rl_state_features is None:
            self.rl_state_features = [
                "n_nodes",
                "fill_mean",
                "fill_std",
                "mandatory_ratio",
                "lp_gap",
                "pool_size",
                "time_remaining",
                "alpha",
            ]

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------

    def stage_budgets(
        self,
        budget_override: Optional[Dict[str, float]] = None,
    ) -> Tuple[float, float, float, float, float]:
        """Compute per-stage time budgets.

        When the RL controller provides a ``budget_override`` dict it replaces
        the alpha-derived fractions.  The five values always sum to time_limit.

        Args:
            budget_override: Optional dict with keys 'lbbd', 'alns', 'bpc',
                'sp' giving explicit fractions ∈ (0, 1).  Missing keys fall
                back to alpha-derived defaults.

        Returns:
            (tau_lbbd, tau_alns, tau_bpc, tau_rl_overhead, tau_sp)
        """
        T = self.time_limit
        a = max(0.0, min(1.0, self.alpha))

        # RL overhead is always small (bookkeeping, not solver time)
        tau_rl_overhead = min(2.0, T * 0.01)

        if budget_override:
            tau_lbbd = T * budget_override.get("lbbd", max(0.05, 0.18 - 0.08 * a))
            tau_alns = T * budget_override.get("alns", 0.20 + 0.15 * a)
            tau_bpc = T * budget_override.get("bpc", 0.00 + 0.60 * a)
            tau_sp = T * budget_override.get("sp", 0.05)
        else:
            tau_lbbd = T * max(0.05, 0.18 - 0.08 * a)
            tau_alns = T * (0.20 + 0.15 * a)
            tau_bpc = T * (0.00 + 0.60 * a)
            tau_sp = min(30.0, T * 0.05)

        total = tau_lbbd + tau_alns + tau_bpc + tau_rl_overhead + tau_sp
        s = T / total if total > 0 else 1.0
        return (
            tau_lbbd * s,
            tau_alns * s,
            tau_bpc * s,
            tau_rl_overhead * s,
            tau_sp * s,
        )

    def alns_iterations(self) -> int:
        """Effective ALNS iteration count.

        Returns:
            int: The number of ALNS iterations.
        """
        if self.alns_max_iterations > 0:
            return self.alns_max_iterations
        return max(500, int(2_000 + 18_000 * max(0.0, min(1.0, self.alpha))))

    def bpc_ng_size(self) -> int:
        """Compute B&B node generation pool size.

        Returns:
            int: The B&B node generation pool size.
        """
        a = max(0.0, min(1.0, self.alpha))
        return self.bpc_ng_size_min + int(a * (self.bpc_ng_size_max - self.bpc_ng_size_min))

    def bpc_max_bb_nodes(self) -> int:
        """Compute B&B maximum number of nodes.

        Returns:
            int: The B&B maximum number of nodes.
        """
        a = max(0.0, min(1.0, self.alpha))
        return self.bpc_max_bb_nodes_min + int(a * (self.bpc_max_bb_nodes_max - self.bpc_max_bb_nodes_min))

    def as_alns_values_dict(self) -> Dict[str, Any]:
        """Build the flat dict expected by the existing ALNSParams.from_config.

        Returns:
            Dict[str, Any]: The dictionary representation of the ALNS parameters.
        """
        return {
            "engine": self.alns_engine,
            "time_limit": 0.0,
            "max_iterations": self.alns_iterations(),
            "start_temp": 0.0,
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
    # Serialisation
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: Any) -> "LASMPipelineParams":
        """Construct from a dict or attribute-bearing object.

        Args:
            config: The configuration object.

        Returns:
            LASMPipelineParams: The configuration object.
        """
        if config is None:
            return cls()
        valid = {f.name for f in fields(cls)}
        raw: Dict[str, Any] = {}
        if isinstance(config, dict):
            raw = {k: v for k, v in config.items() if k in valid}
        else:
            for f in fields(cls):
                if hasattr(config, f.name):
                    raw[f.name] = getattr(config, f.name)
        return cls(**raw)

    def to_dict(self) -> Dict[str, Any]:
        """Return a dictionary representation of the parameters.

        Returns:
            Dict[str, Any]: The dictionary representation.
        """
        return {f.name: getattr(self, f.name) for f in fields(self)}
