"""Runtime parameters for the Exact Guided Heuristic (TCF - ALNS - BPC - SP) pipeline.

The config dataclass ``ExactGuidedHeuristicConfig`` in
``logic/src/configs/policies/egh.py`` is the single field authority.
``ExactGuidedHeuristicParams`` is a thin subclass that adds the derived
runtime members (``stage_budgets``, ``alns_iterations``, ``bpc_ng_size``,
``bpc_max_bb_nodes``, ``as_alns_values_dict``) plus the
``from_config``/``to_dict`` helpers used by the dispatcher and the policy
adapter; it declares no fields of its own.

Example:
    >>> params = ExactGuidedHeuristicParams(alpha=0.5, time_limit=120.0)
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict, Tuple

from logic.src.configs.policies import ExactGuidedHeuristicConfig


@dataclass
class ExactGuidedHeuristicParams(ExactGuidedHeuristicConfig):
    """Runtime parameters for the TCF - ALNS - BPC - SP-merge pipeline.

    Thin subclass of :class:`ExactGuidedHeuristicConfig`: every field and
    default is inherited from the config dataclass.
    """

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
