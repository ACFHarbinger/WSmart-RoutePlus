"""
Action for routing policy execution.

This module provides the RoutingAction class, which invokes
routing algorithms (HGS, ALNS, AM, etc.) to construct collection tours.

Attributes:
    RoutingAction: Command for tour construction.

Example:
    >>> # action = RoutingAction()
    >>> # action.execute(context)
"""

import random
import time
from typing import Any, Dict, List, Optional, cast

import numpy as np
from omegaconf import DictConfig, OmegaConf

from logic.src.pipeline.simulations.actions.policy_config import _as_policy_mapping
from logic.src.pipeline.simulations.reporting import current_solver_status, reset_solver_status
from logic.src.pipeline.simulations.daily import resolve_solver_status
from logic.src.policies.routing.base import RoutingAlgorithmFactory, RoutingAlgorithmRegistry

from .base import SimulationAction, bind_context


def _engine_in_name(name: str, registered: Any) -> Any:
    """The longest registered engine that is a ``_``-delimited part of ``name``, or None."""
    lower = name.lower()
    for eng in sorted(registered, key=lambda k: (-len(k), k)):
        if eng == lower or f"_{eng}_" in lower or lower.startswith(f"{eng}_") or lower.endswith(f"_{eng}"):
            return eng
    return None


class RoutingAction(SimulationAction):
    """
    Executes the routing policy to generate collection tours.

    Attributes:
        None
    """

    timed = True
    reads = ("bins", "config", "distance_matrix")
    writes = ("tour", "cost", "solver_status", "extra_output", "metrics_phase")
    finalizes = ()
    requires_final = ()

    def execute(self, context: Dict[str, Any]) -> None:  # noqa: C901
        """
        Execute the selected routing policy.

        Args:
            context: Shared dictionary containing simulation state.
        """

        context = bind_context(context)
        # Ensure all adapters are imported so the registry is fully populated
        RoutingAlgorithmFactory.ensure_registered()

        # 1. GET FULL POLICY NAME (for identification)
        full_policy = str(getattr(context, "full_policy", None) or "")

        # 2. IDENTIFY ROUTING ENGINE FROM CONFIG
        raw_cfg = getattr(context, "config", {})

        # 2b. INJECT SIMULATION SEED INTO POLICY CONFIG
        # CRITICAL: Use policy-specific seed for RNG isolation
        # If policy_seed is available (set by run_day), use it; otherwise fallback to base seed
        policy_seed = getattr(context, "policy_seed", None)
        if policy_seed is not None:
            seed = policy_seed
        else:
            # Fallback to base simulation seed
            sim_cfg = getattr(getattr(context, "cfg", None), "sim", None)
            seed = getattr(sim_cfg, "seed", 42) if sim_cfg else 42

        # Convert to a standard dict to ensure mutability
        if isinstance(raw_cfg, DictConfig):
            raw_cfg = OmegaConf.to_container(raw_cfg, resolve=True)
            context.config = raw_cfg

        raw_cfg = cast(Dict[str, Any], raw_cfg)
        raw_cfg["seed"] = seed

        # RE-SEED GLOBAL STATE BEFORE EXECUTION
        random.seed(seed)
        np.random.seed(seed)
        try:
            import torch
        except ImportError:
            torch = None  # type: ignore[assignment]
        if torch is not None:
            torch.manual_seed(seed)

        flat_cfg = _as_policy_mapping(raw_cfg)

        # Explicit engine/solver from config
        solver_key = flat_cfg.get("policy.type") or flat_cfg.get("policy.solver") or flat_cfg.get("policy.engine")

        # (register every adapter first: the registry otherwise holds only the imported ones)
        RoutingAlgorithmFactory.ensure_registered()
        registered = set(RoutingAlgorithmRegistry.list_routing_algorithms())

        # Next the engine named in the policy name ('lookahead_hvpl_custom_emp' -> 'hvpl'),
        # then a config key naming an engine. The name comes first because hybrids nest
        # their components' sections ('alns' inside HVPL's config) at the top level.
        if not solver_key:
            solver_key = _engine_in_name(full_policy, registered)
        if not solver_key:
            solver_key = next((k for k in sorted(registered, key=lambda k: (-len(k), k)) if k in raw_cfg), None)

        # Resolve compound names (e.g. 'lookahead_lahc_custom_emp' -> 'lahc')
        if solver_key and solver_key not in registered:
            solver_key = _engine_in_name(str(solver_key), registered) or solver_key
        if not solver_key or solver_key not in registered:
            raise ValueError(
                f"Unknown policy '{solver_key or full_policy}' (from '{full_policy}'). "
                f"Registered policies: {sorted(registered)}"
            )

        # A status noted on a previous day must not be copied onto this one.
        reset_solver_status()

        # An empty mandatory set skips the solver only when the problem does not
        # let the solver choose optional visits. That flag comes from ProblemSpec.
        from logic.src.envs.common.expand_pool import current_expand_pool

        mandatory = getattr(context, "mandatory", [])
        if not mandatory and not current_expand_pool(default=True):
            context.tour = [0, 0]
            context.cost = 0.0
            context.extra_output = None
            context.solver_status = "skipped"
            context.metrics_phase = "preliminary"
            return

        try:
            adapter = RoutingAlgorithmFactory.get_adapter(solver_key, config=raw_cfg)

            # --- MULTI-PERIOD ENHANCEMENTS ---
            # 1. Initialize or Retrieve MultiDayContext
            day_idx = getattr(context, "day", 0)
            multi_day_context = getattr(context, "multi_day_context", None)
            if multi_day_context is None:
                try:
                    from logic.src.interfaces.context.multi_day_context import MultiDayContext
                except ImportError:  # builds without multi-period policies drop the module
                    MultiDayContext = None  # type: ignore[assignment,misc]
                if MultiDayContext is not None:
                    multi_day_context = MultiDayContext.initialize(day_index=day_idx)

            # 2. Generate Scenario Tree unless the adapter opts out.
            # Default is to build. Live multi-period policies read the tree, and
            # none of the retained adapters set uses_scenario_tree to False.
            context.multi_day_context = multi_day_context
            if getattr(adapter, "uses_scenario_tree", True) is False:
                context.scenario_tree = None
            else:
                from logic.src.pipeline.simulations.actions.policy_config import policy_horizon
                from logic.src.pipeline.simulations.bins.prediction import ScenarioGenerator

                horizon = policy_horizon(flat_cfg)
                method = flat_cfg.get("policy.scenario_method") or "stochastic"
                generator = ScenarioGenerator(method=method, horizon=horizon, seed=seed)

                # Generate tree using bin stats and truth (if method is oracle)
                bins_state = getattr(context, "bins", None)
                if bins_state is not None:
                    n = len(bins_state.c)
                    bin_stats = {
                        "means": getattr(bins_state, "means", np.zeros(n)),
                        "stds": getattr(bins_state, "std", np.zeros(n)),
                    }
                else:
                    bin_stats = {"means": np.zeros(0), "stds": np.zeros(0)}
                truth_generator = getattr(context, "truth_generator", None)  # For Perfect Oracle mode

                scenario_tree = generator.generate(
                    current_wastes=bins_state.c if bins_state is not None else np.zeros(0),
                    bin_stats=bin_stats,
                    truth_generator=truth_generator,
                )
                context.scenario_tree = scenario_tree

            start_time = time.perf_counter()
            # 3. Unpack adapter results: tour represents global bin IDs
            from logic.src.tracking.logging.modules.policy_viz_emit import PolicyVizStreamSession

            viz_policy = str(
                getattr(context, "display_name", None) or getattr(context, "policy_name", None) or full_policy
            )
            viz_sample = int(getattr(context, "sample_id", 0))
            viz_day = int(getattr(context, "day", 0))
            viz_log = getattr(context, "realtime_log_path", None)
            viz_lock = getattr(context, "lock", None)

            with PolicyVizStreamSession(adapter, viz_policy, viz_sample, viz_day, viz_log, viz_lock):
                results = adapter.execute(**context.as_kwargs())
            tour, _, _, extra_output, updated_multi_day = results
            elapsed_time = time.perf_counter() - start_time

            # --- PRELIMINARY METRICS ---
            # Calculate preliminary KM from the construction phase. Note that final
            # definitive metrics (KM and Profit) are re-computed in CollectAction
            # to account for the entire policy pipeline (mandatory + construction + improvement).
            from logic.src.utils.routing.tours import (
                get_route_cost,
            )

            raw_km = get_route_cost(context.distance_matrix, tour)

            context.tour = tour
            context.cost = raw_km
            context.extra_output = extra_output
            context.time = elapsed_time  # construction only; run_day replaces it with the full policy time
            context.solver_status = resolve_solver_status(current_solver_status(), cast(Optional[List[int]], tour))
            context.metrics_phase = "preliminary"
            # Preserve updated multi-day state for the next simulation day
            context.multi_day_context = updated_multi_day

            # Legacy caching for regular selection
            if "regular" in full_policy:
                context.cached = extra_output

        except Exception as e:
            context.solver_status = resolve_solver_status(current_solver_status(), getattr(context, "tour", None), e)
            if getattr(context, "tour", None) is None:
                context.tour = [0, 0]
            if isinstance(e, ValueError):
                raise ValueError(f"Failed to load policy adapter for '{solver_key}': {e}") from e
            raise
