"""
Action for routing policy execution.

This module provides the RouteConstructionAction class, which invokes
routing algorithms (HGS, ALNS, AM, etc.) to construct collection tours.

Attributes:
    RouteConstructionAction: Command for tour construction.

Example:
    >>> # action = RouteConstructionAction()
    >>> # action.execute(context)
"""

import random
import time
from typing import Any, Dict

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from logic.src.policies.route_construction.base import RouteConstructorFactory, RouteConstructorRegistry

from .base import SimulationAction, _flatten_config


class RouteConstructionAction(SimulationAction):
    """
    Executes the routing policy to generate collection tours.

    Attributes:
        None
    """

    def execute(self, context: Dict[str, Any]) -> None:  # noqa: C901
        """
        Execute the selected routing policy.

        Args:
            context: Shared dictionary containing simulation state.
        """
        # Ensure all adapters are imported so the registry is fully populated
        RouteConstructorFactory.ensure_registered()

        # 1. GET FULL POLICY NAME (for identification)
        full_policy = str(context.get("full_policy") or "")

        # 2. IDENTIFY ROUTING ENGINE FROM CONFIG
        raw_cfg = context.get("config", {})

        # 2b. INJECT SIMULATION SEED INTO POLICY CONFIG
        # CRITICAL: Use policy-specific seed for RNG isolation
        # If policy_seed is available (set by run_day), use it; otherwise fallback to base seed
        policy_seed = context.get("policy_seed")
        if policy_seed is not None:
            seed = policy_seed
        else:
            # Fallback to base simulation seed
            sim_cfg = getattr(context.get("cfg"), "sim", None)
            seed = getattr(sim_cfg, "seed", 42) if sim_cfg else 42

        # Convert to a standard dict to ensure mutability
        if isinstance(raw_cfg, DictConfig):
            raw_cfg = OmegaConf.to_container(raw_cfg, resolve=True)
            context["config"] = raw_cfg

        raw_cfg["seed"] = seed

        # RE-SEED GLOBAL STATE BEFORE EXECUTION
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        flat_cfg = _flatten_config(raw_cfg)

        # Explicit engine/solver from config
        solver_key = flat_cfg.get("policy.type") or flat_cfg.get("policy.solver") or flat_cfg.get("policy.engine")

        # Fallback: identify solver from keys in raw config matching registry
        # (register every adapter first: the registry otherwise holds only the imported ones)
        RouteConstructorFactory.ensure_registered()
        if not solver_key:
            registered = set(RouteConstructorRegistry.list_route_constructors())
            for key in registered:
                if key in raw_cfg:
                    solver_key = key
                    break

        # Last resort: use full_policy string
        if not solver_key:
            solver_key = full_policy

        # Resolve compound names (e.g. 'lookahead_lahc_custom_emp' -> 'lahc')
        registered = set(RouteConstructorRegistry.list_route_constructors())
        if solver_key and solver_key not in registered:
            lower = str(solver_key).lower()
            # Try to extract a registered engine from the compound name
            resolved = False
            for eng in sorted(registered, key=len, reverse=True):
                # Match with word boundaries (underscore or string ends)
                if f"_{eng}_" in lower or lower.startswith(f"{eng}_") or lower.endswith(f"_{eng}") or eng == lower:
                    solver_key = eng
                    resolved = True
                    break
            if not resolved:
                raise ValueError(
                    f"Unknown policy '{solver_key}' (from '{full_policy}'). Registered policies: {sorted(registered)}"
                )

        # If no nodes to collect, skip policy and return to depot
        # VRPP policies (vrpp: true) let the solver decide which nodes to visit — don't skip them
        mandatory = context.get("mandatory", [])
        if not mandatory and not bool(flat_cfg.get("vrpp", False)):
            context["tour"] = [0, 0]
            context["cost"] = 0.0
            context["extra_output"] = None
            return

        try:
            adapter = RouteConstructorFactory.get_adapter(solver_key, config=raw_cfg)

            # --- MULTI-PERIOD ENHANCEMENTS ---
            # 1. Initialize or Retrieve MultiDayContext
            day_idx = context.get("day", 0)
            multi_day_context = context.get("multi_day_context")
            if multi_day_context is None:
                from logic.src.interfaces.context.multi_day_context import MultiDayContext

                multi_day_context = MultiDayContext.initialize(day_index=day_idx)

            # 2. Generate Scenario Tree (if configured or requested by policy)
            # Check if policy inherits from MultiPeriod base or if specifically requested
            from logic.src.pipeline.simulations.bins.prediction import ScenarioGenerator

            horizon = flat_cfg.get("policy.horizon") or flat_cfg.get("policy.num_days") or 7
            method = flat_cfg.get("policy.scenario_method") or "stochastic"
            generator = ScenarioGenerator(method=method, horizon=horizon, seed=seed)

            # Generate tree using bin stats and truth (if method is oracle)
            bins_state = context.get("bins")
            if bins_state is not None:
                n = len(bins_state.c)
                bin_stats = {
                    "means": getattr(bins_state, "means", np.zeros(n)),
                    "stds": getattr(bins_state, "std", np.zeros(n)),
                }
            else:
                bin_stats = {"means": np.zeros(0), "stds": np.zeros(0)}
            truth_generator = context.get("truth_generator")  # For Perfect Oracle mode

            scenario_tree = generator.generate(
                current_wastes=bins_state.c if bins_state is not None else np.zeros(0),
                bin_stats=bin_stats,
                truth_generator=truth_generator,
            )
            # Inject into context for policy consumption
            context["scenario_tree"] = scenario_tree
            context["multi_day_context"] = multi_day_context

            start_time = time.perf_counter()
            # 3. Unpack adapter results: tour represents global bin IDs
            from logic.src.tracking.logging.modules.policy_viz_emit import PolicyVizStreamSession

            viz_policy = str(context.get("display_name") or context.get("policy_name") or full_policy)
            viz_sample = int(context.get("sample_id", 0))
            viz_day = int(context.get("day", 0))
            viz_log = context.get("realtime_log_path")
            viz_lock = context.get("lock")

            with PolicyVizStreamSession(
                adapter, viz_policy, viz_sample, viz_day, viz_log, viz_lock
            ):
                results = adapter.execute(**context)
            tour, _, _, extra_output, updated_multi_day = results
            elapsed_time = time.perf_counter() - start_time

            # --- PRELIMINARY METRICS ---
            # Calculate preliminary KM from the construction phase. Note that final
            # definitive metrics (KM and Profit) are re-computed in CollectAction
            # to account for the entire policy pipeline (mandatory + construction + improvement).
            from logic.src.policies.route_construction.other_algorithms.travelling_salesman_problem.tsp import (
                get_route_cost,
            )

            raw_km = get_route_cost(context["distance_matrix"], tour)

            context["tour"] = tour
            context["cost"] = raw_km
            context["extra_output"] = extra_output
            context["time"] = elapsed_time  # construction only; run_day replaces it with the full policy time
            # Preserve updated multi-day state for the next simulation day
            context["multi_day_context"] = updated_multi_day

            # Legacy caching for regular selection
            if "regular" in full_policy:
                context["cached"] = extra_output

        except ValueError as e:
            raise ValueError(f"Failed to load policy adapter for '{solver_key}': {e}") from e
