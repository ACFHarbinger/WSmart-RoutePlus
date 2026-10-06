r"""Configuration parameters for the Smart Waste Collection - Two-Commodity Flow (SWC-TCF) policy.

Attributes:
    SWCTCFParams: Dataclass for TCF solver configuration.

Example:
    >>> params = SWCTCFParams(framework="ortools", engine="gurobi")
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict, Optional


@dataclass
class SWCTCFParams:
    """
    Configuration parameters for the SWC-TCF solver.

    Attributes:
        framework: Optimization framework to use ('ortools' or 'pyomo').
        engine: Optimization engine to use ('gurobi', 'scip', 'highs', or 'cplex').
        gurobi_threads: Native Gurobi threads per solve (positive; independent of simulator workers).
        gurobi_soft_mem_limit_gb: Native Gurobi soft memory ceiling in decimal GB.
        time_limit: Time limit for the solver in seconds.
        seed: Random seed for reproducibility.
        formulation: Native Gurobi model. 'paper' (default) is the published SWCR model
            (Ramos et al., 2018, eqs. 6, 8, 11-15, 17-21; collectioncompare ``policy_gurobi``).
            'directed' is the earlier directed two-commodity flow.

        The options below are not part of the published model; each defaults to the
        original algorithm (native Gurobi only).

        depot_inflow: 'equal' is eq. (15), sum y_i0 = Q k; 'le' is the notebook's <=.
        solver_tuning: MIPFocus/heuristics/cut/presolve settings and a 1% MIP gap.
        relax_forced_on_infeasible: re-solve without forced visits when infeasible.
        link_depot_arcs: add x[0, j] <= g[j].
        max_arc_distance_km: drop arcs longer than this (None keeps every arc).
        warm_start: Clarke-Wright MIP starts (forced bins only; every bin with waste);
            the best one is executed if Gurobi ends without a solution (#41).
    """

    framework: str = "ortools"
    engine: str = "gurobi"
    gurobi_threads: int = 2
    gurobi_soft_mem_limit_gb: float = 5.0
    time_limit: float = 60.0
    seed: int = 42
    formulation: str = "paper"
    depot_inflow: str = "equal"
    solver_tuning: bool = False
    relax_forced_on_infeasible: bool = False
    link_depot_arcs: bool = False
    max_arc_distance_km: Optional[float] = None
    warm_start: bool = False

    @classmethod
    def from_config(cls, config: Any) -> SWCTCFParams:
        """Create SWCTCFParams from a configuration object or dictionary.

        Performs explicit type casting for numeric fields to ensure consistency
        with the framework's configuration loading logic.

        Args:
            config (Any): The configuration object or dictionary.

        Returns:
            SWCTCFParams: The initialized parameters.
        """
        if config is None:
            return cls()

        raw_data: Dict[str, Any] = {}
        if isinstance(config, dict):
            raw_data = config
        else:
            for f in fields(cls):
                if hasattr(config, f.name):
                    raw_data[f.name] = getattr(config, f.name)

        kwargs: Dict[str, Any] = {}
        for f in fields(cls):
            val = raw_data.get(f.name, getattr(cls, f.name, f.default))
            if val is not None:
                if f.type is float or f.type == "float":
                    val = float(val)
                elif f.type is int or f.type == "int":
                    val = int(val)
            kwargs[f.name] = val

        return cls(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """Convert Params to a dictionary.

        Returns:
            Dict[str, Any]: Dictionary of parameter values.
        """
        return {f.name: getattr(self, f.name) for f in fields(self)}


# Arcs longer than this are dropped from every SWC-TCF backend. Distance matrices
# are in km (data/network/file.py loads them without rescaling).
MAX_ARC_DISTANCE_KM = 6000.0
