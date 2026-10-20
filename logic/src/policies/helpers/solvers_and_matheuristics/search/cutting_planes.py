"""Cutting plane separation engines for Branch-and-Price-and-Cut algorithms.

Provides modular separation algorithms for VRPP-specific valid inequalities:
Provides modular separation algorithms for VRPP-specific valid inequalities:
- RCC (Rounded Capacity Cuts): Standard VRP cuts derived from bin-packing
  requirements (Lysgaard et al. 2004). Essential for problems with fractional
  load coverage.
- 3-SRI (Subset-Row Inequalities): Tightens the set partitioning relaxation
  for subsets of 3 nodes (Jepsen et al. 2008). These handle the "entanglement"
  of fractional routes that partially cover the same customer set.
- Fleet Cover: 0-1 knapsack covers derived from the global vehicle fleet
  limit (adapted from Balas 1975). These strengthen the fleet-size knapsack
  by lifting minimal covers on the route-selection variables.
- SEC (Subtour Elimination Cuts): Enforces connectivity for prize-collecting
  variants where the route must be a simple cycle connected to the depot.

Methodology:
    Each engine implements a `separate_and_add_cuts` method that inspects the
    fractional LP solution of the Master Problem, identifies violated
    inequalities from its specific family, and registers them into the MP's
    Gurobi model. The dual variables from these cuts are then passed back to
    the RCSPP pricing solver.

References:
    - Lysgaard, J., Letchford, A. N., & Eglese, R. W. (2004).
      "A new branch-and-cut algorithm for the capacitated vehicle routing problem."
      Mathematical Programming, 100(2), 423-445.
    - Jepsen, M., Petersen, B., Spoorendonk, S., & Pisinger, D. (2008).
      "Subset-row inequalities applied to the vehicle-routing problem with time windows."
      Operations Research, 56(2), 497-511.
    - Balas, E. (1975). "Facets of the knapsack polytope."
      Mathematical Programming, 8(1), 146-164 (Cover inequalities).

Attributes:
    CuttingPlaneEngine (class): Abstract base class for separation engines.
    RoundedCapacityCutEngine (class): RCC separation logic.
    SubsetRowCutEngine (class): SRI separation logic.
    EdgeCliqueCutEngine (class): Fleet cover logic.
    KnapsackCoverEngine (class): Generic knapsack cover logic.
    CompositeCuttingPlaneEngine (class): Orchestrator for multiple engines.

Example:
    >>> engine = create_cutting_plane_engine("rcc", v_model, sep_engine)
    >>> cuts_added = engine.separate_and_add_cuts(master, max_cuts=10)
"""

from typing import List, Optional

import numpy as np

from logic.src.policies.helpers.solvers_and_matheuristics.master_problem import VRPPMasterProblem
from logic.src.policies.helpers.solvers_and_matheuristics.search.cut_engine_base import CuttingPlaneEngine
from logic.src.policies.helpers.solvers_and_matheuristics.search.cuts_capacity import (
    KnapsackCoverEngine,
    RoundedCapacityCutEngine,
    RoundedMultistarCutEngine,
)
from logic.src.policies.helpers.solvers_and_matheuristics.search.cuts_clique_rank import (
    EdgeCliqueCutEngine,
    LimitedMemoryRank1CutEngine,
)
from logic.src.policies.helpers.solvers_and_matheuristics.search.cuts_fleet_cover import (
    BasicFleetCoverEngine,
    PhysicalCapacityLCIEngine,
    SaturatedArcLCIEngine,
)
from logic.src.policies.helpers.solvers_and_matheuristics.search.cuts_subset_row import SubsetRowCutEngine
from logic.src.policies.helpers.solvers_and_matheuristics.separation import (
    SeparationEngine,
)
from logic.src.policies.helpers.solvers_and_matheuristics.vrpp_model import VRPPModel


class CompositeCuttingPlaneEngine(CuttingPlaneEngine):
    """Combines multiple separation engines to run in sequence.

    Attributes:
        _engines (List[CuttingPlaneEngine]): Sub-engines to run.
    """

    def __init__(self, engines: List[CuttingPlaneEngine]) -> None:
        """Initializes the composite engine with a list of sub-engines.

        Args:
            engines: Sub-engines to run.
        """
        self._engines = engines

    @property
    def engines(self) -> List[CuttingPlaneEngine]:
        """Returns the list of sub-engines.

        Returns:
            List[CuttingPlaneEngine]: Sub-engines.
        """
        return self._engines

    def separate_and_add_cuts(self, master: VRPPMasterProblem, max_cuts: int, **kwargs) -> int:
        """Runs separation algorithms for all sub-engines in sequence.

        Args:
            master: Master problem instance.
            max_cuts: Total maximum cuts to add (split among sub-engines).
            kwargs: Forwarded to sub-engines.

        Returns:
            int: Total number of cuts added by all sub-engines.
        """
        per_engine = max(1, max_cuts // len(self.engines))
        total_added = 0
        for engine in self.engines:
            total_added += engine.separate_and_add_cuts(master, per_engine, **kwargs)
        return total_added

    def get_name(self) -> str:
        """Returns the identifier for this engine.

        Returns:
            str: The engine name 'composite'.
        """
        return "composite"


def create_cutting_plane_engine(  # noqa: C901
    engine_name: str,
    v_model: VRPPModel,
    sep_engine: Optional[SeparationEngine] = None,
    dist_matrix: Optional[np.ndarray] = None,
    route_budget: float = float("inf"),
    max_subset: int = 50,
    **kwargs: dict,
) -> CuttingPlaneEngine:
    """Factory function to create cutting plane engines.

    Args:
        engine_name: Name of the engine ("rcc", "sri", "edge_clique", "fleet_cover",
            "physical_lci", "saturated_arc_lci", "cover", "all", "composite").
        v_model: VRPP model for problem data.
        sep_engine: Separation engine (required for RCC/SRC/Cover).
        dist_matrix: The distance matrix.
        route_budget: The route budget.
        max_subset: The maximum subset size.
        kwargs: Additional parameters for the engine.

    Returns:
        CuttingPlaneEngine: Instance of the requested cutting plane engine.

    Raises:
        ValueError: If engine_name is not recognized or required args missing.

    Example:
        >>> sep_engine = SeparationEngine(v_model)
        >>> cut_engine = create_cutting_plane_engine("rcc", v_model, sep_engine)
        >>> cuts_added = cut_engine.separate_and_add_cuts(master, max_cuts=5)
    """
    if engine_name == "rcc":
        if sep_engine is None:
            raise ValueError("RCC engine requires sep_engine parameter")
        return RoundedCapacityCutEngine(v_model, sep_engine)
    elif engine_name == "sri":
        return SubsetRowCutEngine(v_model)
    elif engine_name == "edge_clique":
        return EdgeCliqueCutEngine(v_model)
    elif engine_name == "fleet_cover":
        return BasicFleetCoverEngine(v_model)
    elif engine_name == "physical_lci":
        return PhysicalCapacityLCIEngine(v_model)
    elif engine_name == "saturated_arc_lci":
        return SaturatedArcLCIEngine(v_model)
    elif engine_name == "multistar":
        if sep_engine is None:
            raise ValueError("Multistar engine requires sep_engine parameter")
        return RoundedMultistarCutEngine(v_model, sep_engine)
    elif engine_name == "cover":
        if sep_engine is None:
            raise ValueError("Cover engine requires sep_engine parameter")
        return KnapsackCoverEngine(v_model, sep_engine)
    elif engine_name == "limited_memory_rank1":
        max_subset = kwargs.get("max_subset_size", 5)
        return LimitedMemoryRank1CutEngine(v_model, max_subset_size=max_subset)
    elif engine_name in ("all", "composite"):
        engines = []
        if sep_engine is not None:
            engines.append(RoundedCapacityCutEngine(v_model, sep_engine))
        engines.append(SubsetRowCutEngine(v_model))  # type: ignore[arg-type]
        # Edge clique is always added as a spatial base cut
        engines.append(EdgeCliqueCutEngine(v_model))  # type: ignore[arg-type]
        engines.append(BasicFleetCoverEngine(v_model))  # type: ignore[arg-type]
        engines.append(PhysicalCapacityLCIEngine(v_model))  # type: ignore[arg-type]
        engines.append(SaturatedArcLCIEngine(v_model))  # type: ignore[arg-type]
        if sep_engine is not None:
            engines.append(KnapsackCoverEngine(v_model, sep_engine))  # type: ignore[arg-type]
            engines.append(RoundedMultistarCutEngine(v_model, sep_engine))  # type: ignore[arg-type]
        # ── Tier 1 + selection-routing duality cuts ──────────────────────
        engines.append(LimitedMemoryRank1CutEngine(v_model, max_subset_size=5))

        comp = CompositeCuttingPlaneEngine(engines)  # type: ignore[arg-type]
        return comp
    else:
        valid = [
            "rcc",
            "sri",
            "edge_clique",
            "fleet_cover",
            "physical_lci",
            "saturated_arc_lci",
            "multistar",
            "cover",
            "limited_memory_rank1",
            "all",
            "composite",
        ]
        raise ValueError(f"Unknown cutting plane engine '{engine_name}'. Valid options are: {valid}")


__all__ = [
    "CompositeCuttingPlaneEngine",
    "create_cutting_plane_engine",
    "CuttingPlaneEngine",
    "RoundedCapacityCutEngine",
    "KnapsackCoverEngine",
    "RoundedMultistarCutEngine",
    "SubsetRowCutEngine",
    "EdgeCliqueCutEngine",
    "LimitedMemoryRank1CutEngine",
    "BasicFleetCoverEngine",
    "PhysicalCapacityLCIEngine",
    "SaturatedArcLCIEngine",
]
