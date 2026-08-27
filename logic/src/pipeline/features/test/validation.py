"""
Validation logic for simulation testing pipeline.

Attributes:
    validate_sim_config: Validates and sanitizes simulation configuration values in-place.

Example:
    >>> validate_sim_config(cfg)
    Traceback (most recent call last):
        ...  # doctest: +ELLIPSIS
    AssertionError: Must run the simulation for 1 or more days
"""

import re
from multiprocessing import cpu_count

from logic.src.configs import Config
from logic.src.constants import MAP_DEPOTS, PROBLEMS, WASTE_TYPES


def validate_sim_config(cfg: Config) -> None:
    """
    Validates and sanitizes simulation configuration values in-place.

    Performs the same checks previously done by ``validate_test_sim_args``
    on the flattened args dict, now applied directly to the typed Config.

    Args:
        cfg: Root Hydra configuration with ``cfg.sim`` containing simulation
            parameters.

    Raises:
        AssertionError: If any validation constraint is violated.
    """
    sim = cfg.sim
    graph = sim.graph

    # --- Core constraints ---
    assert graph.n_days >= 1, "Must run the simulation for 1 or more days"
    assert graph.n_samples > 0, "Number of samples must be non-negative integer"

    # --- Problem validation ---
    problem = str(getattr(sim, "problem", "vrpp") or "vrpp").lower()
    assert problem in PROBLEMS, f"Unknown problem {problem}, available problems: {PROBLEMS}"
    sim.problem = problem

    # --- Sanitize area ---
    area = re.sub(r"[^a-zA-Z]", "", (graph.area or "").lower())
    assert area in MAP_DEPOTS, f"Unknown area {area}, available areas: {list(MAP_DEPOTS.keys())}"
    graph.area = area

    # --- Sanitize waste_type ---
    waste = re.sub(r"[^a-zA-Z]", "", (graph.waste_type or "").lower())
    assert waste in WASTE_TYPES or waste == "", (
        f"Unknown waste type {waste}, available waste types: {list(WASTE_TYPES.keys())}"
    )
    if waste:
        graph.waste_type = waste

    # --- Parse edge_threshold ---
    et_str = str(graph.edge_threshold or "0")
    graph.edge_threshold = et_str

    # --- CPU cores ---
    cores = getattr(sim, "cpu_cores", 0) or 0
    assert cores >= 0, "Number of CPU cores must be non-negative integer"
    assert cores <= cpu_count(), "Number of CPU cores to use cannot exceed system specifications"
    if cores == 0:
        sim.cpu_cores = cpu_count()
