"""TpksConfig: configuration dataclass for the ``tpks`` policy.

Moved from the policy's ``params.py`` by the WP-06 scripted migration(Q-8b): the config dataclass is the single source of truth (P-10).
"""

from __future__ import annotations

from dataclasses import dataclass, fields


@dataclass
class TpksConfig:
    """
    Parameters for the Two-Phase Kernel Search (TPKS) algorithm.

    Attributes:
        phase1_kernel_size: Initial kernel size for feasibility search.
        phase1_bucket_size: Bucket size during Phase I expansion.
        phase1_time_fraction: Fraction of time_limit for Phase I.
        phase1_mip_node_limit: Node limit for Phase I MIP solves.
        phase2_bucket_size_easy: Bucket size for easy instances in Phase II.
        phase2_bucket_size_normal: Bucket size for normal instances in Phase II.
        phase2_time_limit_per_bucket: Time limit per bucket in Phase II.
        max_buckets: Maximum number of buckets in Phase II.
        t_easy: Difficulty threshold (seconds) to distinguish easy/normal instances.
        epsilon: Hardness parameter for fixing variables in Phase II (relaxation > 1 - ε).
        time_limit: Total time limit for the algorithm.
        mip_gap: MIP relative gap tolerance.
        mip_limit_nodes: Node limit for all MIP solves.
        initial_kernel_size: Fallback kernel size if phase1 var stats unavailable.
        seed: Random seed for reproducibility.
        engine: Solver engine to use (e.g., "gurobi", "cbc").
    """

    # Phase I
    phase1_kernel_size: int = 30  # initial kernel for feasibility search
    phase1_bucket_size: int = 15  # bucket size during Phase I expansion
    phase1_time_fraction: float = 0.35  # fraction of time_limit for Phase I
    phase1_mip_node_limit: int = 2000

    # Phase II
    phase2_bucket_size_easy: int = 30  # larger buckets for EASY instances
    phase2_bucket_size_normal: int = 15
    phase2_time_limit_per_bucket: float = 15.0  # seconds (adaptive override)
    max_buckets: int = 20
    t_easy: float = 8.0  # Phase II difficulty threshold (seconds)
    epsilon: float = 0.1  # HARD: fix variables with relaxation > 1 - ε

    # Shared
    time_limit: float = 300.0
    mip_gap: float = 0.01
    mip_limit_nodes: int = 10000
    initial_kernel_size: int = 50  # fallback if phase1 var stats unavailable
    seed: int = 42
    engine: str = "gurobi"


    def to_dict(self):
        """Converts the parameter dataclass into a standard Python dictionary.

        Returns:
            Dict[str, Any]: Dictionary representation of the parameters.
        """
        return {f.name: getattr(self, f.name) for f in fields(self)}
