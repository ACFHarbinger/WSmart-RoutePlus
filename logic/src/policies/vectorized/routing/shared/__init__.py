"""Shared components of the vectorised experts.

Attributes:
    vrpp: Batched VRPP instances, the profit Split DP and action/reward helpers.
    routes: Batched route-sequence encoding with the simulator's destroy, repair and
        local-search operators.
    acceptance: Per-instance simulator acceptance criteria.
    llh: The ILS low-level heuristics.
    base: ``VectorizedRoutingPolicy`` and simulator-config loading.
    factory: Simulator policy config -> vectorised expert.

Example:
    >>> from logic.src.policies.vectorized.routing.shared.factory import create_expert_policy
"""
