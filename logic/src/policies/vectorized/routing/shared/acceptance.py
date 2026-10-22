"""Batched acceptance criteria for the vectorised policies.

Each instance of the batch gets its own criterion built by the simulator's
:class:`AcceptanceCriterionFactory` from the same ``acceptance_criterion`` config the
simulator policies take, so the accept/step rules (Boltzmann calibration and cooling,
only-improving, late acceptance, ...) are the simulator's own.

Attributes:
    BatchedAcceptance: One simulator acceptance criterion per batch instance.

Example:
    >>> acc = BatchedAcceptance(AcceptanceConfig(method="oi"), batch_size=4)
    >>> acc.setup(profit)
    >>> mask = acc.accept(current, candidate, best, iteration, max_iterations)
"""

from __future__ import annotations

from typing import Any, List, Optional

import torch


class BatchedAcceptance:
    """One acceptance criterion per instance, built from a simulator acceptance config.

    Attributes:
        criteria: The per-instance criteria (None means plain hill climbing).
    """

    def __init__(self, spec: Any, batch_size: int) -> None:
        """Build ``batch_size`` criteria from ``spec``.

        Args:
            spec: ``AcceptanceConfig``, ``{"method", "params"}`` mapping or None.
            batch_size: Number of instances.
        """
        from logic.src.policies.acceptance_criteria.base.factory import AcceptanceCriterionFactory

        self.criteria: List[Optional[Any]] = []
        for _ in range(batch_size):
            self.criteria.append(AcceptanceCriterionFactory.build(spec) if spec is not None else None)

    def setup(self, objective: torch.Tensor) -> None:
        """Initialise every criterion with its instance's starting objective.

        Args:
            objective: Starting objective (profit) ``[B]``.
        """
        vals = objective.detach().cpu().tolist()
        for crit, v in zip(self.criteria, vals, strict=True):
            if crit is not None and hasattr(crit, "setup"):
                crit.setup(float(v))

    def accept(
        self,
        current: torch.Tensor,
        candidate: torch.Tensor,
        best: Optional[torch.Tensor] = None,
        iteration: int = 0,
        max_iterations: int = 0,
        active: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Accept flags ``[B]``.

        Args:
            current: Current objective ``[B]``.
            candidate: Candidate objective ``[B]``.
            best: Best objective so far ``[B]``.
            iteration: Iteration index.
            max_iterations: Iteration budget.
            active: Instances to decide for (others get False).

        Returns:
            torch.Tensor: Boolean accept mask.
        """
        cur = current.detach().cpu().tolist()
        cand = candidate.detach().cpu().tolist()
        bst = best.detach().cpu().tolist() if best is not None else cur
        act = active.detach().cpu().tolist() if active is not None else [True] * len(cur)
        out = []
        for i, crit in enumerate(self.criteria):
            if not act[i]:
                out.append(False)
            elif crit is None:
                out.append(cand[i] > cur[i] + 1e-6)
            else:
                ok, _ = crit.accept(
                    current_obj=cur[i],
                    candidate_obj=cand[i],
                    f_best=bst[i],
                    iteration=iteration,
                    max_iterations=max_iterations,
                )
                out.append(bool(ok))
        return torch.tensor(out, dtype=torch.bool, device=current.device)

    def step(
        self,
        current: torch.Tensor,
        candidate: torch.Tensor,
        accepted: torch.Tensor,
        best: Optional[torch.Tensor] = None,
        iteration: int = 0,
        max_iterations: int = 0,
        active: Optional[torch.Tensor] = None,
    ) -> None:
        """Advance every criterion's state (cooling, counters, ...).

        Args:
            current: Current objective ``[B]``.
            candidate: Candidate objective ``[B]``.
            accepted: Accept flags ``[B]``.
            best: Best objective so far ``[B]``.
            iteration: Iteration index.
            max_iterations: Iteration budget.
            active: Instances to step (others are left alone).
        """
        cur = current.detach().cpu().tolist()
        cand = candidate.detach().cpu().tolist()
        acc = accepted.detach().cpu().tolist()
        bst = best.detach().cpu().tolist() if best is not None else cur
        act = active.detach().cpu().tolist() if active is not None else [True] * len(cur)
        for i, crit in enumerate(self.criteria):
            if crit is None or not act[i]:
                continue
            crit.step(
                current_obj=cur[i],
                candidate_obj=cand[i],
                accepted=bool(acc[i]),
                f_best=bst[i],
                iteration=iteration,
                max_iterations=max_iterations,
            )
