"""
Lagrangian Multi-Period Matheuristic for the Profitable Tour Problem
with stochastic, time-varying, capacity-bounded prizes.

Architecture (see design discussion for full derivation):

    * Scenario-tree based lookahead prize valuation (ScenarioGenerator).
    * Per-node DP with overflow saturation -> expected collectable prize V[i,t].
    * Adaptive regret-based preprocessing for overflow-bound bins
      (soft coefficient bias, escalating to hard fixing on primal stagnation).
    * Insertion-cost oracle shared between selection and routing, with gated
      EMA updates (ignore RS returns that are worse than incumbent by > threshold).
    * Lagrangian relaxation of the linking x^K == x^R with decoupled variable
      copies.  Multiplier updates piggyback on RS completion events (option 1b
      / RS-thread update, no dedicated coordinator thread).
    * Dual-bound tracking: selectable between (a) gated-EMA best-so-far or
      (b) proximal bundle method with a QP subsolve.
    * TPKS (Two-Phase Kernel Search) as the per-period selection engine,
      invoked with Lagrangian-corrected objective coefficients.
    * RL augmentations:
        - LinUCB contextual bandit selects {TPKS engine variant, cut strategy}
          per outer iteration.

Entry point: :class:`policy.CALMPolicy`.

Attributes:
  None

Example:
  None
"""

from .params import (
    BanditParams,
    CALMParams,
    DualBoundParams,
    LagrangianParams,
    LookaheadParams,
    RegretParams,
)
from .policy_calm import CALMPolicy

__all__ = [
    "BanditParams",
    "DualBoundParams",
    "CALMParams",
    "CALMPolicy",
    "LagrangianParams",
    "LookaheadParams",
    "RegretParams",
]
