"""
Potential-Based Reward Shaping (PBRS) for constructive RL.

Reference:
    Ng, A. Y., Harada, D., & Russell, S. J. (1999).
    Policy invariance under reward transformations: Theory and application to
    reward shaping.
    In Proceedings of ICML (Vol. 99, pp. 278-287).

Mathematical Guarantee (Ng et al., 1999):
    The shaped reward  R_total = R_base + F(s, a, s')
    where  F(s, a, s') = γ·Φ(s') − Φ(s)
    preserves the optimal policy of R_base IFF Φ(terminal) = 0.

Episode-Level Note:
    WSmart-Route uses *constructive* heuristics (AM, POMO) in which the policy
    unrolls a complete tour in a single ``policy(td, env)`` call. From the
    training-loop perspective this is a **single-step MDP**:

        s  = initial state after ``env.reset()``        (Φ = 0: no waste collected)
        s' = terminal state after the full tour completes (Φ = collected/max)

    F = γ·Φ(s') − Φ(s) = γ·(collected/max) − 0  →  positive bonus ∝ tour quality.

    The terminal-zero convention of Ng et al. is satisfied at *s* (the current
    state), because nothing has been collected yet: Φ(s_0) = 0.  We do NOT
    force Φ(s') = 0 at the terminal state because that would collapse the shaping
    to a constant and provide no learning signal.

Attributes:
    PBRSShaper:            Episode-level PBRS engine.
    get_potential_ptp:    Potential function for PTPEnv.
    get_potential_fn:      Factory that returns the correct Φ for an env name.

Example:
    >>> shaper = PBRSShaper(gamma=1.0, env_name="ptp")
    >>> shaper.record_initial(td_after_reset)
    >>> shaped, F = shaper.apply(base_reward, final_td)
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch
from tensordict import TensorDict

from logic.src.tracking.logging.pylogger import get_pylogger

logger = get_pylogger(__name__)

# ---------------------------------------------------------------------------
# Potential functions  Φ(s)
# ---------------------------------------------------------------------------


def get_potential_ptp(td: TensorDict) -> torch.Tensor:
    """Potential function Φ(s) for PTPEnv.

    Φ(s) = collected_waste / max_possible_waste  ∈  [0, 1]

    Properties:
    - **State-only**: depends only on ``collected_waste`` and ``waste`` fields,
      never on the action taken or the base reward.
    - **Non-negative**: guaranteed by clamping.
    - **Φ(s_0) = 0**: at reset ``collected_waste`` is initialised to 0, so the
      potential of the initial state is exactly 0 — satisfying the terminal-zero
      convention for the *current-state* side of the PBRS formula.

    Args:
        td: TensorDict containing at minimum ``collected_waste`` (batch,) and
            ``waste`` (batch, num_nodes).

    Returns:
        torch.Tensor: Scalar potential per instance, shape ``(batch,)``.
    """
    collected: Optional[torch.Tensor] = td.get("collected_waste", None)
    waste: Optional[torch.Tensor] = td.get("waste", None)

    if collected is None or waste is None:
        return torch.zeros(td.batch_size, device=td.device)

    # Depot entry in ``waste`` is always 0; summing over all nodes is safe.
    max_waste = waste.sum(dim=-1).clamp(min=1e-8)  # (batch,)
    return (collected / max_waste).clamp(0.0, 1.0)


def _get_potential_not_implemented(td: TensorDict) -> torch.Tensor:
    """Stub Φ returning zeros for unsupported environments.

    Replace this with a domain-specific function once a potential is defined
    for the target environment.

    Args:
        td: TensorDict (unused).

    Returns:
        torch.Tensor: Zero tensor matching the batch size.
    """
    logger.warning(
        "PBRS: No potential function defined for this environment. Shaping will be zero — PBRS is effectively disabled."
    )
    return torch.zeros(td.batch_size, device=td.device)


#: Registry mapping env-name → Φ callable.
_POTENTIAL_REGISTRY: dict[str, Callable[[TensorDict], torch.Tensor]] = {
    "ptp": get_potential_ptp,
}


def get_potential_fn(env_name: str) -> Callable[[TensorDict], torch.Tensor]:
    """Return the potential function registered for *env_name*.

    Args:
        env_name: Name of the environment (e.g. ``"ptp"``).

    Returns:
        Callable that maps a TensorDict state → scalar potential tensor.
    """
    fn = _POTENTIAL_REGISTRY.get(env_name)
    if fn is None:
        logger.warning(
            f"PBRS: No potential function registered for env '{env_name}'. "
            "Falling back to zero potential — shaping disabled."
        )
        return _get_potential_not_implemented
    return fn


# ---------------------------------------------------------------------------
# Episode-level PBRS shaper
# ---------------------------------------------------------------------------


class PBRSShaper:
    """Episode-level Potential-Based Reward Shaper.

    Computes the shaping bonus:

        F(s, a, s') = γ · Φ(s') − Φ(s)

    where *s* is the initial state (after ``env.reset()``) and *s'* is the
    terminal state (after the full constructive tour).

    Usage (called from ``shared_step`` in :class:`StepMixin`):

    .. code-block:: python

        # 1. After env.reset():
        shaper.record_initial(td)

        # 2. After policy rollout:
        shaped_reward, F = shaper.apply(out["reward"], final_td)

        # 3. Store shaped reward for training, log base separately:
        out["reward_base"]    = out["reward"]
        out["reward_shaping"] = F
        out["reward"]         = shaped_reward

    Attributes:
        gamma:           Discount factor γ (pulled from ``rl.gamma``).
        env_name:        Environment name used to look up Φ.
        shaping_weight:  Scalar multiplier for F before adding to R_base.
        _potential_fn:   Active Φ callable.
        _phi_0:          Φ(s_0) recorded at the last ``record_initial`` call.
    """

    def __init__(
        self,
        gamma: float,
        env_name: str,
        shaping_weight: float = 1.0,
        potential_fn: Optional[Callable[[TensorDict], torch.Tensor]] = None,
    ) -> None:
        """Initialise the shaper.

        Args:
            gamma:          Discount factor γ. Pass ``rl.gamma`` from config.
                            For non-discounted episodic tasks (PTP/POMO/REINFORCE)
                            this is typically ``1.0``.
            env_name:       Environment name (e.g. ``"ptp"``). Used to select
                            the registered potential function.
            shaping_weight: Scale applied to F before adding to R_base (default 1.0).
            potential_fn:   Override the registered Φ with a custom callable.
        """
        self.gamma: float = gamma
        self.env_name: str = env_name
        self.shaping_weight: float = shaping_weight
        self._potential_fn: Callable[[TensorDict], torch.Tensor] = (
            potential_fn if potential_fn is not None else get_potential_fn(env_name)
        )
        self._phi_0: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_initial(self, td: TensorDict) -> None:
        """Record Φ(s_0) immediately after ``env.reset()``.

        Must be called once per batch before ``apply()``.

        For PTPEnv this will always be **0** because ``collected_waste`` is
        initialised to zero at reset. The call is still required to ensure
        the batch dimension is captured correctly.

        Args:
            td: The TensorDict returned by ``env.reset()``.
        """
        # γ=1, Φ=0 → F will equal γ·Φ(s_final).  Store anyway for generality.
        self._phi_0 = self._potential_fn(td).detach().clone()

    def apply(
        self,
        base_reward: torch.Tensor,
        final_td: TensorDict,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute F and return (shaped_reward, shaping_reward).

        Should be called once per batch after the full policy rollout.

        **γ is defined and applied here**:
            shaping = γ · Φ(s_final) − Φ(s_0)

        **Φ(s) is formulated as**:
            collected_waste / sum(waste)  for PTPEnv  (see :func:`get_potential_ptp`).

        **Terminal state handling**:
            For PTPEnv, ``Φ(s_0) = 0`` (no waste collected at reset).
            ``Φ(s_final)`` is evaluated from ``final_td["collected_waste"]`` after
            the complete tour; it is **not** forced to 0 because doing so would
            make F = 0 always (degenerate case for constructive RL).
            See module docstring for the full rationale.

        Args:
            base_reward: Unmodified env reward, shape ``(batch,)`` or ``(batch, 1)``.
            final_td:    TensorDict from the last rollout step (contains
                         ``collected_waste``, ``waste``, etc.).

        Returns:
            Tuple of:
            - **shaped_reward** ``(batch,)`` or ``(batch, 1)``: R_base + shaping_weight·F.
            - **shaping_reward** ``(batch,)`` or ``(batch, 1)``: F alone (for logging).
        """
        if self._phi_0 is None:
            logger.warning("PBRSShaper.apply() called before record_initial(). Shaping disabled for this batch.")
            return base_reward, torch.zeros_like(base_reward)

        # --- Evaluate Φ(s') -----------------------------------------------------
        phi_final = self._potential_fn(final_td).detach()

        # Broadcast phi tensors to match base_reward shape
        phi_0 = self._phi_0
        if base_reward.dim() > phi_final.dim():
            phi_final = phi_final.unsqueeze(-1)
            phi_0 = phi_0.unsqueeze(-1)

        # --- Apply the PBRS formula: F = γ·Φ(s') − Φ(s) ------------------------
        # γ is self.gamma, defined at construction from rl.gamma config key.
        shaping = self.gamma * phi_final - phi_0  # (batch,) or (batch, 1)
        shaping = self.shaping_weight * shaping

        shaped_reward = base_reward + shaping
        return shaped_reward, shaping

    def reset(self) -> None:
        """Clear the stored initial potential (call between episodes if needed)."""
        self._phi_0 = None

    def __repr__(self) -> str:
        """Human-readable representation.

        Returns:
            str: String representation.
        """
        return (
            f"PBRSShaper("
            f"env={self.env_name}, "
            f"gamma={self.gamma}, "
            f"weight={self.shaping_weight}, "
            f"phi_fn={self._potential_fn.__name__})"
        )
