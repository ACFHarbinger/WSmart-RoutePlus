"""
CVaR Packing Algorithm Module.

Computes the Conditional Value-at-Risk (CVaR) at level alpha for the anticipated
overflow volume in the next period. Under a Gaussian assumption for accumulation,
this risk measure provides a robust estimate of "tail" overflow risk, focusing
on extreme outcomes rather than just the mean.

Attributes:
    None

Example:
    >>> from logic.src.policies.packing.cvar import CVaRSelection
    >>> strategy = CVaRSelection()
    >>> bins, ctx = strategy.pack(context)
"""

from typing import List, Tuple

import numpy as np
from scipy.stats import norm

from logic.src.enums import PolicyTag
from logic.src.interfaces.context import PackingContext, SearchContext
from logic.src.interfaces.packing import IPackingAlgorithm
from logic.src.policies.packing.base.registry import PackingRegistry
from logic.src.utils.registry import ALL


@ALL.register(
    PolicyTag.PACKING,
    PolicyTag.HEURISTIC,
    PolicyTag.STOCHASTIC,
)
@PackingRegistry.register("cvar")
class CVaRSelection(IPackingAlgorithm):
    """Selection strategy based on Tail risk (CVaR of overflow).

    Attributes:
        None
    """

    def pack(self, context: PackingContext) -> Tuple[List[int], SearchContext]:
        """
        Select bins where tail overflow risk exceeds the threshold.

        Args:
            context: The selection context.

        Returns:
            Tuple[List[int], SearchContext]: Selected bin IDs and search context.
        """
        n_bins = len(context.current_fill)
        if n_bins == 0:
            return [], SearchContext.initialize(selection_metrics={"strategy": "CVaRSelection"})

        mu = context.accumulation_rates
        if mu is None:
            raise ValueError("CVaRSelection requires accumulation_rates.")

        sigma = context.std_deviations if context.std_deviations is not None else np.zeros(n_bins)
        alpha = context.cvar_alpha
        max_fill = context.max_fill
        threshold = context.threshold

        # Future fill: F ~ N(current_fill + mu, sigma^2)
        # Surplus: X = F - max_fill ~ N(current_fill + mu - max_fill, sigma^2)
        x_mu = context.current_fill + mu - max_fill
        x_sigma = np.where(sigma <= 1e-9, 1e-9, sigma)

        # VaR_alpha for X
        z_alpha = norm.ppf(alpha)
        var_alpha = x_mu + x_sigma * z_alpha

        # CVaR_alpha for X (signed surplus)
        cvar_x = x_mu + x_sigma * (norm.pdf(z_alpha) / (1.0 - alpha))

        # We care about CVaR_alpha(max(0, X)) i.e. the tail of overflow. Since max(0, .) is
        # monotone, VaR_u(max(0, X)) = max(0, VaR_u(X)), so:
        # 1. If var_alpha > 0, the whole tail is positive, so CVaR(max(0, X)) = CVaR(X).
        # 2. If var_alpha <= 0, the tail starts at the atom at zero. The quantiles below
        #    P(X <= 0) are clipped to 0, which leaves CVaR(max(0, X)) = E[max(0, X)] / (1 - alpha).
        #    This is exact, not a bound (checked by quadrature in test_selection_cvar.py).

        # Expected value E[max(0, X)]
        z_zero = x_mu / x_sigma
        expected_overflow = x_mu * norm.cdf(z_zero) + x_sigma * norm.pdf(z_zero)

        # Handle deterministic case
        deterministic_overflow = np.maximum(0, x_mu)

        # Selection criterion
        if np.all(sigma <= 1e-9):
            final_risk = deterministic_overflow
        else:
            # Vectorized selection: check if tail starts above zero
            final_risk = np.where(var_alpha > 0, cvar_x, expected_overflow / (1.0 - alpha))

        mandatory_indices = np.nonzero(final_risk > threshold)[0]

        return sorted((mandatory_indices + 1).tolist()), SearchContext.initialize(
            selection_metrics={"strategy": "CVaRSelection"}
        )
