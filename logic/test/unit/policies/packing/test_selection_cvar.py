"""CVaR selection: the selection threshold compares against the exact CVaR of max(0, X)."""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm

from logic.src.interfaces.context.selection_context import SelectionContext
from logic.src.policies.packing.selection_cvar import CVaRSelection


def _exact_cvar_of_overflow(m, s, alpha):
    """CVaR_alpha(max(0, X)) for X ~ N(m, s^2), by quadrature in quantile space (z-space)."""
    z_a = norm.ppf(alpha)
    kink = [-m / s] if z_a < -m / s else None
    integral = quad(lambda z: max(0.0, m + s * z) * norm.pdf(z), z_a, 40.0, limit=400, points=kink)[0]
    return integral / (1.0 - alpha)


@pytest.mark.parametrize(
    "surplus_mean, sigma, alpha",
    [
        (-6.0, 2.0, 0.95),  # VaR below zero: tail starts at the atom at zero
        (-5.0, 4.0, 0.9),
        (0.0, 3.0, 0.95),  # VaR above zero: tail is the whole of X
        (8.0, 2.0, 0.99),
        (-1.0, 0.5, 0.6),
    ],
)
def test_selection_threshold_matches_exact_cvar(surplus_mean, sigma, alpha):
    max_fill = 100.0
    current_fill = np.array([surplus_mean + max_fill])  # so that mu = 0 gives X mean = surplus_mean
    exact = _exact_cvar_of_overflow(surplus_mean, sigma, alpha)

    def selected(threshold):
        ctx = SelectionContext(
            bin_ids=np.array([1], dtype="int32"),
            current_fill=current_fill,
            threshold=threshold,
            max_fill=max_fill,
            bin_volume=1.0,
            bin_density=1.0,
            revenue_kg=1.0,
            vehicle_capacity=100.0,
            distance_matrix=np.zeros((2, 2)),
            accumulation_rates=np.zeros(1),
            std_deviations=np.array([sigma]),
            cvar_alpha=alpha,
        )
        bins, _ = CVaRSelection().select_bins(ctx)
        return bool(bins)

    assert selected(exact * 0.999)
    assert not selected(exact * 1.001)
