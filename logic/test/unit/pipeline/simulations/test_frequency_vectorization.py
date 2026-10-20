"""Tests for the vectorized collection frequency/level calculation.

``calculate_frequency_and_level_vectorized`` must reproduce the scalar
``calculate_frequency_and_level`` exactly, element for element, including
degenerate zero mean/variance inputs.
"""

import numpy as np
import pytest
from logic.src.pipeline.simulations.bins.prediction import (
    calculate_frequency_and_level,
    calculate_frequency_and_level_vectorized,
)

pytestmark = [pytest.mark.unit, pytest.mark.fast]


class TestFrequencyVectorization:
    """Vectorized vs scalar equivalence tests."""

    @pytest.mark.unit
    def test_matches_scalar_on_random_inputs(self):
        """Vectorized output equals the scalar per-bin results elementwise."""
        rng = np.random.default_rng(0)
        n = 2000
        ui = rng.uniform(0.0, 5.0, n)
        vi = rng.uniform(0.0, 5.0, n)
        ui[::17] = 0.0  # exact zeros must hit the same 1e-9 guard as the scalar
        vi[::19] = 0.0

        for cf in (0.5, 0.9, 0.99):
            freq_v, level_v = calculate_frequency_and_level_vectorized(ui, vi, cf)

            freq_s = np.empty(n, dtype=int)
            level_s = np.empty(n, dtype=float)
            for i in range(n):
                freq_s[i], level_s[i] = calculate_frequency_and_level(float(ui[i]), float(vi[i]), cf)

            assert np.array_equal(freq_v, freq_s), f"frequency mismatch at cf={cf}"
            np.testing.assert_allclose(level_v, level_s, rtol=0, atol=1e-10, err_msg=f"level mismatch at cf={cf}")

    @pytest.mark.unit
    def test_frequency_dtype_is_integer(self):
        """Frequencies come back with an integer dtype."""
        rng = np.random.default_rng(0)
        ui = rng.uniform(0.0, 5.0, 64)
        vi = rng.uniform(0.0, 5.0, 64)
        freq_v, level_v = calculate_frequency_and_level_vectorized(ui, vi, 0.9)
        assert np.issubdtype(freq_v.dtype, np.integer)
        assert np.issubdtype(level_v.dtype, np.floating)
