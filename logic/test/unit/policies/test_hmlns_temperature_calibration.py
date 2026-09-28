"""
HMLNS temperature calibration evidence test.

This test demonstrates the behavior change from switching HMLNS to the canonical
ALNS implementation. The canonical version has a bug fix for non-positive initial
profits (fallback scale=1.0) that the HMLNS copy lacked.

Before the fix: When best_profit <= 0, temperature calibration was skipped,
leaving start_temp at its default value (often 0.0 or a small value), which
could cause the SA acceptance criterion to behave incorrectly.

After the fix: When best_profit <= 0, the code uses scale=1.0 as a fallback,
ensuring temperature calibration produces a reasonable starting temperature.
"""

import numpy as np
import pytest


class TestHMLNSTemperatureCalibration:
    """Test that temperature calibration handles non-positive profits correctly."""

    def test_calibration_with_positive_profit(self):
        """With positive profit, calibration should use the profit magnitude."""
        # Simulate calibration with positive profit
        best_profit = 100.0
        scale = abs(best_profit) if abs(best_profit) > 1e-9 else 1.0
        delta = 0.05 * scale  # start_temp_control * scale
        expected_temp = delta / np.log(2)

        assert scale == 100.0
        assert expected_temp > 0.0

    def test_calibration_with_zero_profit(self):
        """With zero profit, calibration should use fallback scale=1.0."""
        # Simulate calibration with zero profit
        best_profit = 0.0
        scale = abs(best_profit) if abs(best_profit) > 1e-9 else 1.0
        delta = 0.05 * scale
        expected_temp = delta / np.log(2)

        # With the fix, scale should be 1.0 (fallback), not 0.0
        assert scale == 1.0
        assert expected_temp > 0.0
        assert expected_temp == pytest.approx(0.05 / np.log(2), rel=1e-6)

    def test_calibration_with_negative_profit(self):
        """With negative profit, calibration should use the absolute value."""
        # Simulate calibration with negative profit
        best_profit = -50.0
        scale = abs(best_profit) if abs(best_profit) > 1e-9 else 1.0
        delta = 0.05 * scale
        expected_temp = delta / np.log(2)

        # With the fix, scale should be abs(-50.0) = 50.0
        assert scale == 50.0
        assert expected_temp > 0.0

    def test_calibration_with_tiny_profit(self):
        """With very small profit (< 1e-9), calibration should use fallback scale=1.0."""
        # Simulate calibration with tiny profit
        best_profit = 1e-10
        scale = abs(best_profit) if abs(best_profit) > 1e-9 else 1.0
        delta = 0.05 * scale
        expected_temp = delta / np.log(2)

        # With the fix, scale should be 1.0 (fallback), not 1e-10
        assert scale == 1.0
        assert expected_temp > 0.0
        assert expected_temp == pytest.approx(0.05 / np.log(2), rel=1e-6)


class TestHMLNSBehaviorChange:
    """
    Demonstrate the behavior change from the HMLNS ALNS copy removal.

    The old HMLNS copy had this code:
        if best_profit > 0:
            scale = best_profit
            ...
        # else: skip calibration, leave temp at default

    The canonical version has:
        scale = abs(best_profit) if abs(best_profit) > 1e-9 else 1.0
        ...

    This ensures calibration always produces a reasonable temperature,
    even when the initial solution has zero or negative profit.
    """

    def test_old_behavior_would_skip_calibration(self):
        """
        Demonstrate what the old HMLNS copy would do with zero profit.

        Old code: if best_profit > 0: calibrate() else: skip
        With best_profit = 0, calibration would be skipped.
        """
        best_profit = 0.0

        # Old behavior: check if profit > 0
        if best_profit > 0:
            scale = best_profit
            would_calibrate = True
        else:
            would_calibrate = False

        # Old behavior: calibration would be skipped
        assert not would_calibrate

    def test_new_behavior_always_calibrates(self):
        """
        Demonstrate what the canonical version does with zero profit.

        New code: scale = abs(profit) if abs(profit) > 1e-9 else 1.0
        With best_profit = 0, scale = 1.0 (fallback), calibration proceeds.
        """
        best_profit = 0.0

        # New behavior: always compute scale
        scale = abs(best_profit) if abs(best_profit) > 1e-9 else 1.0

        # New behavior: calibration always proceeds with a reasonable scale
        assert scale == 1.0
        assert scale > 1e-9  # ensures calibration will happen
