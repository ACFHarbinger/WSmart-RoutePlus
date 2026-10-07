"""
Unit tests for PBRSShaper and the PTP potential function.

Tests verify:
1. Φ(s_0) = 0 at episode start.
2. F = γ·Φ(s_final) − Φ(s_0) is computed correctly.
3. R_total = R_base + shaping_weight·F.
4. Policy-invariance property: when shaping_weight=0, R_total == R_base.
5. Unsupported env names log a warning and return zero shaping.
6. `reward_base` and `reward_shaping` are returned for logging.
7. apply() before record_initial() returns zero shaping (safe guard).
8. γ from config is correctly threaded through.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
from logic.src.pipeline.rl.common.pbrs_wrapper import (
    PBRSShaper,
    get_potential_fn,
    get_potential_ptp,
)
from tensordict import TensorDict

pytestmark = [pytest.mark.unit, pytest.mark.fast]




# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_td(
    collected: float,
    waste: list,
    batch_size: int = 1,
    device: str = "cpu",
) -> TensorDict:
    """Build a minimal PTP TensorDict for testing."""
    n = len(waste)
    return TensorDict(
        {
            "collected_waste": torch.tensor([collected] * batch_size),
            "waste": torch.tensor([waste] * batch_size, dtype=torch.float32),
            "locs": torch.zeros(batch_size, n + 1, 2),
            "current_node": torch.zeros(batch_size, 1, dtype=torch.long),
        },
        batch_size=[batch_size],
        device=device,
    )


# ---------------------------------------------------------------------------
# get_potential_ptp
# ---------------------------------------------------------------------------


class TestGetPotentialPTP:
    """Tests for the PTP potential function Φ(s)."""

    def test_zero_at_initial_state(self):
        """Φ(s_0) = 0 when nothing has been collected."""
        td = _make_td(collected=0.0, waste=[0.3, 0.5, 0.2])
        phi = get_potential_ptp(td)
        assert phi.shape == torch.Size([1])
        assert torch.allclose(phi, torch.zeros(1))

    def test_full_collection(self):
        """Φ = 1.0 when all waste is collected (perfect tour)."""
        waste = [0.3, 0.5, 0.2]
        td = _make_td(collected=sum(waste), waste=waste)
        phi = get_potential_ptp(td)
        assert torch.allclose(phi, torch.ones(1), atol=1e-6)

    def test_partial_collection(self):
        """Φ = collected / total for a partial tour."""
        waste = [1.0, 1.0, 1.0, 1.0]  # total = 4
        td = _make_td(collected=2.0, waste=waste)
        phi = get_potential_ptp(td)
        assert torch.allclose(phi, torch.tensor([0.5]), atol=1e-6)

    def test_batch_size_preserved(self):
        """Φ output batch size matches input."""
        td = _make_td(collected=1.0, waste=[1.0, 1.0], batch_size=8)
        phi = get_potential_ptp(td)
        assert phi.shape == torch.Size([8])

    def test_missing_fields_returns_zeros(self):
        """Missing 'collected_waste' or 'waste' → safe zero fallback."""
        td = TensorDict({}, batch_size=[4])
        phi = get_potential_ptp(td)
        assert phi.shape == torch.Size([4])
        assert (phi == 0.0).all()

    def test_clamped_above_one(self):
        """Φ is clamped to [0, 1] even if collected > total (data anomaly)."""
        waste = [1.0]
        td = _make_td(collected=9999.0, waste=waste)
        phi = get_potential_ptp(td)
        assert phi.item() <= 1.0

    def test_zero_waste_does_not_divide_by_zero(self):
        """All-zero waste tensor should not produce NaN or inf."""
        td = _make_td(collected=0.0, waste=[0.0, 0.0])
        phi = get_potential_ptp(td)
        assert not torch.isnan(phi).any()
        assert not torch.isinf(phi).any()


# ---------------------------------------------------------------------------
# get_potential_fn registry
# ---------------------------------------------------------------------------


class TestGetPotentialFn:
    """Tests for the potential function registry."""

    def test_ptp_returns_correct_fn(self):
        """Registry returns get_potential_ptp for 'ptp'."""
        fn = get_potential_fn("ptp")
        assert fn is get_potential_ptp

    def test_unknown_env_logs_warning_and_returns_zero(self):
        """Unknown env name logs a warning and returns a zero-shaping stub."""
        with patch("logic.src.pipeline.rl.common.pbrs_wrapper.logger") as mock_log:
            fn = get_potential_fn("not_a_real_env")
            mock_log.warning.assert_called_once()

        # The stub should return zeros
        td = TensorDict(
            {"collected_waste": torch.tensor([1.0])},
            batch_size=[1],
        )
        result = fn(td)
        assert (result == 0.0).all()


# ---------------------------------------------------------------------------
# PBRSShaper core functionality
# ---------------------------------------------------------------------------


class TestPBRSShaperCore:
    """Tests for the PBRSShaper class."""

    def _make_shaper(
        self, gamma: float = 1.0, shaping_weight: float = 1.0
    ) -> PBRSShaper:
        return PBRSShaper(gamma=gamma, env_name="ptp", shaping_weight=shaping_weight)

    # --- Requirement: γ is defined and applied --------------------------------

    def test_gamma_applied_in_formula(self):
        """F = γ·Φ(s') − Φ(s_0).  Different γ values produce different F."""
        td_initial = _make_td(collected=0.0, waste=[1.0, 1.0])  # Φ=0
        td_final = _make_td(collected=1.0, waste=[1.0, 1.0])    # Φ=0.5
        base_reward = torch.tensor([10.0])

        shaper_g1 = self._make_shaper(gamma=1.0)
        shaper_g1.record_initial(td_initial)
        _, F_g1 = shaper_g1.apply(base_reward, td_final)

        shaper_g05 = self._make_shaper(gamma=0.5)
        shaper_g05.record_initial(td_initial)
        _, F_g05 = shaper_g05.apply(base_reward, td_final)

        # F(γ=1) = 1.0 * 0.5 - 0 = 0.5
        assert torch.allclose(F_g1, torch.tensor([0.5]), atol=1e-5)
        # F(γ=0.5) = 0.5 * 0.5 - 0 = 0.25
        assert torch.allclose(F_g05, torch.tensor([0.25]), atol=1e-5)

    # --- Requirement: Φ(s) is state-only --------------------------------------

    def test_phi_s0_equals_zero_at_reset(self):
        """Φ(s_0) = 0 because collected_waste=0 at env reset."""
        td_initial = _make_td(collected=0.0, waste=[2.0, 3.0])
        shaper = self._make_shaper()
        shaper.record_initial(td_initial)
        # phi_0 should be 0
        assert shaper._phi_0 is not None
        assert torch.allclose(shaper._phi_0, torch.zeros(1))

    def test_shaping_formula_correct(self):
        """Verify F = γ·Φ(s') − Φ(s) for a known state pair (γ=1)."""
        waste = [2.0, 2.0]  # total=4
        td_initial = _make_td(collected=0.0, waste=waste)  # Φ=0
        td_final = _make_td(collected=3.0, waste=waste)    # Φ=0.75
        base_reward = torch.tensor([5.0])

        shaper = self._make_shaper(gamma=1.0)
        shaper.record_initial(td_initial)
        shaped, F = shaper.apply(base_reward, td_final)

        # F = 1.0 * 0.75 - 0.0 = 0.75
        assert torch.allclose(F, torch.tensor([0.75]), atol=1e-5)

    # --- Requirement: R_total = R_base + F ------------------------------------

    def test_total_reward_equals_base_plus_shaping(self):
        """R_total = R_base + shaping_weight·F."""
        waste = [1.0, 3.0]  # total=4
        td_initial = _make_td(collected=0.0, waste=waste)
        td_final = _make_td(collected=2.0, waste=waste)    # Φ=0.5
        base_reward = torch.tensor([8.0])

        shaper = self._make_shaper(gamma=1.0, shaping_weight=2.0)
        shaper.record_initial(td_initial)
        shaped, F = shaper.apply(base_reward, td_final)

        # apply() returns shaping already scaled by weight:
        # F_returned = weight * (γ * Φ(s') - Φ(s_0)) = 2.0 * 0.5 = 1.0
        # so shaped = base + F_returned = 8 + 1 = 9
        assert torch.allclose(F, torch.tensor([1.0]), atol=1e-5)   # weight * raw_F
        assert torch.allclose(shaped, torch.tensor([9.0]), atol=1e-5)
        assert torch.allclose(shaped, base_reward + F, atol=1e-5)  # identity


    # --- Anti-pattern guard: shaping_weight=0 recovers base reward ------------

    def test_zero_shaping_weight_recovers_base_reward(self):
        """Setting shaping_weight=0 disables PBRS (R_total == R_base)."""
        td_initial = _make_td(collected=0.0, waste=[1.0, 1.0])
        td_final = _make_td(collected=1.0, waste=[1.0, 1.0])
        base_reward = torch.tensor([7.5])

        shaper = self._make_shaper(shaping_weight=0.0)
        shaper.record_initial(td_initial)
        shaped, F = shaper.apply(base_reward, td_final)

        assert torch.allclose(shaped, base_reward, atol=1e-6)

    # --- Logging decomposition ------------------------------------------------

    def test_apply_returns_both_shaped_and_F(self):
        """apply() returns (shaped_reward, shaping_reward) as a 2-tuple."""
        td_initial = _make_td(collected=0.0, waste=[1.0])
        td_final = _make_td(collected=0.5, waste=[1.0])
        base_reward = torch.tensor([3.0])

        shaper = self._make_shaper()
        shaper.record_initial(td_initial)
        result = shaper.apply(base_reward, td_final)

        assert isinstance(result, tuple)
        assert len(result) == 2
        shaped, F = result
        assert shaped.shape == base_reward.shape
        assert F.shape == base_reward.shape

    # --- Safe guard: apply() before record_initial() --------------------------

    def test_apply_before_record_returns_zero_shaping(self):
        """Calling apply() without record_initial() is safe; returns zero F."""
        td_final = _make_td(collected=1.0, waste=[1.0])
        base_reward = torch.tensor([5.0])

        shaper = self._make_shaper()  # _phi_0 is None
        with patch("logic.src.pipeline.rl.common.pbrs_wrapper.logger") as mock_log:
            shaped, F = shaper.apply(base_reward, td_final)
            mock_log.warning.assert_called_once()

        assert torch.allclose(shaped, base_reward)
        assert torch.allclose(F, torch.zeros_like(base_reward))

    # --- Batch dimension handling ---------------------------------------------

    def test_batch_shaping_applies_per_instance(self):
        """Each instance in the batch gets its own shaping bonus."""
        B = 4
        [[1.0] * 4] * B     # total=4 per instance
        [0.0] * B
        collected_final = [1.0, 2.0, 3.0, 4.0]  # Φ = 0.25, 0.5, 0.75, 1.0

        td_initial = TensorDict(
            {
                "collected_waste": torch.zeros(B),
                "waste": torch.ones(B, 4),
            },
            batch_size=[B],
        )
        td_final = TensorDict(
            {
                "collected_waste": torch.tensor(collected_final),
                "waste": torch.ones(B, 4),
            },
            batch_size=[B],
        )
        base_reward = torch.zeros(B)

        shaper = self._make_shaper(gamma=1.0, shaping_weight=1.0)
        shaper.record_initial(td_initial)
        shaped, F = shaper.apply(base_reward, td_final)

        expected_F = torch.tensor([0.25, 0.5, 0.75, 1.0])
        assert torch.allclose(F, expected_F, atol=1e-5)
        assert torch.allclose(shaped, expected_F, atol=1e-5)  # base=0

    # --- repr -----------------------------------------------------------------

    def test_repr_contains_key_info(self):
        """__repr__ includes env name, gamma, and weight."""
        shaper = self._make_shaper(gamma=0.95, shaping_weight=2.0)
        r = repr(shaper)
        assert "ptp" in r
        assert "0.95" in r
        assert "2.0" in r

    # --- reset() --------------------------------------------------------------

    def test_reset_clears_phi0(self):
        """reset() sets _phi_0 back to None."""
        td = _make_td(collected=0.0, waste=[1.0])
        shaper = self._make_shaper()
        shaper.record_initial(td)
        assert shaper._phi_0 is not None
        shaper.reset()
        assert shaper._phi_0 is None
