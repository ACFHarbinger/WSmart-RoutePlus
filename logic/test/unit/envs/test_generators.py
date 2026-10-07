"""Tests for Instance Generators."""

import pytest
import torch
from logic.src.envs.generators import (
    TSPGenerator,
    VRPPGenerator,
    get_generator,
)

pytestmark = [pytest.mark.unit, pytest.mark.fast]



class TestGeneratorsBase:
    """Tests for the base Generator class logic."""

    def test_generator_to_device(self):
        """Test the 'to' method for device switching."""
        gen = VRPPGenerator(num_loc=10, device="cpu")
        gen_cuda = gen.to("cpu")  # Still CPU as we don't assume CUDA in tests
        assert gen_cuda.device.type == "cpu"
        assert gen_cuda.num_loc == 10

    def test_generator_call(self):
        """Test the __call__ method for generating batches."""
        gen = VRPPGenerator(num_loc=5, device="cpu")
        td = gen(batch_size=(2, 3))
        assert td.batch_size == (2, 3)
        assert td["locs"].shape == (2, 3, 5, 2)

    def test_location_distributions(self):
        """Test different location distributions."""
        # Normal
        gen_normal = VRPPGenerator(num_loc=10, loc_distribution="normal")
        locs_normal = gen_normal._generate_locations((2,))
        assert locs_normal.shape == (2, 10, 2)

        # Clustered
        gen_clustered = VRPPGenerator(num_loc=10, loc_distribution="clustered", num_clusters=2)
        locs_clustered = gen_clustered._generate_locations((2,))
        assert locs_clustered.shape == (2, 10, 2)

        # Callable
        def my_dist(bs, n):
            return torch.ones(*bs, n, 2)

        gen_callable = VRPPGenerator(num_loc=10, loc_distribution=my_dist)
        locs_callable = gen_callable._generate_locations((2,))
        assert torch.all(locs_callable == 1.0)

    def test_invalid_distribution(self):
        """Test error raise for unknown distribution."""
        gen = VRPPGenerator(loc_distribution="invalid")
        with pytest.raises(ValueError, match="Unknown location distribution"):
            gen._generate_locations((1,))


class TestVRPPGenerator:
    """Tests for VRPPGenerator specific logic."""

    def test_depot_types(self):
        """Test center, corner, and random depot placement."""
        # Center
        gen_c = VRPPGenerator(depot_type="center", min_loc=0, max_loc=1)
        td_c = gen_c(1)
        assert torch.all(td_c["depot"] == 0.5)

        # Corner
        gen_cor = VRPPGenerator(depot_type="corner", min_loc=0)
        td_cor = gen_cor(1)
        assert torch.all(td_cor["depot"] == 0.0)

        # Random
        gen_r = VRPPGenerator(depot_type="random")
        td_r = gen_r(1)
        assert td_r["depot"].shape == (1, 2)

    def test_waste_distributions(self):
        """Test uniform and gamma waste distributions."""
        # Gamma - use 'gamma' or 'gamma1'
        gen_g = VRPPGenerator(waste_distribution="gamma1", waste_alpha=2.0, waste_beta=0.5)
        td_g = gen_g(10)
        assert td_g["waste"].shape == (10, 50)
        assert torch.all(td_g["waste"] >= 0)

        # Uniform
        gen_u = VRPPGenerator(waste_distribution="uniform")
        td_u = gen_u(10)
        assert td_u["waste"].shape == (10, 50)

    def test_waste_distance_distribution(self):
        """Test distance correlated waste."""
        gen = VRPPGenerator(waste_distribution="dist")
        td = gen(5)
        assert td["waste"].shape == (5, 50)


class TestTSPGenerator:
    """Tests for TSPGenerator."""

    def test_tsp_generation(self):
        """Test TSP instance structure."""
        gen = TSPGenerator(num_loc=20)
        td = gen(2)
        assert td["locs"].shape == (2, 20, 2)
        assert td["depot"].shape == (2, 2)


def test_get_generator():
    """Test the factory function."""
    gen = get_generator("vrpp", num_loc=15)
    assert isinstance(gen, VRPPGenerator)
    assert gen.num_loc == 15

    with pytest.raises(ValueError, match="Unknown generator"):
        get_generator("invalid_env")
