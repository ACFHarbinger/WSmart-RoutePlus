"""
Base generator class for problem instances.

Attributes:
    Generator: Abstract base class for problem instance generators.

Example:
    >>> from src.envs.generators import get_generator
    >>> generator = get_generator("ptp", num_loc=20)
    >>> problem = generator.generate()
    >>> problem
    <ProblemInstance: ...>

    >>> generator = get_generator("atsp", num_loc=20, tmat_class=True)
    >>> problem = generator.generate()
    >>> problem
    <ProblemInstance: ...>
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Union

import numpy as np
import torch
from tensordict import TensorDict


class Generator(ABC):
    """
    Abstract base class for problem instance generators.

    Generators create batches of problem instances as TensorDict objects,
    supporting various probability distributions for feature generation.

    Attributes:
        num_loc (int): Number of nodes in the problem.
        min_loc (float): Minimum location value.
        max_loc (float): Maximum location value.
        loc_distribution (Union[str, Callable]): Distribution for location generation.
        device (torch.device): Device for computation.
        bins (Optional[int]): Number of bins for location distribution.
        kwargs: Additional keyword arguments.
        rng: Random number generator.
        generator: PyTorch generator for reproducibility.
    """

    def __init__(
        self,
        num_loc: int = 50,
        min_loc: float = 0.0,
        max_loc: float = 1.0,
        loc_distribution: Union[str, Callable] = "uniform",
        device: Union[str, torch.device] = "cpu",
        rng: Optional[np.random.Generator] = None,
        generator: Optional[torch.Generator] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the generator.



        Args:
            num_loc: Number of locations (excluding depot).
            min_loc: Minimum coordinate value.
            max_loc: Maximum coordinate value.
            loc_distribution: Distribution for location generation.
                Can be "uniform", "normal", "clustered", or a callable.
            device: Device to place tensors on.
            rng: Random number generator for numpy operations.
            generator: Random number generator for torch operations.
            kwargs: Additional keyword arguments.
        """
        self.num_loc = num_loc
        self.min_loc = min_loc
        self.max_loc = max_loc
        self.loc_distribution = loc_distribution
        self.device = torch.device(device)
        self.bins = kwargs.get("bins")
        self._kwargs = kwargs
        self.rng = np.random.default_rng() if rng is None else rng
        self.generator = generator if generator is not None else torch.Generator(device=device)

    @property
    def kwargs(self) -> dict[str, Any]:
        """Return all arguments used to initialize the generator.

        Returns:
            Description of return value.
        """
        return self._kwargs

    def to(self, device: Union[str, torch.device]) -> Generator:
        """
        Return a copy of the generator on the specified device.

        Args:
            device: Target device.

        Returns:
            New Generator instance on device.
        """
        # Get the class
        cls = self.__class__

        # We can use the variables stored in self.
        kwargs = self._kwargs.copy()
        kwargs.update(
            {
                "num_loc": self.num_loc,
                "min_loc": self.min_loc,
                "max_loc": self.max_loc,
                "loc_distribution": self.loc_distribution,
                "device": device,
            }
        )

        if hasattr(self, "area"):
            kwargs["area"] = self.area
        if hasattr(self, "data_dir"):
            kwargs["data_dir"] = self.data_dir
        if hasattr(self, "indices"):
            kwargs["indices"] = self.indices

        # For PTP:
        if hasattr(self, "min_waste"):
            kwargs.update(
                {
                    "min_waste": self.min_waste,  # type: ignore[attr-defined]
                    "max_waste": self.max_waste,  # type: ignore[attr-defined]
                    "waste_distribution": self.waste_distribution,  # type: ignore[attr-defined]
                    "capacity": self.capacity,  # type: ignore[attr-defined]
                    "depot_type": self.depot_type,  # type: ignore[attr-defined]
                }
            )


        # For IRP:
        if hasattr(self, "num_periods"):
            kwargs.update(
                {
                    "num_periods": self.num_periods,  # type: ignore[attr-defined]
                    "vehicle_capacity": self.vehicle_capacity,  # type: ignore[attr-defined]
                    "min_demand": self.min_demand,  # type: ignore[attr-defined]
                    "max_demand": self.max_demand,  # type: ignore[attr-defined]
                    "demand_distribution": self.demand_distribution,  # type: ignore[attr-defined]
                    "min_holding_cost": self.min_holding_cost,  # type: ignore[attr-defined]
                    "max_holding_cost": self.max_holding_cost,  # type: ignore[attr-defined]
                    "min_init_inventory": self.min_init_inventory,  # type: ignore[attr-defined]
                    "max_init_inventory": self.max_init_inventory,  # type: ignore[attr-defined]
                    "node_inventory_capacity": self.node_inventory_capacity,  # type: ignore[attr-defined]
                    "depot_type": self.depot_type,  # type: ignore[attr-defined]
                }
            )

        # For ATSP:
        if hasattr(self, "min_dist"):
            kwargs.update(
                {
                    "min_dist": self.min_dist,  # type: ignore[attr-defined]
                    "max_dist": self.max_dist,  # type: ignore[attr-defined]
                    "tmat_class": self.tmat_class,  # type: ignore[attr-defined]
                }
            )

        # For CVRP:
        if hasattr(self, "min_demand") and not hasattr(self, "num_periods"):
            kwargs.update(
                {
                    "min_demand": self.min_demand,  # type: ignore[attr-defined]
                    "max_demand": self.max_demand,  # type: ignore[attr-defined]
                    "vehicle_capacity": self.vehicle_capacity,  # type: ignore[attr-defined]
                    "capacity": self.capacity,  # type: ignore[attr-defined]
                    "depot_type": self.depot_type,  # type: ignore[attr-defined]
                }
            )

        # For OP:
        if hasattr(self, "prize_type") and not hasattr(self, "penalty_factor"):
            kwargs.update(
                {
                    "prize_type": self.prize_type,  # type: ignore[attr-defined]
                    "max_length": self.max_length,  # type: ignore[attr-defined]
                    "depot_type": self.depot_type,  # type: ignore[attr-defined]
                }
            )

        # For PCTSP / SPCTSP:
        if hasattr(self, "penalty_factor"):
            kwargs.update(
                {
                    "penalty_factor": self.penalty_factor,  # type: ignore[attr-defined]
                    "prize_required": self.prize_required,  # type: ignore[attr-defined]
                    "depot_type": self.depot_type,  # type: ignore[attr-defined]
                }
            )

        # For PDP:
        if hasattr(self, "depot_type") and not any(
            hasattr(self, a)
            for a in (
                "min_dist",
                "min_demand",
                "prize_type",
                "penalty_factor",
                "num_periods",
                "num_items_per_city",
            )
        ):
            kwargs.update({"depot_type": self.depot_type})  # type: ignore[attr-defined]

        # For ThOP:
        if hasattr(self, "num_items_per_city"):
            kwargs.update(
                {
                    "num_items_per_city": self.num_items_per_city,  # type: ignore[attr-defined]
                    "min_weight": self.min_weight,  # type: ignore[attr-defined]
                    "max_weight": self.max_weight,  # type: ignore[attr-defined]
                    "min_profit": self.min_profit,  # type: ignore[attr-defined]
                    "max_profit": self.max_profit,  # type: ignore[attr-defined]
                    "capacity": self.capacity,  # type: ignore[attr-defined]
                    "max_time": self.max_time,  # type: ignore[attr-defined]
                    "v_max": self.v_max,  # type: ignore[attr-defined]
                    "v_min": self.v_min,  # type: ignore[attr-defined]
                    "depot_type": self.depot_type,  # type: ignore[attr-defined]
                }
            )

        return cls(**kwargs)

    @abstractmethod
    def _generate(self, batch_size: tuple[int, ...]) -> TensorDict:
        """
        Generate a batch of problem instances.

        This method should be implemented by subclasses to create
        problem-specific instance data.

        Args:
            batch_size: Batch size(s) to generate.

        Returns:
            TensorDict containing the generated instances.
        """
        raise NotImplementedError

    def __call__(self, batch_size: Union[int, list[int], tuple[int, ...]] = 1) -> TensorDict:
        """
        Generate a batch of problem instances.

        Args:
            batch_size: Number of instances or batch size tuple to generate.

        Returns:
            TensorDict containing the generated instances.
        """
        batch_size = (batch_size,) if isinstance(batch_size, int) else tuple(batch_size)

        td = self._generate(batch_size)
        return td.to(self.device)

    def _generate_locations(self, batch_size: tuple[int, ...], num_loc: Optional[int] = None) -> torch.Tensor:
        """
        Generate location coordinates based on the specified distribution.

        Args:
            batch_size: Batch size tuple.
            num_loc: Number of locations (uses self.num_loc if None).

        Returns:
            Tensor of shape (*batch_size, num_loc, 2) containing coordinates.
        """
        num_loc = num_loc or self.num_loc

        if callable(self.loc_distribution):
            return self.loc_distribution(batch_size, num_loc)

        if self.loc_distribution == "uniform":
            return self._uniform_locations(batch_size, num_loc)
        elif self.loc_distribution == "normal":
            return self._normal_locations(batch_size, num_loc)
        elif self.loc_distribution == "clustered":
            return self._clustered_locations(batch_size, num_loc)
        else:
            raise ValueError(f"Unknown location distribution: {self.loc_distribution}")

    def _uniform_locations(self, batch_size: tuple[int, ...], num_loc: int) -> torch.Tensor:
        """Generate uniformly distributed locations.

        Args:
            batch_size: Batch size tuple.
            num_loc: Number of locations.

        Returns:
            Tensor of shape (*batch_size, num_loc, 2) containing uniformly distributed coordinates.
        """
        return (
            torch.rand(*batch_size, num_loc, 2, device=self.device, generator=self.generator)
            * (self.max_loc - self.min_loc)
            + self.min_loc
        )

    def _normal_locations(self, batch_size: tuple[int, ...], num_loc: int) -> torch.Tensor:
        """Generate normally distributed locations (clipped to bounds).

        Args:
            batch_size: Batch size tuple.
            num_loc: Number of locations.

        Returns:
            Tensor of shape (*batch_size, num_loc, 2) containing normally distributed coordinates.
        """
        center = (self.max_loc + self.min_loc) / 2
        std = (self.max_loc - self.min_loc) / 6  # ~99.7% within bounds
        locs = torch.randn(*batch_size, num_loc, 2, device=self.device, generator=self.generator) * std + center
        return torch.clamp(locs, self.min_loc, self.max_loc)

    def _clustered_locations(self, batch_size: tuple[int, ...], num_loc: int, num_clusters: int = 3) -> torch.Tensor:
        """Generate clustered locations using Gaussian mixture.

        Args:
            batch_size: Batch size tuple.
            num_loc: Number of locations.
            num_clusters: Number of clusters.

        Returns:
            Tensor of shape (*batch_size, num_loc, 2) containing clustered coordinates.
        """
        num_clusters = self._kwargs.get("num_clusters", num_clusters)

        # Generate cluster centers
        centers = self._uniform_locations(batch_size, num_clusters)

        # Assign points to clusters
        cluster_assignments = torch.randint(
            0, num_clusters, (*batch_size, num_loc), device=self.device, generator=self.generator
        )

        # Generate points around cluster centers
        std = (self.max_loc - self.min_loc) / 10
        offsets = torch.randn(*batch_size, num_loc, 2, device=self.device, generator=self.generator) * std

        # Get assigned cluster centers
        indices = cluster_assignments.unsqueeze(-1).expand(*batch_size, num_loc, 2)
        locs = torch.gather(centers, -2, indices)

        locs = locs + offsets
        return torch.clamp(locs, self.min_loc, self.max_loc)
