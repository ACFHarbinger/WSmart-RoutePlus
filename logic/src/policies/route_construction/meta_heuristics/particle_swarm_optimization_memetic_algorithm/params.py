"""
Configuration parameters for the Particle Swarm Optimization Memetic Algorithm (PSOMA).

Attributes:
    PSOMAParams: Dataclass holding PSOMA solver hyperparameters.

Example:
    >>> from logic.src.policies.route_construction.meta_heuristics.particle_swarm_optimization_memetic_algorithm.params import PSOMAParams
    >>> params = PSOMAParams(pop_size=20, omega=0.1, c1=1.5, c2=2.0)
    >>> print(params.max_iterations)
    200
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Optional

from logic.src.interfaces.acceptance_criterion import IAcceptanceCriterion


@dataclass
class PSOMAParams:
    """
    Configuration parameters for the PSOMA-Split solver.

    Attributes:
        pop_size (int): Number of particles in the swarm.
        omega (float): Inertia weight.
        c1 (float): Cognitive coefficient.
        c2 (float): Social coefficient.
        max_iterations (int): Maximum number of iterations.
        x_min (float): Minimum value for continuous position vector.
        x_max (float): Maximum value for continuous position vector.
        v_min (float): Minimum value for continuous velocity vector.
        v_max (float): Maximum value for continuous velocity vector.
        L (int): If gbest keeps fixed at consecutive L steps, then output the best solution.
        T0 (float): Initial temperature for simulated annealing.
        lambda_cooling (float): Cooling rate for simulated annealing.
        time_limit (float): Time limit in seconds.
        vrpp (bool): Whether to use VRPP mode.
        profit_aware_operators (bool): Whether to use profit-aware operators.
        seed (Optional[int]): Random seed.
        acceptance_criterion (Optional[IAcceptanceCriterion]): Acceptance criterion.
    """

    pop_size: int = 20
    omega: float = 1.0
    c1: float = 2.0
    c2: float = 2.0
    max_iterations: int = 200
    x_min: float = 0.0
    x_max: float = 4.0
    v_min: float = -4.0
    v_max: float = 4.0
    L: int = 30
    T0: float = 3.0
    lambda_cooling: float = 0.9
    time_limit: float = 60.0
    sa_steps: int = 5000
    seed_nearest_neighbour: bool = True
    vrpp: bool = True
    profit_aware_operators: bool = False
    seed: Optional[int] = 42
    acceptance_criterion: Optional[IAcceptanceCriterion] = None

    @classmethod
    def from_config(cls, config: Any) -> PSOMAParams:
        """Create PSOMAParams from a configuration source.

        Args:
            config: Configuration object or dictionary.

        Returns:
            PSOMAParams: Configured PSOMAParams instance.
        """
        if isinstance(config, dict):
            params = cls(**{k: v for k, v in config.items() if k in {f.name for f in fields(cls)}})
        else:
            params = cls(
                pop_size=getattr(config, "pop_size", 20),
                omega=getattr(config, "omega", 1.0),
                c1=getattr(config, "c1", 2.0),
                c2=getattr(config, "c2", 2.0),
                max_iterations=getattr(config, "max_iterations", 200),
                x_min=getattr(config, "x_min", 0.0),
                x_max=getattr(config, "x_max", 4.0),
                v_min=getattr(config, "v_min", -4.0),
                v_max=getattr(config, "v_max", 4.0),
                L=getattr(config, "L", 30),
                T0=getattr(config, "T0", 3.0),
                lambda_cooling=getattr(config, "lambda_cooling", 0.9),
                time_limit=getattr(config, "time_limit", 60.0),
                sa_steps=getattr(config, "sa_steps", 5000),
                seed_nearest_neighbour=getattr(config, "seed_nearest_neighbour", True),
                vrpp=getattr(config, "vrpp", True),
                seed=getattr(config, "seed", 42),
            )

        # Handle Acceptance Criterion Injection
        from logic.src.policies.acceptance_criteria.base.factory import AcceptanceCriterionFactory

        acceptance_cfg = (
            getattr(config, "acceptance_criterion", None)
            if not isinstance(config, dict)
            else config.get("acceptance_criterion")
        )

        if acceptance_cfg and hasattr(acceptance_cfg, "method"):
            params.acceptance_criterion = AcceptanceCriterionFactory.create(
                name=acceptance_cfg.method,
                config=acceptance_cfg.params,
            )
        else:
            # Default to the classic Boltzmann-Metropolis criterion using PSOMA's SA parameters
            params.acceptance_criterion = AcceptanceCriterionFactory.create(
                name="bmc",
                initial_temp=params.T0,
                alpha=params.lambda_cooling,
                seed=params.seed,
            )

        return params
