"""
Configuration parameters for Hybrid Genetic Search.

Attributes:
    HGSParams: Dataclass for configuration parameters.

Example:
    >>> params = HGSParams(mu=25, lambda_param=40)
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, Optional

from logic.src.interfaces.acceptance_criterion import IAcceptanceCriterion

if TYPE_CHECKING:
    pass


@dataclass
class HGSParams:
    """
    Configuration parameters for Hybrid Genetic Search.
    Based on Vidal et al. (2022) - "Hybrid genetic search for the CVRP".

    Attributes:
        restart_timer: Maximum wall-clock seconds for optimization, before restarting the algorithm (0 = unlimited)
        time_limit: Maximum search time in seconds (0 = unlimited).
        mu: Minimum population size (for each subpopulation).
        lambda_param: Generation size - number of individuals before survivor selection.
        nb_elite: Number of elite individuals to preserve.
        nb_close: Number of close individuals for diversity measurement.
        nb_granular: Granular search parameter for local search moves.
        target_feasible: Target proportion of feasible solutions (e.g., 0.2 = 20%).
        n_iterations_no_improvement: Max iterations without improvement before stopping.
        restart_timer: Iterations without improvement before restarting (used only when time_limit > 0). Set to 0 to disable restarts.
        mutation_rate: Probability of applying local search (education) to offspring.
        repair_probability: Probability of repairing infeasible offspring (default 0.5).
        crossover_rate: Probability of applying crossover.
        local_search_iterations: Number of iterations for local search.
        use_cross_exchange: Whether to use cross exchange moves.
        use_lambda_interchange: Whether to use lambda interchange moves.
        lambda_max: Maximum lambda for lambda interchange moves.
        use_ejection_chains: Whether to use ejection chain moves.
        use_3opt: Whether to use 3-opt moves.
        max_vehicles: Maximum number of vehicles allowed (0 = unlimited).
        initial_penalty_capacity: Initial penalty coefficient for capacity violations.
        penalty_increase: Multiplier for increasing penalty (when too many feasible).
        penalty_decrease: Multiplier for decreasing penalty (when too many infeasible).
        engine: Engine to use for the solver.

    Example:
        >>> params = HGSParams(mu=25)
    """

    # Core HGS parameters (Vidal 2022)
    restart_timer: float = 0.0
    time_limit: float = 0.0  # 0 = no time limit
    mu: int = 25  # Minimum population size per subpopulation
    n_offspring: int = 40  # Generation size (number of individuals before survivor selection)
    nb_elite: int = 4  # Number of elite individuals
    nb_close: int = 5  # Number of close individuals for diversity
    nb_granular: int = 20  # Granular search parameter
    target_feasible: float = 0.2  # Target 20% feasible solutions
    n_iterations_no_improvement: int = 20000  # Stopping criterion

    # Genetic operators
    mutation_rate: float = 1.0  # Always educate offspring with local search
    repair_probability: float = 0.5  # 50% chance to repair infeasible offspring
    crossover_rate: float = 1.0  # Always apply crossover

    # Local search
    local_search_iterations: int = 500
    max_vehicles: int = 0

    # Penalty management
    initial_penalty_capacity: float = 1.0
    penalty_increase: float = 1.2
    penalty_decrease: float = 0.85
    use_3opt: bool = False
    use_cross_exchange: bool = False
    use_lambda_interchange: bool = False
    lambda_max: int = 0
    use_ejection_chains: bool = False

    # Infrastructure
    seed: Optional[int] = None
    vrpp: bool = True
    engine: str = "custom"
    profit_aware_operators: bool = False
    acceptance_criterion: Optional[IAcceptanceCriterion] = None

    @classmethod
    def from_config(cls, config: Any) -> "HGSParams":
        """Create HGSParams from a HGSConfig dataclass.

        Args:
            config: HGSConfig dataclass with solver parameters.

        Returns:
            HGSParams instance with values from config.
        """

        # Both simulator dicts and typed configs must use the same defaults.
        def get(name: str, default: Any = None) -> Any:
            return config.get(name, default) if isinstance(config, Mapping) else getattr(config, name, default)

        # Map config parameters to HGSParams, using defaults for new parameters
        params = cls(
            time_limit=get("time_limit", 0.0),
            mu=get("mu", 25),
            n_offspring=get("n_offspring", get("lambda_param", 40)),
            nb_elite=get("nb_elite", 4),
            nb_close=get("nb_close", 5),
            nb_granular=get("nb_granular", 20),
            target_feasible=get("target_feasible", 0.2),
            n_iterations_no_improvement=get("n_iterations_no_improvement", 20000),
            mutation_rate=get("mutation_rate", 1.0),
            repair_probability=get("repair_probability", 0.5),
            crossover_rate=get("crossover_rate", 1.0),
            local_search_iterations=get("local_search_iterations", 500),
            max_vehicles=get("max_vehicles", 0),
            initial_penalty_capacity=get("initial_penalty_capacity", 1.0),
            penalty_increase=get("penalty_increase", 1.2),
            penalty_decrease=get("penalty_decrease", 0.85),
            use_3opt=get("use_3opt", False),
            use_cross_exchange=get("use_cross_exchange", False),
            use_lambda_interchange=get("use_lambda_interchange", False),
            lambda_max=get("lambda_max", 0),
            use_ejection_chains=get("use_ejection_chains", False),
            vrpp=get("vrpp", True),
            profit_aware_operators=get("profit_aware_operators", False),
            seed=get("seed", 42),
            engine=get("engine", "custom"),
            restart_timer=get("restart_timer", 0.0),
        )

        # Handle Acceptance Criterion Injection
        from logic.src.policies.acceptance_criteria.base.factory import AcceptanceCriterionFactory

        acceptance_cfg = get("acceptance_criterion", None)
        if acceptance_cfg:
            params.acceptance_criterion = AcceptanceCriterionFactory.create(
                name=acceptance_cfg.get("method") if isinstance(acceptance_cfg, dict) else acceptance_cfg.method,
                config=acceptance_cfg.get("params") if isinstance(acceptance_cfg, dict) else acceptance_cfg.params,
            )
        else:
            # Default to only_improving for standard HGS
            params.acceptance_criterion = AcceptanceCriterionFactory.create(name="oi")

        return params

    @property
    def lambda_param(self) -> int:
        """
        Alias for n_offspring (Vidal 2022 terminology).

        Returns:
            int: The number of offspring per generation.
        """
        return self.n_offspring

    @lambda_param.setter
    def lambda_param(self, value: int):
        """Sets the number of offspring per generation.

        Args:
            value (int): New generation size.

        Returns:
            None.
        """
        self.n_offspring = value

    @property
    def population_size(self) -> int:
        """
        Alias for mu (common evolutionary terminology).

        Returns:
            int: The minimum population size per subpopulation.
        """
        return self.mu

    @population_size.setter
    def population_size(self, value: int):
        """Sets the minimum population size per subpopulation.

        Args:
            value (int): New minimum population size.

        Returns:
            None.
        """
        self.mu = value

    @property
    def elite_size(self) -> int:
        """
        Alias for nb_elite.

        Returns:
            int: The number of elite individuals.
        """
        return self.nb_elite

    @elite_size.setter
    def elite_size(self, value: int):
        """Sets the number of elite individuals to preserve.

        Args:
            value (int): New elite size.

        Returns:
            None.
        """
        self.nb_elite = value

    @property
    def no_improvement_threshold(self) -> int:
        """
        Alias for n_iterations_no_improvement.

        Returns:
            int: The stop threshold.
        """
        return self.n_iterations_no_improvement

    @no_improvement_threshold.setter
    def no_improvement_threshold(self, value: int):
        """Sets the stopping criterion iteration threshold.

        Args:
            value (int): Max iterations without improvement.

        Returns:
            None.
        """
        self.n_iterations_no_improvement = value

    @property
    def neighbor_list_size(self) -> int:
        """
        Alias for nb_granular.

        Returns:
            int: The neighbor list size.
        """
        return self.nb_granular

    @neighbor_list_size.setter
    def neighbor_list_size(self, value: int):
        """Sets the granular search neighbor list size.

        Args:
            value (int): New neighbor list size.

        Returns:
            None.
        """
        self.nb_granular = value
