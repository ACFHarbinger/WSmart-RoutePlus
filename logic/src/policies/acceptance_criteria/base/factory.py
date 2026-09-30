"""Factory for move acceptance criteria.

Provides a centralized mechanism to instantiate registered criteria by name.

Attributes:
    AcceptanceCriterionFactory: Factory for creating move acceptance criteria.

Example:
    >>> from logic.src.policies.acceptance_criteria.base import AcceptanceCriterionFactory
    >>> factory = AcceptanceCriterionFactory()
    >>> criterion = factory.create("boltzmann_metropolis_criterion", p0=0.5, window_size=100, alpha=0.95)
    >>> criterion.accept(100, 98)
    True, {'accepted': True, 'delta': -2, 'temperature': 0.0, 'sigma': 0.0, 'window_len': 0}
"""

from typing import Any, Dict, Optional, Union

from logic.src.interfaces.acceptance_criterion import IAcceptanceCriterion

from .registry import AcceptanceCriterionRegistry


class AcceptanceCriterionFactory:
    """Factory for creating move acceptance criteria.

    Integrates with the configuration system to provide solver-specific
    move acceptance logic.

    Attributes:
        _registered (bool): Flag indicating if modules have been loaded.
    """

    _registered = False

    @classmethod
    def ensure_registered(cls) -> None:
        """Imports all criteria modules to trigger registration."""
        if cls._registered:
            return

        import contextlib
        import importlib

        # List of modular acceptance criteria implementations to load
        modules = [
            "adaptive_boltzmann_metropolis",
            "all_moves_accepted",
            "aspiration_criterion",
            "binary_tournament_acceptance",
            "boltzmann_metropolis_criterion",
            "demon_algorithm",
            "ensemble_move_acceptance",
            "epsilon_dominance",
            "exponential_monte_carlo_counter",
            "fitness_proportional",
            "generalized_tsallis_simulated_annealing",
            "great_deluge",
            "improving_and_equal",
            "late_acceptance_hill_climbing",
            "monte_carlo",
            "non_linear_great_deluge",
            "old_bachelor_acceptance",
            "only_improving",
            "pareto_dominance",
            "probabilistic_transition",
            "record_to_record_travel",
            "skewed_variable_neighborhood_search",
            "step_counting_hill_climbing",
            "threshold_accepting",
        ]

        for mod in modules:
            with contextlib.suppress(ImportError):
                importlib.import_module(f"logic.src.policies.acceptance_criteria.{mod}")

        cls._registered = True

    @classmethod
    def create(
        cls,
        name: str,
        config: Optional[Union[Dict[str, Any], Any]] = None,
        **kwargs: Any,
    ) -> IAcceptanceCriterion:
        """Instantiate an acceptance criterion.

        Args:
            name (str): Name of the acceptance criterion.
            config (Optional[Union[Dict[str, Any], Any]]): Configuration for the acceptance criterion.
            kwargs (Any): Additional keyword arguments for the acceptance criterion.

        Returns:
            IAcceptanceCriterion: The instantiated acceptance criterion.

        Raises:
            ValueError: If no criterion is registered under the given name.
        """
        cls.ensure_registered()

        criterion_cls = AcceptanceCriterionRegistry.get(name)
        if not criterion_cls:
            raise ValueError(f"Unknown acceptance criterion: {name}")

        # Extract parameters from config if it's a dict or dataclass
        params: Dict[str, Any] = {}
        if config is not None:
            if isinstance(config, dict):
                params = config.copy()
            elif hasattr(config, "__dict__"):
                # Handle dataclasses or simple objects
                params = {k: v for k, v in config.__dict__.items() if not k.startswith("_")}

        # Merge with kwargs (kwargs take precedence for dynamic injection)
        params.update(kwargs)

        # Instantiate. We use filter_kwargs pattern if needed, or rely on constructor signatures.
        # For now, we assume constructors are compatible with the params they receive.
        inst = criterion_cls(**params)
        consumed: Dict[str, Any] = {"name": name, "params": dict(params)}
        if hasattr(inst, "T"):
            consumed["instance_T"] = inst.T
        if hasattr(inst, "alpha"):
            consumed["instance_alpha"] = inst.alpha
        try:
            from logic.src.pipeline.simulations.actions.base import _record_live_params

            _record_live_params("consumer_acceptance", consumed)
        except Exception:
            pass
        return inst
