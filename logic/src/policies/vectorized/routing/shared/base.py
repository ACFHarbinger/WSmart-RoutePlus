"""Common base of the vectorised routing policies.

Every vectorised policy is configured by the **same dataclass as its simulator
counterpart** (``logic.src.configs.policies``): the constructor takes that config (a
dataclass instance or a mapping such as a Hydra node) and/or its fields as keyword
arguments, with the same names and defaults. Fields that only matter inside the
simulator (``packing``, ``route_improvement``, ``engine`` variants backed by external
packages, mandatory-bin handling) are accepted and ignored.

Attributes:
    VectorizedRoutingPolicy: Base class (config loading, I/O, seeding).
"""

from __future__ import annotations

import dataclasses
import random
from typing import Any, ClassVar, Dict, Optional, Type

import torch
from tensordict import TensorDict

from logic.src.models.common.improvement.policy import ImprovementPolicy
from logic.src.tracking.viz_mixin import PolicyVizMixin

from . import routes as rs
from .vrpp import VRPPBatch, parse_instance, replay_reward, routes_to_actions

#: Config fields that only drive simulator-only features; the vectorised policies ignore them.
SIMULATION_ONLY_FIELDS = ("packing", "route_improvement", "engine")


def _has_raw_mappings(config: Any) -> bool:
    """Whether a config instance holds raw mappings (e.g. nested Hydra ``DictConfig``s) in dataclass fields."""
    from omegaconf import DictConfig

    return any(
        isinstance(getattr(config, f.name), DictConfig)
        or (isinstance(getattr(config, f.name), dict) and _is_dataclass_field(f))
        for f in dataclasses.fields(config)
    )


def _is_dataclass_field(f: Any) -> bool:
    factory = f.default_factory
    if factory is dataclasses.MISSING:
        return dataclasses.is_dataclass(f.default)
    try:
        return dataclasses.is_dataclass(factory())
    except Exception:
        return False


def load_config(config_cls: Type, config: Any = None, **overrides: Any) -> Any:
    """Build a ``config_cls`` instance from a dataclass, a mapping and/or field overrides.

    Mappings are loaded strictly with the simulator's own loader (unknown keys fail).

    Args:
        config_cls: Simulator config dataclass.
        config: Instance of ``config_cls``, mapping (dict/DictConfig) or None.
        overrides: Field values.

    Returns:
        The config instance.
    """
    from omegaconf import DictConfig, OmegaConf

    from logic.src.policies.routing.base.base_routing_policy import _structured_load

    if config is None and not overrides:
        return config_cls()
    if isinstance(config, config_cls) and not overrides and not _has_raw_mappings(config):
        return config
    if config is None:
        mapping: Dict[str, Any] = {}
    elif dataclasses.is_dataclass(config) and not isinstance(config, type):
        mapping = {f.name: getattr(config, f.name) for f in dataclasses.fields(config)}
    elif isinstance(config, DictConfig):
        mapping = dict(OmegaConf.to_container(config, resolve=True))  # type: ignore[arg-type]
    elif isinstance(config, dict):
        mapping = dict(config)
    else:
        raise TypeError(f"Cannot build {config_cls.__name__} from {type(config).__name__}")
    mapping.pop("_target_", None)
    mapping.update(overrides)
    seed = mapping.pop("seed", None)
    cfg = _structured_load(config_cls, mapping)
    if seed is not None and "seed" in {f.name for f in dataclasses.fields(config_cls)}:
        cfg = dataclasses.replace(cfg, seed=seed)
    return cfg


class VectorizedRoutingPolicy(ImprovementPolicy, PolicyVizMixin):
    """Base of the vectorised policies: config, seeding, instance parsing and outputs.

    Subclasses set :attr:`config_cls` and implement :meth:`solve`.

    Attributes:
        cfg: The simulator config dataclass instance.
        expand_pool: Repair operators may insert any unvisited bin (VRPP mode, as in the
            simulator's VRPP runs).
    """

    config_cls: ClassVar[Type]
    expand_pool: ClassVar[bool] = True

    def __init__(
        self,
        env_name: Optional[str] = None,
        config: Any = None,
        seed: Optional[int] = None,
        device: str = "cpu",
        **params: Any,
    ) -> None:
        """Load the config and seed the random generators.

        Args:
            env_name: Environment name.
            config: Simulator config (dataclass or mapping); fields in ``params`` override it.
            seed: Random seed; overrides the config's ``seed``. Defaults to the config seed or 42.
            device: Torch device.
            params: Config fields.
        """
        params.pop("_target_", None)
        cfg = load_config(self.config_cls, config, **params)
        cfg_seed = getattr(cfg, "seed", None)
        eff_seed = seed if seed is not None else (cfg_seed if cfg_seed is not None else 42)
        super().__init__(env_name=env_name, seed=int(eff_seed), device=device)
        self.cfg = cfg

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def solve(self, inst: VRPPBatch, initial: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Solve the batch; return route sequences ``[B, L]``.

        Args:
            inst: Instances.
            initial: Optional starting route sequences.

        Returns:
            torch.Tensor: Best route sequences.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Shared machinery
    # ------------------------------------------------------------------

    def _sync_device(self, device: torch.device) -> None:
        if str(device) != str(self.generator.device):
            self.device = torch.device(device)
            self.generator = torch.Generator(device=device).manual_seed(self.seed)

    def _py_rng(self) -> random.Random:
        return self.rng

    def forward(
        self,
        td: TensorDict,
        env: Optional[Any] = None,
        strategy: str = "greedy",
        num_starts: int = 1,
        max_steps: Optional[int] = None,
        phase: str = "train",
        return_actions: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Solve every instance of ``td``.

        Args:
            td: Reset environment TensorDict.
            env: Environment (objective weights, reward replay).
            strategy: Unused (the experts search, they do not decode).
            num_starts: Unused.
            max_steps: Unused.
            phase: Unused.
            return_actions: Unused; actions are always returned.
            kwargs: ``initial_solution`` (route sequences or giant tours) is used as the
                starting solution when given; other decoding arguments are ignored.

        Returns:
            Dict with ``actions``, ``reward``, ``profit``, ``cost`` and a zero ``log_likelihood``.
        """
        inst = parse_instance(td, env)
        self._sync_device(inst.device)
        init = kwargs.get("initial_solution")
        S0 = None
        if init is not None:
            init = init.to(inst.device).long()
            S0 = rs.from_giant_tours(init[:, (init != 0).any(dim=0)] if init.dim() == 2 else init, inst)
        S = self.solve(inst, S0)
        routes = rs.to_route_lists(S)
        actions = routes_to_actions(routes, inst.device)
        profit = rs.seq_profit(S, inst)
        cost = rs.seq_cost(S, inst)
        B = td.batch_size[0]
        return {
            "actions": actions,
            "reward": replay_reward(td, env, actions, profit),
            "profit": profit,
            "cost": cost,
            "routes": routes,
            "log_likelihood": torch.zeros(B, device=inst.device),
        }
