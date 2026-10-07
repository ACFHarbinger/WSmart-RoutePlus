"""builder.py module.

Attributes:
    MODULE_VAR (Type): Description of module level variable.

Example:
    >>> import builder
"""

import contextlib
from dataclasses import asdict, is_dataclass
from typing import TYPE_CHECKING, Any, Dict, cast

import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

from logic.src.configs import Config
from logic.src.envs import get_env
from logic.src.interfaces import IEnv, ITraversable
from logic.src.pipeline.features.base import deep_sanitize, flatten_config_dict, remap_legacy_keys
from logic.src.pipeline.rl import REINFORCE, MetaRLModule
from logic.src.policies.vector import (
    AttentionModelPolicy,
    DeepDecoderPolicy,
    HybridTwoStagePolicy,
    MoEPolicy,
    NeuralHeuristicHybrid,
    PointerNetworkPolicy,
    SymNCOPolicy,
    TemporalAMPolicy,
    VectorizedALNS,
    VectorizedHGS,
    VectorizedHGSALNS,
)
from logic.src.policies.mandatory_selection.vectorized.factory import create_selector_from_config
from logic.src.tracking.logging.pylogger import get_pylogger

try:
    from logic.src.tracking.core.run import get_active_run
except ImportError:
    get_active_run = None  # type: ignore[assignment]

from .registry import _ALGO_REGISTRY

if TYPE_CHECKING:
    from logic.src.models.common.autoregressive.policy import AutoregressivePolicy

logger = get_pylogger(__name__)


def _resolve_configs(cfg: Any):
    """Helper to dynamically resolve env, policy, and model based on the active task."""
    task = getattr(cfg, "task", "train")
    task_cfg = getattr(cfg, task, cfg)
    env_cfg = getattr(task_cfg, "env", getattr(cfg, "env", None))
    policy_cfg = getattr(task_cfg, "policy", task_cfg)
    model_cfg = getattr(policy_cfg, "model", getattr(cfg, "model", None))
    return env_cfg, policy_cfg, model_cfg


def create_model(cfg: Config) -> pl.LightningModule:
    """Helper to create the RL model based on config.

    Args:
        cfg: Root configuration object.

    Returns:
        RL model.
    """
    # 1. Initialize Environment
    env = _init_environment(cfg)

    # 2. Initialize Policy
    policy = _init_policy(cfg, env)

    # 3. Initialize RL Module Keyword Arguments
    common_kwargs = _prepare_rl_kwargs(cfg, env, policy)

    # 4. Algorithm Factory
    algo_name = cfg.rl.algorithm
    factory = _ALGO_REGISTRY.get(algo_name)
    if factory is not None:
        model: pl.LightningModule = factory(cfg, policy, env, common_kwargs)
    else:
        model = REINFORCE(**common_kwargs)

    if getattr(cfg.meta_rl, "use_meta", False):
        model = MetaRLModule(
            agent=model,
            meta_lr=cfg.meta_rl.meta_lr,
            history_length=cfg.meta_rl.meta_history_length,
            hidden_size=cfg.meta_rl.meta_hidden_dim,
        )

    env_cfg, _, model_cfg = _resolve_configs(cfg)
    with contextlib.suppress(Exception):
        run = get_active_run() if get_active_run is not None else None
        if run is not None:
            params: Dict[str, Any] = {
                "model.name": getattr(model_cfg, "name", ""),
                "model.algo": algo_name,
                "env.name": getattr(env_cfg, "name", ""),
                "env.graph.num_loc": (
                    getattr(getattr(env_cfg, "graph", {}), "num_loc", None)
                    if env_cfg
                    else None
                    or getattr(
                        (getattr(env_cfg, "curriculum_graphs", [{}]) or [{}])[0] if env_cfg else {}, "num_loc", ""
                    )
                ),
            }
            if model_cfg and hasattr(model_cfg, "encoder"):
                enc = model_cfg.encoder
                params.update(
                    {
                        "model.embed_dim": getattr(enc, "embed_dim", ""),
                        "model.n_encode_layers": getattr(enc, "n_layers", ""),
                        "model.n_heads": getattr(enc, "n_heads", ""),
                        "model.normalization": enc.normalization.norm_type if hasattr(enc, "normalization") else "",
                    }
                )
            if model_cfg and hasattr(model_cfg, "decoder"):
                dec = model_cfg.decoder
                params["model.decoder_type"] = getattr(dec, "type", "")
                params["model.n_decode_layers"] = getattr(dec, "n_layers", "")
            if getattr(cfg.meta_rl, "use_meta", False):
                params["model.use_meta"] = True
                params["model.meta_lr"] = cfg.meta_rl.meta_lr
            run.log_params(params)

    return model


def _init_environment(cfg: Config) -> IEnv:
    """Initialize the environment based on config.

    Args:
        cfg: Root configuration object.

    Returns:
        Initialized environment.
    """
    env_cfg, _, _ = _resolve_configs(cfg)
    env_dict = _config_to_dict(env_cfg) if env_cfg else {}
    env_name = env_dict.get("name", "ptp")

    # Extract base env fields (excluding name and the structured sub-configs)
    _ENV_DICT_SKIP = {"name", "graph", "reward", "curriculum_graphs", "eval_graphs"}
    env_kwargs = {k: v for k, v in env_dict.items() if k not in _ENV_DICT_SKIP}

    # Flatten GraphConfig
    graph = env_dict.get("graph")
    if graph is None:
        # Fallback to primary curriculum entry if graph was not dynamically injected
        curriculum = env_dict.get("curriculum_graphs", [])
        graph = curriculum[0] if curriculum else {}
    graph_dict = graph if isinstance(graph, dict) else _config_to_dict(graph)
    env_kwargs.update(
        {
            "num_loc": graph_dict.get("num_loc", 50),
            "area": graph_dict.get("area", "riomaior"),
            "waste_type": graph_dict.get("waste_type", "plastic"),
            "vertex_method": graph_dict.get("vertex_method", "mmn"),
            "distance_method": graph_dict.get("distance_method", "ogd"),
            "dm_filepath": graph_dict.get("dm_filepath"),
            "edge_threshold": graph_dict.get("edge_threshold", "0"),
            "edge_method": graph_dict.get("edge_method"),
            "focus_graph": graph_dict.get("focus_graph"),
            "focus_size": graph_dict.get("focus_size"),
            "n_samples": graph_dict.get("n_samples", 1),
            "start_day": graph_dict.get("start_day", 0),
            "n_days": graph_dict.get("n_days", 1),
        }
    )

    # Flatten Reward/ObjectiveConfig.
    # Priority: env.reward (dynamically set by _build_stage_config for curriculum stages)
    # → env.graph.reward (per-graph reward defined in the graph entry)
    # → safe defaults (all weights = 1.0)
    reward_dict: Dict[str, Any] = {}
    if "graph" in env_dict:
        graph_reward = env_dict["graph"].get("reward") if isinstance(env_dict["graph"], dict) else None
        if graph_reward:
            reward_dict = graph_reward if isinstance(graph_reward, dict) else _config_to_dict(graph_reward)
    if "reward" in env_dict:
        # env.reward is injected dynamically by _build_stage_config for per-stage overrides
        stage_reward = env_dict["reward"]
        stage_reward_dict = stage_reward if isinstance(stage_reward, dict) else _config_to_dict(stage_reward)
        reward_dict.update(stage_reward_dict)
    if reward_dict:
        env_kwargs.update(
            {
                "cost_weight": reward_dict.get("cost_weight", 1.0),
                "waste_weight": reward_dict.get("waste_weight", 1.0),
                "overflow_penalty": reward_dict.get("overflow_penalty", 0.0),
            }
        )

    # Override distribution if specified in train config
    dist = getattr(cfg.train, "data_distribution", None)
    env_kwargs["data_distribution"] = dist if dist is not None else (env_kwargs.get("data_distribution") or "unif")
    env_kwargs["device"] = cfg.device
    env_kwargs["batch_size"] = cfg.train.batch_size

    return get_env(env_name, **env_kwargs)


def _init_policy(cfg: Config, env: Any):
    """Initialize the policy based on config.

    Args:
        cfg: Root configuration object.
        env: Initialized environment.

    Returns:
        Initialized policy.
    """
    env_cfg, _, model_cfg = _resolve_configs(cfg)

    policy_map = {
        "am": AttentionModelPolicy,
        "moe": MoEPolicy,
        "deep_decoder": DeepDecoderPolicy,
        "temporal": TemporalAMPolicy,
        "pointer": PointerNetworkPolicy,
        "symnco": SymNCOPolicy,
        "alns": VectorizedALNS,
        "hgs": VectorizedHGS,
        "hgs_alns": VectorizedHGSALNS,
        "hybrid": NeuralHeuristicHybrid,
        "hybrid_two_stage": HybridTwoStagePolicy,
    }

    if getattr(model_cfg, "name", "") == "hybrid":
        return _init_hybrid_policy(cfg)

    # Flatten ModelConfig for policy initialization
    policy_kwargs: Dict[str, Any] = {"env_name": getattr(env_cfg, "name", "ptp") if env_cfg else "ptp"}

    if model_cfg and hasattr(model_cfg, "encoder"):
        enc = model_cfg.encoder
        policy_kwargs.update(
            {
                "embed_dim": enc.embed_dim,
                "hidden_dim": enc.hidden_dim,
                "n_encode_layers": enc.n_layers,
                "n_heads": enc.n_heads,
                "normalization": enc.normalization.norm_type if hasattr(enc, "normalization") else "instance",
                "activation": enc.activation.name if hasattr(enc, "activation") else "relu",
                "norm_config": enc.normalization,
                "activation_config": enc.activation,
                "encoder_config": enc,
            }
        )

    if model_cfg and hasattr(model_cfg, "decoder"):
        dec = model_cfg.decoder
        policy_kwargs.update(
            {
                "n_decode_layers": dec.n_layers,
                "decoder_type": dec.type,
                "decoder_config": dec,
            }
        )

    # Copy other top-level model params
    if model_cfg:
        for k, v in vars(model_cfg).items():
            if k not in ["encoder", "decoder", "name"]:
                policy_kwargs[k] = v

    for key in ["lr_critic", "lr_critic_value"]:
        policy_kwargs.pop(key, None)

    return policy_map[getattr(model_cfg, "name", "am") if model_cfg else "am"](**policy_kwargs)


def _init_hybrid_policy(cfg: Config):
    """Special handling for hybrid construction-refinement policy.

    Args:
        cfg: Root configuration object.

    Returns:
        Initialized hybrid policy.
    """
    env_cfg, _, _ = _resolve_configs(cfg)

    # 1. Neural construction policy
    neural_cfg = cast(Any, cfg).copy()  # Simplified copy
    task = getattr(neural_cfg, "task", "train")
    task_cfg = getattr(neural_cfg, task, neural_cfg)
    policy_cfg = getattr(task_cfg, "policy", task_cfg)
    if hasattr(policy_cfg, "model"):
        policy_cfg.model.name = "am"

    neural_policy = cast("AutoregressivePolicy", create_model(cast(Config, neural_cfg)).policy)

    # 2. Heuristic refinement policy
    ref_strategy = getattr(cfg.rl, "refinement_strategy", "alns")
    ref_time = getattr(cfg.rl, "refinement_time_limit", 5.0)
    ref_iters = getattr(cfg.rl, "refinement_iterations", 500)
    max_v = getattr(cfg.rl, "max_vehicles", 0)
    env_name = getattr(env_cfg, "name", "ptp") if env_cfg else "ptp"

    if ref_strategy == "alns":
        heuristic_policy = VectorizedALNS(
            env_name=env_name, time_limit=ref_time, max_iterations=ref_iters, max_vehicles=max_v
        )
    else:
        heuristic_policy = VectorizedHGS(env_name=env_name, time_limit=ref_time, max_iterations=ref_iters)  # type: ignore[assignment]
    return NeuralHeuristicHybrid(neural_policy=neural_policy, heuristic_policy=heuristic_policy)


def _config_to_dict(obj: Any) -> Dict[str, Any]:
    """Convert config object (dataclass or DictConfig) to dict.

    Args:
        obj: Config object.

    Returns:
        Dictionary representation of config.
    """
    if is_dataclass(obj):
        return asdict(obj)  # type: ignore[arg-type]
    elif isinstance(obj, DictConfig):
        return cast(Dict[str, Any], OmegaConf.to_container(obj, resolve=True))
    elif isinstance(obj, dict):
        return obj
    else:
        # Fallback: try to wrap in OmegaConf.create and convert
        return cast(Dict[str, Any], OmegaConf.to_container(OmegaConf.create(obj), resolve=True))


def _prepare_rl_kwargs(cfg: Config, env: Any, policy: Any):
    """Prepare keyword arguments for RL module initialization.

    Args:
        cfg: Root configuration object.
        env: Initialized environment.
        policy: Initialized policy.

    Returns:
        Keyword arguments for RL module initialization.
    """
    env_cfg, policy_cfg, model_cfg = _resolve_configs(cfg)

    # Prepare base dicts
    common_kwargs: Dict[str, Any] = _config_to_dict(cfg.rl)

    task = getattr(cfg, "task", "train")
    if hasattr(cfg, task):
        train_params = _config_to_dict(getattr(cfg, task))
        train_params = flatten_config_dict(train_params)
        common_kwargs.update(train_params)

    # Dataset paths: prefer cfg.env.graph.load_dataset (set by env yamls), fall back to cfg.train.env
    _env_graph = getattr(env_cfg, "graph", None)
    load_dataset = getattr(_env_graph, "load_dataset", None)
    if load_dataset:
        common_kwargs["train_dataset_path"] = load_dataset
    if "val_dataset" in common_kwargs:
        common_kwargs["val_dataset_path"] = common_kwargs.pop("val_dataset")

    # Model params
    if model_cfg:
        model_params = _config_to_dict(model_cfg)
        model_params = flatten_config_dict(model_params)
        common_kwargs.update(model_params)
    common_kwargs.pop("name", None)

    # Inject core objects
    common_kwargs.update(
        {
            "env": env,
            "policy": policy,
            "optimizer": cfg.optim.optimizer,
            "optimizer_kwargs": {"lr": cfg.optim.lr, "weight_decay": cfg.optim.weight_decay},
            "lr_scheduler": cfg.optim.lr_scheduler,
        }
    )

    # Scheduler config
    sch_kwargs = cfg.optim.lr_scheduler_kwargs.copy()
    if cfg.optim.lr_decay != 1.0:
        sch_kwargs["gamma"] = cfg.optim.lr_decay
    if cfg.optim.lr_min_value != 0.0:
        sch_kwargs["eta_min"] = cfg.optim.lr_min_value
    common_kwargs["lr_scheduler_kwargs"] = sch_kwargs
    common_kwargs["baseline"] = cfg.rl.baseline
    common_kwargs["cfg"] = cfg  # Pass the full config for robust attribute access in RL module

    remap_legacy_keys(common_kwargs, cfg)
    common_kwargs = cast(Dict[str, Any], deep_sanitize(common_kwargs))

    # Algorithm specific overrides
    algo_name = cfg.rl.algorithm
    algo_specific_obj: object = common_kwargs.get(algo_name)
    if algo_name in common_kwargs and isinstance(algo_specific_obj, (dict, ITraversable)):
        algo_specific = cast(Dict[str, Any], algo_specific_obj)
        if algo_name == "ppo":
            common_kwargs["ppo_epochs"] = algo_specific.get("epochs", 10)
        elif algo_name == "sapo":
            common_kwargs.update(
                {"tau_pos": algo_specific.get("tau_pos", 0.1), "tau_neg": algo_specific.get("tau_neg", 1.0)}
            )
        common_kwargs.update(algo_specific)

    for key in ["lr_critic", "lr_critic_value"]:
        common_kwargs.pop(key, None)

    # Mandatory selector
    mandatory_selector = None
    mandatory_selection = getattr(policy_cfg, "mandatory_selection", getattr(cfg, "mandatory_selection", None))
    if mandatory_selection is not None:
        mandatory_selector = create_selector_from_config(mandatory_selection)
        if mandatory_selector:
            logger.info(f"Mandatory selector created: {mandatory_selection.strategy}")
    common_kwargs["mandatory_selector"] = mandatory_selector
    return common_kwargs
