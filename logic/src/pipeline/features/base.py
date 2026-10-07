"""
Shared utilities and base classes for pipeline features.

Attributes:
    deep_sanitize: Recursively convert OmegaConf objects and other types to primitive types.
    remap_legacy_keys: Remap config keys to legacy names for simulation compatibility.
    flatten_config_dict: Flatten nested 'graph', 'reward', 'decoding', 'model', and 'policy' configs into the main dictionary.

Example:
    >>> from logic.src.configs import Config
    >>> from logic.src.pipeline.features.base import deep_sanitize, remap_legacy_keys, flatten_config_dict
    >>>
    >>> # Example for deep_sanitize
    >>> cfg = Config()
    >>> sanitized = deep_sanitize(cfg)
    >>>
    >>> # Example for remap_legacy_keys
    >>> common_kwargs = {}
    >>> remap_legacy_keys(common_kwargs, cfg)
    >>>
    >>> # Example for flatten_config_dict
    >>> flattened = flatten_config_dict(cfg)
"""

from typing import Any, Dict

from omegaconf import DictConfig, ListConfig, OmegaConf

from logic.src.interfaces import ITraversable
from logic.src.tracking.logging.pylogger import get_pylogger

logger = get_pylogger(__name__)


def deep_sanitize(obj: Any) -> Any:
    """
    Recursively convert OmegaConf objects and other types to primitive types.

    Args:
        obj: The object to sanitize.

    Returns:
        The sanitized object.
    """
    if isinstance(obj, (DictConfig, ListConfig)):
        obj = OmegaConf.to_container(obj, resolve=True)

    # Use ITraversable protocol for dict-like objects
    if isinstance(obj, ITraversable):
        return {str(k): deep_sanitize(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [deep_sanitize(v) for v in obj]
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    # Do not stringify complex objects (modules, solvers, states)
    return obj


def remap_legacy_keys(common_kwargs: Dict[str, Any], cfg: Any) -> None:
    """Remap config keys to legacy names for simulation compatibility.

    Args:
        common_kwargs: Dictionary to store remapped keys.
        cfg: Configuration object containing model and environment settings.
    """
    # Resolve env and model from task config
    task = getattr(cfg, "task", "train")
    task_cfg = getattr(cfg, task, cfg)
    env = getattr(task_cfg, "env", getattr(cfg, "env", None))
    policy = getattr(task_cfg, "policy", task_cfg)
    model = getattr(policy, "model", getattr(cfg, "model", None))

    # The simulation's load_model expects legacy key names
    common_kwargs["problem"] = getattr(env, "name", "ptp") if env else "ptp"
    common_kwargs["model"] = getattr(model, "name", "am") if model else "am"

    if model:
        enc = getattr(model, "encoder", None)
        dec = getattr(model, "decoder", None)

        if enc:
            common_kwargs["encoder"] = getattr(enc, "type", None)
            common_kwargs["embed_dim"] = getattr(enc, "embed_dim", 128)
            common_kwargs["n_encode_layers"] = getattr(enc, "n_layers", 3)
            common_kwargs["n_heads"] = getattr(enc, "n_heads", 8)
            common_kwargs["hidden_dim"] = getattr(enc, "hidden_dim", 512)

            common_kwargs.update(
                {
                    "n_encode_sublayers": getattr(enc, "n_sublayers", None),
                    "learn_affine": getattr(enc.normalization, "learn_affine", True)
                    if hasattr(enc, "normalization")
                    else True,
                    "track_stats": getattr(enc.normalization, "track_stats", True)
                    if hasattr(enc, "normalization")
                    else True,
                    "af_param": getattr(enc.activation, "activation_param", 0.1) if hasattr(enc, "activation") else 0.1,
                    "af_threshold": getattr(enc.activation, "activation_threshold", 0.0)
                    if hasattr(enc, "activation")
                    else 0.0,
                    "af_replacement": getattr(enc.activation, "activation_replacement", 0.0)
                    if hasattr(enc, "activation")
                    else 0.0,
                    "af_nparams": getattr(enc.activation, "activation_num_parameters", 1)
                    if hasattr(enc, "activation")
                    else 1,
                    "af_urange": getattr(enc.activation, "activation_uniform_range", [0, 1])
                    if hasattr(enc, "activation")
                    else [0, 1],
                }
            )

        if dec:
            common_kwargs["n_decode_layers"] = getattr(dec, "n_layers", 1)
            common_kwargs["n_predict_layers"] = getattr(dec, "n_predictor_layers", None)

        common_kwargs.update(
            {
                "epsilon_alpha": getattr(model, "epsilon_alpha", 0.1),
                "momentum_beta": getattr(model, "momentum_beta", 0.1),
                "aggregation": getattr(model, "aggregation_node", "max"),
                "aggregation_graph": getattr(model, "aggregation_graph", "mean"),
            }
        )


def flatten_config_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flatten nested 'graph', 'reward', 'decoding', 'model', and 'policy' configs into the main dictionary.

    Args:
        d: The configuration dictionary to flatten.

    Returns:
        The flattened configuration dictionary.
    """
    new_dict = d.copy()

    # Flatten 'policy' first if it exists
    if "policy" in new_dict and isinstance(new_dict["policy"], ITraversable):
        policy_cfg = new_dict.pop("policy")
        new_dict.update(policy_cfg)

    # Flatten other standard sub-configs
    for key in ["env", "graph", "reward", "decoding", "model"]:
        if key in new_dict and isinstance(new_dict[key], ITraversable):
            nested = new_dict.pop(key)
            new_dict.update(nested)

    return new_dict
