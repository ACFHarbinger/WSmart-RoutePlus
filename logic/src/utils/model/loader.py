"""
Central model loading utility.

Attributes:
    load_model: Loads the entire model from a checkpoint or directory.

Example:
    >>> from logic.src.utils.model.loader import load_model
    >>> model, args = load_model("path/to/checkpoint.pt")
    >>> isinstance(model, torch.nn.Module)
    True
    >>> isinstance(args, dict)
    True
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, Optional, Tuple, Type, cast

from omegaconf import OmegaConf
from torch import nn

from logic.src.configs.models.activation_function import ActivationConfig
from logic.src.configs.models.normalization import NormalizationConfig
from logic.src.models.subnets.factories.attention import AttentionComponentFactory
from logic.src.models.subnets.factories.base import NeuralComponentFactory
from logic.src.models.subnets.factories.gac import GACComponentFactory
from logic.src.models.subnets.factories.gcn import GCNComponentFactory
from logic.src.models.subnets.factories.ggac import GGACComponentFactory
from logic.src.models.subnets.factories.mlp import MLPComponentFactory
from logic.src.models.subnets.factories.tgc import TGCComponentFactory

from .checkpoint_utils import torch_load_cpu
from .config_utils import load_args
from .problem_factory import load_problem

_UNUSED_LEGACY_KEYS = (
    "context_embedder.project_step_context.",
    "decoder.project_fixed_context.",
    "project_fixed_context.",
)


def load_model(path: str, epoch: Optional[int] = None) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Loads the entire model from a checkpoint or directory.

    Args:
        path: Path to checkpoint file or directory containing checkpoints.
        epoch: Specific epoch to load if path is a directory. If None, loads latest.

    Returns:
        tuple: (model, args)

    Raises:
        ValueError: If no valid epoch files found in directory.
    """

    if os.path.isfile(path):
        model_filename = path
        path = os.path.dirname(model_filename)
    elif os.path.isdir(path):
        model_filename = _find_latest_checkpoint(path, epoch)
    else:
        raise FileNotFoundError(f"{path} is not a valid directory or file.")

    # Load hyperparameters
    args = _load_hyperparameters(path)

    if not args:
        raise ValueError("Could not find hyperparameters in directory: {}".format(path))

    from logic.src.models import AttentionModel, TemporalAttentionModel

    problem = load_problem(args["problem"])

    # Map encoder name to Factory Class
    factory_class = {
        "gat": AttentionComponentFactory,
        "gac": GACComponentFactory,
        "tgc": TGCComponentFactory,
        "ggac": GGACComponentFactory,
        "mlp": MLPComponentFactory,
        "gcn": GCNComponentFactory,
    }.get(args.get("encoder", "gat"))

    # Fallback/Check
    if factory_class is None:
        raise ValueError(f"Unknown encoder type: {args.get('encoder', 'gat')}")

    factory_type = cast(Type[NeuralComponentFactory], factory_class)
    component_factory = factory_type()

    decoder_type = args.get("decoder_type", "glimpse")

    # Use TemporalAttentionModel only for 'tam', otherwise AttentionModel
    model_class = TemporalAttentionModel if args.get("model") == "tam" else AttentionModel
    # AttentionModel reads normalization/activation only from the structured configs;
    # the flat kwargs below are kept for older call sites but are not consumed.
    norm_config = NormalizationConfig(
        norm_type=args["normalization"],
        epsilon=args["epsilon_alpha"],
        learn_affine=args["learn_affine"],
        track_stats=args["track_stats"],
        momentum=args["momentum_beta"],
        n_groups=args["gnorm_groups"],
        k_lrnorm=args["lrnorm_k"],
    )
    activation_config = ActivationConfig(
        name=args["activation"],
        param=args["af_param"],
        threshold=args["af_threshold"],
        replacement_value=args["af_replacement"],
        n_params=args["af_nparams"],
        # Older configs store a single number here; ActivationConfig expects [low, high].
        range=list(args["af_urange"]) if isinstance(args["af_urange"], (list, tuple)) else ActivationConfig().range,
    )

    model = model_class(
        args["embed_dim"],
        args["hidden_dim"],
        problem,
        component_factory,
        args["n_encode_layers"],
        args["n_encode_sublayers"],
        args["n_decode_layers"],
        n_heads=args["n_heads"],
        norm_config=norm_config,
        activation_config=activation_config,
        normalization=args["normalization"],
        norm_learn_affine=args["learn_affine"],
        norm_track_stats=args["track_stats"],
        norm_eps_alpha=args["epsilon_alpha"],
        norm_momentum_beta=args["momentum_beta"],
        lrnorm_k=args["lrnorm_k"],
        gnorm_groups=args["gnorm_groups"],
        activation_function=args["activation"],
        af_param=args["af_param"],
        af_threshold=args["af_threshold"],
        af_replacement_value=args["af_replacement"],
        af_num_params=args["af_nparams"],
        af_uniform_range=args["af_urange"],
        dropout_rate=args["dropout"],
        aggregation=args["aggregation"],
        aggregation_graph=args["aggregation_graph"],
        tanh_clipping=args["tanh_clipping"],
        mask_inner=args.get("mask_inner", True),
        mask_logits=args.get("mask_logits", True),
        mask_graph=args.get("mask_graph", False),
        checkpoint_encoder=args.get("checkpoint_encoder", False),
        shrink_size=args.get("shrink_size", None),
        temporal_horizon=args.get("temporal_horizon", 0),
        spatial_bias=args.get("spatial_bias", False),
        spatial_bias_scale=args.get("spatial_bias_scale", 1.0),
        entropy_weight=args.get("entropy_weight", 0.0),
        predictor_layers=args.get("n_predict_layers", 0),
        decoder_type=decoder_type,
    )

    # Overwrite model parameters by parameters to load
    data = torch_load_cpu(model_filename)
    if "model" in data:
        loaded_state_dict = data["model"]
    elif "state_dict" in data:
        # PyTorch Lightning checkpoint: keys are prefixed "policy.*" inside the REINFORCE module.
        # Remap to the old AttentionModel key naming.
        _KEY_MAP = {
            "init_embedding.node_embed.": "context_embedder.init_embed.",
            "init_embedding.depot_embed.": "context_embedder.init_embed_depot.",
        }
        loaded_state_dict = {}
        for k, v in data["state_dict"].items():
            if not k.startswith("policy."):
                continue
            k = k[len("policy.") :]
            for src, dst in _KEY_MAP.items():
                if k.startswith(src):
                    k = dst + k[len(src) :]
                    break
            loaded_state_dict[k] = v
    else:
        raise ValueError(
            f"Unsupported checkpoint layout in {model_filename}: expected a 'model' or 'state_dict' entry."
        )
    if not loaded_state_dict:
        raise ValueError(f"Checkpoint {model_filename} contains no model parameters.")

    result = model.load_state_dict(loaded_state_dict, strict=False)
    # PL checkpoints of AttentionModelPolicy have no weights for the legacy model's
    # context_embedder.project_step_context, which the decode path never uses (the
    # glimpse decoder owns its own context embedding). Anything else must match.
    missing = [k for k in result.missing_keys if not k.startswith(_UNUSED_LEGACY_KEYS)]
    unexpected = [k for k in result.unexpected_keys if not k.startswith(_UNUSED_LEGACY_KEYS)]
    if missing or unexpected:
        msg = (
            f"Checkpoint {model_filename} does not match the model: missing={missing[:10]} unexpected={unexpected[:10]}"
        )
        # Strictness is verified for AttentionModel checkpoints; other model classes
        # (e.g. TAM) may carry layout differences, so only warn for them.
        if model_class is AttentionModel:
            raise ValueError(msg)
        print(f"  [!] {msg}")

    rel_model_filename = model_filename.replace(os.path.expanduser("~"), "~")
    print("  [*] Loaded model from {}".format(rel_model_filename))
    model.eval()
    return model, args


def _find_latest_checkpoint(path: str, epoch: Optional[int]) -> str:
    """
    Helper to find the correct .pt file in a directory.

    Args:
        path: Path to directory containing checkpoints.
        epoch: Specific epoch to load. If None, loads latest.

    Returns:
        str: Path to the checkpoint file.

    Raises:
        ValueError: If no valid epoch files are found.
    """
    if epoch is None:
        pt_files = [f for f in os.listdir(path) if f.endswith(".pt")]
        epochs = []
        for f in pt_files:
            name = os.path.splitext(f)[0]
            m = re.match(r"epoch-?(\d+)", name)
            if m:
                epochs.append(int(m.group(1)))

        if not epochs:
            raise ValueError("No valid epoch files (epoch-N.pt or epochN.pt) found in directory: {}".format(path))
        epoch = max(epochs)

    # Check if version with hyphen exists first
    hyphen_path = os.path.join(path, f"epoch-{epoch}.pt")
    if os.path.exists(hyphen_path):
        return hyphen_path
    return os.path.join(path, f"epoch{epoch}.pt")


def _parse_hydra_config(cfg: Any) -> Dict[str, Any]:
    """
    Helper to parse raw Hydra configuration into flat arguments.
    Supports both legacy flat models and newer nested architectures.

    Args:
        cfg: Hydra configuration object.

    Returns:
        dict: Flat dictionary of arguments.
    """
    # Convert to container for easier .get() access if it's a DictConfig
    container = OmegaConf.to_container(cfg, resolve=True) if OmegaConf.is_config(cfg) else cfg
    assert container is not None and isinstance(container, dict), f"Container is not a dict: {container}"

    model = container.get("model", {})
    env = container.get("env", {})

    # Detect nested structure (encoder is a dict) vs legacy (encoder_type string)
    is_nested = isinstance(model.get("encoder"), dict)
    if is_nested:
        enc = model.get("encoder", {})
        dec = model.get("decoder", {})
        norm = enc.get("normalization", {})
        act = enc.get("activation", {})

        args = {
            "problem": env.get("name", "vrpp"),
            "encoder": enc.get("type", "gat"),
            "model": model.get("name", "am"),
            "embed_dim": enc.get("embed_dim", 128),
            "hidden_dim": enc.get("hidden_dim", 512),
            "n_heads": enc.get("n_heads", 8),
            "n_encode_layers": enc.get("n_layers", 3),
            "n_encode_sublayers": enc.get("n_sublayers", 1),
            "n_decode_layers": dec.get("n_layers", 2),
            "n_predict_layers": dec.get("n_predictor_layers", 0),
            "normalization": norm.get("norm_type", "instance"),
            "activation": act.get("name", "gelu"),
            "dropout": enc.get("dropout", 0.1),
            "tanh_clipping": dec.get("tanh_clipping", 10.0),
            "learn_affine": norm.get("learn_affine", True),
            "track_stats": norm.get("track_stats", False),
            "epsilon_alpha": norm.get("epsilon", 1e-5),
            "momentum_beta": norm.get("momentum", 0.1),
            "lrnorm_k": norm.get("k_lrnorm", 1.0),
            "gnorm_groups": norm.get("n_groups", 1),
            "af_param": act.get("param", 1.0),
            "af_threshold": act.get("threshold", 6.0),
            "af_replacement": act.get("replacement_value", 6.0),
            "af_nparams": act.get("n_params", 1),
            "af_urange": act.get("range", [0.1, 0.4]),
            "aggregation": enc.get("aggregation_node", "sum"),
            "aggregation_graph": enc.get("aggregation_graph", "avg"),
            "spatial_bias": enc.get("spatial_bias", False),
            "spatial_bias_scale": enc.get("spatial_bias_scale", 1.0),
            "shrink_size": model.get("shrink_size"),
            "temporal_horizon": model.get("temporal_horizon", 0),
            "mask_inner": enc.get("mask_inner", True),
            "mask_logits": dec.get("mask_logits", True),
            "mask_graph": enc.get("mask_graph", False),
            "checkpoint_encoder": enc.get("checkpoint_encoder", False),
        }
    else:
        # Fallback to legacy flat structure
        args = {
            "problem": env.get("name", "vrpp"),
            "encoder": model.get("encoder_type", "gat"),
            "model": model.get("name", "am"),
            "embed_dim": model.get("embed_dim", 128),
            "hidden_dim": model.get("hidden_dim", 512),
            "n_heads": model.get("n_heads", 8),
            "n_encode_layers": model.get("n_encoder_layers", 3),
            "n_encode_sublayers": model.get("n_encoder_sublayers", 1),
            "n_decode_layers": model.get("n_decoder_layers", 2),
            "n_predict_layers": model.get("n_predictor_layers", 0),
            "normalization": model.get("normalization", "instance"),
            "activation": model.get("activation", "gelu"),
            "dropout": model.get("dropout", 0.1),
            "tanh_clipping": model.get("tanh_clipping", 10.0),
            "learn_affine": model.get("learn_affine", True),
            "track_stats": model.get("track_stats", False),
            "epsilon_alpha": model.get("epsilon_alpha", 1e-5),
            "momentum_beta": model.get("momentum_beta", 0.1),
            "lrnorm_k": model.get("lrnorm_k", 1.0),
            "gnorm_groups": model.get("gnorm_groups", 1),
            "af_param": model.get("activation_param", 1.0),
            "af_threshold": model.get("activation_threshold", 6.0),
            "af_replacement": model.get("activation_replacement", 6.0),
            "af_nparams": model.get("activation_num_parameters", 1),
            "af_urange": model.get("activation_uniform_range", [0.1, 0.4]),
            "aggregation": model.get("aggregation_node", "sum"),
            "aggregation_graph": model.get("aggregation_graph", "avg"),
            "spatial_bias": model.get("spatial_bias", False),
            "spatial_bias_scale": model.get("spatial_bias_scale", 1.0),
            "shrink_size": model.get("shrink_size"),
            "temporal_horizon": model.get("temporal_horizon", 0),
            "mask_inner": model.get("mask_inner", True),
            "mask_logits": model.get("mask_logits", True),
            "mask_graph": model.get("mask_graph", False),
            "checkpoint_encoder": model.get("checkpoint_encoder", False),
        }

    if "rl" in container:
        args["entropy_weight"] = container["rl"].get("entropy_weight", 0.0)
    return args


def _load_hyperparameters(path: str) -> Dict[str, Any]:
    """
    Helper to detect and load hyperparameters from various formats.

    Args:
        path: Path to directory containing hyperparameters.

    Returns:
        dict: Dictionary of hyperparameters.
    """
    config_yaml_path = os.path.join(path, "config.yaml")
    hparams_yaml_path = os.path.join(path, "hparams.yaml")
    args_json_path = os.path.join(path, "args.json")

    if os.path.exists(config_yaml_path) or os.path.exists(hparams_yaml_path):
        yaml_path = config_yaml_path if os.path.exists(config_yaml_path) else hparams_yaml_path
        cfg = OmegaConf.load(yaml_path)

        if "model" in cfg and "env" in cfg:
            # Full Hydra Config
            args = _parse_hydra_config(cfg)
            if "rl" in cfg:
                args["entropy_weight"] = cfg.rl.entropy_weight
            return args

        # Maybe it's a flat DictConfig or Lightning hparams
        args = cast(Dict[str, Any], OmegaConf.to_container(cfg, resolve=True))
        # Handle renames
        if "num_encoder_layers" in args and "n_encode_layers" not in args:
            args["n_encode_layers"] = args["num_encoder_layers"]
        return args

    if os.path.exists(args_json_path):
        return load_args(args_json_path)

    return {}
