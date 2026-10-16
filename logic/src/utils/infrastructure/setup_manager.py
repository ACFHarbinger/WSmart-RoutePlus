"""
HRL Manager setup utilities.

Attributes:
    setup_hrl_manager: Initializes and loads the Manager model for Hierarchical RL.

Example:
    setup_hrl_manager(sim_cfg, device)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from torch import nn


import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from logic.src.models import MandatoryManager

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_cfg_attr(sim_cfg: Any, name: str, default: Any = None) -> Any:
    """
    Retrieve a value from *sim_cfg* (``SimConfig`` or plain ``dict``).

    Args:
        sim_cfg: Simulation config object (``SimConfig`` or dict).
        name: Name of the attribute to retrieve.
        default: Default value to return if the attribute is not found.

    Returns:
        The value of the attribute.
    """
    if isinstance(sim_cfg, dict):
        return sim_cfg.get(name, default)
    return getattr(sim_cfg, name, default)


def _resolve_hrl_path(
    model_paths: Dict[str, str],
    policy: str,
) -> Optional[str]:
    """
    Match *policy* to a checkpoint path in *model_paths*.

    Args:
        model_paths: Dictionary of model paths.
        policy: Policy name.

    Returns:
        The checkpoint path for the given policy.
    """
    if policy in model_paths:
        return model_paths[policy]
    base_policy: str = policy.split("_gamma")[0].split("_emp")[0]
    if base_policy in model_paths:
        return model_paths[base_policy]
    for key in model_paths:
        if policy.startswith(key):
            return model_paths[key]
    return None


def _resolve_checkpoint_file(hrl_path: str, base_path: Optional[str]) -> Optional[str]:
    """
    Return a concrete ``.pt`` file path, or ``None`` if nothing is found.

    Args:
        hrl_path: Path to the HRL model.
        base_path: Base path for models.

    Returns:
        The checkpoint path for the given HRL model.
    """
    if base_path is not None and not os.path.exists(hrl_path):
        hrl_path = os.path.join(base_path, hrl_path)

    if os.path.isfile(hrl_path):
        return hrl_path
    if os.path.isdir(hrl_path):
        pt_files: List[str] = [f for f in os.listdir(hrl_path) if f.endswith(".pt") and "epoch" in f]
        if not pt_files:
            return None
        epoch: int = max(int(os.path.splitext(f)[0].split("-")[1]) for f in pt_files)
        return os.path.join(hrl_path, f"epoch-{epoch}.pt")
    return None


def _resolve_param(
    configs: Dict[str, Any],
    sim_cfg: Any,
    name: str,
    default: Any,
) -> Any:
    """
    Look up *name* from *configs* first, then *sim_cfg*, then *default*.

    Args:
        configs: Configuration dictionary from the loaded neural checkpoint.
        sim_cfg: Simulation config object (``SimConfig`` or dict).
        name: Name of the attribute to retrieve.
        default: Default value to return if the attribute is not found.

    Returns:
        The value of the attribute.
    """
    val = configs.get(name)
    if val is not None:
        return val
    sim_val = _get_cfg_attr(sim_cfg, name)
    return sim_val if sim_val is not None else default


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def setup_hrl_manager(
    sim_cfg: Any,
    device: torch.device,
    configs: Optional[Dict[str, Any]] = None,
    policy: Optional[str] = None,
    base_path: Optional[str] = None,
    worker_model: Optional[nn.Module] = None,
) -> Optional[MandatoryManager]:
    """
    Initializes and loads the Manager model for Hierarchical RL.

    Args:
        sim_cfg: Simulation config object (``SimConfig`` or dict) providing
            ``model_path`` and HRL hyper-parameters as attributes or keys.
        device: Torch device.
        configs: Configuration dictionary from the loaded neural checkpoint.
        policy: Policy name.
        base_path: Base path for models.
        worker_model: Worker model instance for shared encoder.

    Returns:
        The initialized manager model, or None if not applicable.
    """

    if configs is None:
        configs = {}

    if configs.get("mrl_method") != "hrl":
        return None

    # --- Resolve checkpoint path ---
    model_paths: Optional[Dict[str, str]] = _get_cfg_attr(sim_cfg, "model_path")
    hrl_path: Optional[str] = (
        _resolve_hrl_path(model_paths, policy) if model_paths is not None and policy is not None else None
    )
    if hrl_path is None:
        return None

    hrl_path = _resolve_checkpoint_file(hrl_path, base_path)
    if hrl_path is None:
        return None

    # --- Resolve HRL hyper-parameters ---
    mrl_history: int = _resolve_param(configs, sim_cfg, "mrl_history", 10)
    gat_hidden: int = _resolve_param(configs, sim_cfg, "gat_hidden", 128)
    lstm_hidden: int = _resolve_param(configs, sim_cfg, "lstm_hidden", 64)
    global_input_dim: int = _resolve_param(configs, sim_cfg, "global_input_dim", 3)
    shared_encoder_flag: bool = _get_cfg_attr(sim_cfg, "shared_encoder", True)

    # --- Load state dict ---
    from logic.src.utils.model.checkpoint_utils import torch_load_cpu

    load_data: Any = torch_load_cpu(hrl_path)
    state_dict: Dict[str, torch.Tensor] = (
        load_data["manager"] if isinstance(load_data, dict) and "manager" in load_data else load_data
    )

    if "gate_head.0.weight" in state_dict:
        detected_dim: int = state_dict["gate_head.0.weight"].shape[1] - gat_hidden
        if detected_dim > 0 and detected_dim != global_input_dim:
            global_input_dim = detected_dim

    # --- Build manager ---
    from logic.src.models import MandatoryManager

    manager: MandatoryManager = MandatoryManager(
        input_dim_static=2,
        input_dim_dynamic=mrl_history,
        hidden_dim=gat_hidden,
        lstm_hidden=lstm_hidden,
        device=device,
        global_input_dim=global_input_dim,
        shared_encoder=(
            worker_model.embedder
            if (worker_model is not None and shared_encoder_flag and hasattr(worker_model, "embedder"))
            else None  # type: ignore
        ),
    ).to(device)

    manager.load_state_dict(state_dict)
    manager.eval()
    return manager
