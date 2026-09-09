"""Attention heatmap capture and WandB / TensorBoard logging (§A.2 Option C).

Captures runtime multi-head attention matrices during evaluation or validation
and logs them as image summaries to Weights & Biases and TensorBoard.
"""

from __future__ import annotations

import contextlib
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch import nn

from logic.src.tracking.attention_buffer import ensure_attention_buffer
from logic.src.tracking.hooks.attention_hooks import add_attention_hooks
from logic.src.tracking.logging.modules.attention_emit import maybe_emit_attention_viz
from logic.src.tracking.logging.pylogger import get_pylogger
from logic.src.utils.functions import move_to

logger = get_pylogger(__name__)

try:
    import wandb
except ImportError:
    wandb = None  # type: ignore[assignment]


def _resolve_policy_module(model: Any) -> Any:
    """Return the policy or network submodule to instrument."""
    if hasattr(model, "policy"):
        return model.policy
    return model


def _resolve_encoder(policy: Any) -> Optional[nn.Module]:
    """Locate the graph encoder used by attention hooks."""
    for attr in ("embedder", "encoder"):
        encoder = getattr(policy, attr, None)
        if encoder is not None and hasattr(encoder, "layers"):
            return encoder
    return None


def extract_attention_matrix(
    tensor: torch.Tensor,
    head_idx: int = 0,
    batch_idx: int = 0,
) -> Optional[np.ndarray]:
    """Normalize attention tensors to a 2-D ``(nodes, nodes)`` matrix.

    Args:
        tensor: Raw attention weight tensor from a forward hook.
        head_idx: Head index when a head dimension is present.
        batch_idx: Batch index when a batch dimension is present.

    Returns:
        Normalized attention matrix or ``None`` when shape is unsupported.
    """
    if tensor is None or not torch.is_tensor(tensor):
        return None

    arr = tensor.detach().float().cpu().numpy()
    if arr.ndim == 4:
        # (batch, heads, query, key)
        arr = arr[batch_idx, head_idx]
    elif arr.ndim == 3:
        dim0, dim1, dim2 = arr.shape
        # (heads, query, key) vs (batch, query, key)
        arr = arr[head_idx] if dim0 <= 32 and dim1 >= 4 and dim2 >= 4 else arr[batch_idx]
    elif arr.ndim != 2:
        return None

    row_sum = arr.sum(axis=-1, keepdims=True)
    row_sum = np.where(row_sum > 0, row_sum, 1.0)
    return arr / row_sum


def render_attention_heatmap_png(
    matrix: np.ndarray,
    title: str,
    output_path: str,
) -> str:
    """Render a node × node attention matrix to a PNG file.

    Args:
        matrix: Normalized attention weights.
        title: Plot title.
        output_path: Destination PNG path.

    Returns:
        The output path written.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, cmap="viridis", annot=False, square=True)
    plt.title(title)
    plt.xlabel("Key node")
    plt.ylabel("Query node")
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()
    return output_path


def capture_runtime_attention(
    model: Any,
    batch: Any,
    device: Optional[torch.device] = None,
    head_idx: int = 0,
) -> List[Tuple[int, np.ndarray]]:
    """Run one forward pass and capture encoder attention matrices.

    Args:
        model: Policy or Lightning module wrapping a policy.
        batch: Evaluation batch (TensorDict or feature dict).
        device: Optional device override.
        head_idx: Attention head to visualise.

    Returns:
        List of ``(layer_index, matrix)`` pairs.
    """
    policy = _resolve_policy_module(model)
    encoder = _resolve_encoder(policy)
    if encoder is None:
        return []

    if device is not None:
        batch = move_to(batch, device)

    buffer = ensure_attention_buffer(model, head_idx=head_idx)
    if buffer is not None:
        buffer.reset()

    hook_data = add_attention_hooks(encoder)
    matrices: List[Tuple[int, np.ndarray]] = []

    try:
        policy.eval()
        with torch.no_grad():
            if hasattr(policy, "forward"):
                env = getattr(policy, "problem", None) or getattr(model, "env", None)
                if env is not None:
                    policy(batch, env, strategy="greedy")
                else:
                    policy(batch, strategy="greedy")
            elif hasattr(encoder, "forward"):
                if isinstance(batch, dict):
                    h = batch.get("loc") or batch.get("locs")
                    edges = batch.get("edges")
                    if h is not None:
                        encoder(h, mask=edges)
                else:
                    encoder(batch)

        if buffer is not None and len(buffer) > 0:
            matrices = buffer.as_layer_matrices(head_idx=head_idx)
        else:
            for layer_idx, weights in enumerate(hook_data.get("weights", [])):
                mat = extract_attention_matrix(weights, head_idx=head_idx)
                if mat is not None and mat.shape[0] >= 2:
                    matrices.append((layer_idx, mat))
    except Exception:
        logger.debug("Failed to capture runtime attention", exc_info=True)
    finally:
        for handle in hook_data.get("handles", []):
            with contextlib.suppress(Exception):
                handle.remove()

    return matrices


def plot_attention_heatmaps(model: Any, output_dir: str, epoch: int = 0) -> List[str]:
    """Plot static Q/K/V weight heatmaps for all encoder layers.

    Delegates to :mod:`logic.src.utils.plotting.heatmaps` for weight-matrix plots.

    Args:
        model: Policy or module containing attention layers.
        output_dir: Directory for PNG artefacts.
        epoch: Epoch index for filenames.

    Returns:
        List of generated PNG paths (may be empty when unsupported).
    """
    try:
        from logic.src.utils.plotting.heatmaps import plot_attention_heatmaps as _plot_weights

        policy = _resolve_policy_module(model)
        _plot_weights(policy, output_dir, epoch=epoch)
        if not os.path.isdir(output_dir):
            return []
        return [os.path.join(output_dir, name) for name in sorted(os.listdir(output_dir)) if name.endswith(".png")]
    except Exception:
        logger.debug("Failed to plot Q/K/V attention heatmaps", exc_info=True)
        return []


def log_attention_heatmaps_to_backends(
    image_paths: Dict[str, str],
    step: int = 0,
    wandb_mode: str = "disabled",
    tb_writer: Any = None,
) -> None:
    """Log attention PNG artefacts to WandB and/or TensorBoard.

    Args:
        image_paths: Mapping of metric key → PNG path.
        step: Global step / epoch for logging.
        wandb_mode: WandB mode from tracking config.
        tb_writer: Optional TensorBoard ``SummaryWriter`` instance.
    """
    if wandb_mode != "disabled" and wandb is not None:
        with contextlib.suppress(Exception):
            if getattr(wandb, "run", None) is not None:
                payload = {
                    key: wandb.Image(path, caption=key)
                    for key, path in image_paths.items()
                    if path and os.path.isfile(path)
                }
                if payload:
                    wandb.log(payload, step=step)

    if tb_writer is not None:
        with contextlib.suppress(Exception):
            import torchvision

            for key, path in image_paths.items():
                if not path or not os.path.isfile(path):
                    continue
                img = torchvision.io.read_image(path)
                tb_writer.add_image(key.replace("/", "_"), img, global_step=step)


def maybe_log_eval_attention_heatmaps(
    model: Any,
    batch: Any,
    cfg: Any,
    output_subdir: str = "eval_attention",
    step: int = 0,
    tb_writer: Any = None,
    phase: str = "eval",
    epoch: int = 0,
) -> List[str]:
    """Capture and log attention heatmaps when tracking flags are enabled.

    Args:
        model: Model or policy under evaluation.
        batch: First evaluation batch for runtime capture.
        cfg: Root config with a ``tracking`` section.
        output_subdir: Subdirectory under ``tracking.log_dir``.
        step: Logging step index.
        tb_writer: Optional TensorBoard writer.

    Returns:
        List of PNG paths written to disk.
    """
    tracking = getattr(cfg, "tracking", None)
    if tracking is None:
        return []

    log_runtime = bool(getattr(tracking, "log_attention", False))
    log_weights = bool(getattr(tracking, "log_attention_heatmaps", False))
    if not log_runtime and not log_weights:
        return []

    log_dir = getattr(tracking, "log_dir", "logs")
    output_dir = os.path.join(log_dir, output_subdir)
    os.makedirs(output_dir, exist_ok=True)

    wandb_mode = getattr(tracking, "wandb_mode", "disabled")
    image_paths: Dict[str, str] = {}
    written: List[str] = []

    device = next(model.parameters()).device if hasattr(model, "parameters") else None

    if log_runtime:
        for layer_idx, matrix in capture_runtime_attention(model, batch, device=device):
            rel_key = f"attention/runtime/layer_{layer_idx}"
            path = os.path.join(output_dir, f"runtime_layer{layer_idx}.png")
            render_attention_heatmap_png(matrix, f"Runtime Attention L{layer_idx}", path)
            image_paths[rel_key] = path
            written.append(path)

    if log_weights:
        for path in plot_attention_heatmaps(model, output_dir, epoch=step):
            name = os.path.splitext(os.path.basename(path))[0]
            image_paths[f"attention/weights/{name}"] = path
            if path not in written:
                written.append(path)

    if image_paths:
        log_attention_heatmaps_to_backends(
            image_paths,
            step=step,
            wandb_mode=wandb_mode,
            tb_writer=tb_writer,
        )

        run = None
        with contextlib.suppress(Exception):
            from logic.src.tracking.core.run import get_active_run

            run = get_active_run()
        if run is not None:
            with contextlib.suppress(Exception):
                run.log_artifact(output_dir, artifact_type="attention_heatmap")

    with contextlib.suppress(Exception):
        maybe_emit_attention_viz(
            model,
            cfg,
            phase=phase,
            epoch=epoch,
            step=step,
        )

    return written
