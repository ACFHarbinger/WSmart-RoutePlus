#!/usr/bin/env python3
"""Export a 2D loss surface grid for WSmart-Route Studio §G.5.2.

Produces an NPZ archive with keys:
  - loss_grid: (n, n) loss values on a 2D parameter grid
  - theta1, theta2: 1D axis coordinates (filter-normalized directions when --checkpoint given)

When --checkpoint is omitted, exports a demo Rosenbrock surface for UI development.
With --checkpoint, uses Li et al.-style filter-normalized random directions around the
loaded weights.  When hyperparameters are discoverable beside the checkpoint, a full
training-loss probe runs a greedy forward pass on a synthetic instance per grid point;
otherwise a parameter-distance proxy is used.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np


def _demo_surface(n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    t1 = np.linspace(-1.0, 1.0, n)
    t2 = np.linspace(-1.0, 1.0, n)
    T1, T2 = np.meshgrid(t1, t2)
    loss = (1.0 - T1) ** 2 + 100.0 * (T2 - T1**2) ** 2
    return t1, t2, loss.astype(np.float64)


def _filter_normalized_directions(flat: "np.ndarray") -> Tuple["np.ndarray", "np.ndarray"]:
    import torch

    t = torch.as_tensor(flat, dtype=torch.float32)
    d1 = torch.randn_like(t)
    d2 = torch.randn_like(t)
    d1 = d1 / (d1.norm() + 1e-9) * t.norm()
    d2 = d2 - d2.dot(d1) / (d1.dot(d1) + 1e-9) * d1
    d2 = d2 / (d2.norm() + 1e-9) * t.norm()
    return d1.numpy(), d2.numpy()


def _probe_parameter_proxy_surface(
    checkpoint: Path,
    n: int,
    span: float,
    device: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    import torch

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    state = ckpt.get("state_dict", ckpt.get("model", ckpt))
    flat = torch.cat([p.detach().float().reshape(-1) for p in state.values() if hasattr(p, "numel") and p.numel() > 0])
    if flat.numel() < 2:
        raise ValueError("Checkpoint has too few parameters for a 2D loss grid")

    d1_np, d2_np = _filter_normalized_directions(flat.cpu().numpy())
    d1 = torch.from_numpy(d1_np).to(device)
    d2 = torch.from_numpy(d2_np).to(device)

    alphas = np.linspace(-span, span, n)
    betas = np.linspace(-span, span, n)
    grid = np.zeros((n, n), dtype=np.float64)
    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            theta = flat + float(a) * d1 + float(b) * d2
            grid[i, j] = float((theta - flat).pow(2).mean().sqrt())

    return alphas, betas, grid


def _model_flat_vector(model) -> "np.ndarray":
    import torch

    return (
        torch.cat([p.detach().float().reshape(-1) for p in model.parameters() if p.requires_grad])
        .cpu()
        .numpy()
    )


def _assign_flat_vector(model, flat_vec: "np.ndarray") -> None:
    import torch

    offset = 0
    flat_t = torch.as_tensor(flat_vec, dtype=torch.float32)
    with torch.no_grad():
        for param in model.parameters():
            if not param.requires_grad:
                continue
            n = param.numel()
            param.copy_(flat_t[offset : offset + n].view_as(param))
            offset += n


def _td_to_batch_dict(td) -> dict:
    batch = {}
    for key in td.keys():
        value = td.get(key)
        if hasattr(value, "shape"):
            batch[key] = value
    return batch


def _forward_loss(model, input_dict) -> float:

    out = model(input_dict, strategy="greedy")
    cost = out.get("cost")
    if cost is not None:
        return float(cost.mean().cpu())
    reward = out.get("reward")
    if reward is not None:
        return float(-reward.mean().cpu())
    return float("nan")


def _probe_training_loss_surface(
    checkpoint: Path,
    n: int,
    span: float,
    device: str,
    batch_size: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    import torch

    from logic.src.envs.generators import get_generator
    from logic.src.utils.model.loader import load_model

    load_path = str(checkpoint.resolve())
    model, args = load_model(load_path)
    model = model.to(device)
    model.eval()

    base_flat = _model_flat_vector(model)
    d1_np, d2_np = _filter_normalized_directions(base_flat)
    d1 = torch.from_numpy(d1_np).to(device)
    d2 = torch.from_numpy(d2_np).to(device)
    base_t = torch.as_tensor(base_flat, dtype=torch.float32, device=device)

    problem_name = str(args.get("problem", "vrpp"))
    num_loc = int(args.get("num_loc", args.get("graph_size", 20)) or 20)
    generator = get_generator(problem_name, num_loc=num_loc, device=device)
    batch_size = max(1, int(batch_size))

    alphas = np.linspace(-span, span, n)
    betas = np.linspace(-span, span, n)
    grid = np.zeros((n, n), dtype=np.float64)

    with torch.no_grad():
        for i, a in enumerate(alphas):
            for j, b in enumerate(betas):
                perturbed = base_t + float(a) * d1 + float(b) * d2
                _assign_flat_vector(model, perturbed.cpu().numpy())
                losses: list[float] = []
                for _ in range(batch_size):
                    instance = generator(batch_size=1)
                    input_dict = _td_to_batch_dict(instance)
                    loss_val = _forward_loss(model, input_dict)
                    if np.isfinite(loss_val):
                        losses.append(loss_val)
                if losses:
                    grid[i, j] = float(np.mean(losses))
                else:
                    grid[i, j] = float((perturbed - base_t).pow(2).mean().sqrt().cpu())

    _assign_flat_vector(model, base_flat)
    probe_label = "training_forward" if batch_size == 1 else f"training_forward_b{batch_size}"
    return alphas, betas, grid, probe_label


def _resolve_probe(
    checkpoint: Path,
    n: int,
    span: float,
    device: str,
    probe_mode: str,
    batch_size: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    if probe_mode == "proxy":
        t1, t2, loss = _probe_parameter_proxy_surface(checkpoint, n, span, device)
        return t1, t2, loss, "parameter_proxy"

    if probe_mode == "training":
        return _probe_training_loss_surface(checkpoint, n, span, device, batch_size)

    # auto: prefer training forward when hyperparameters are discoverable
    parent = checkpoint.parent if checkpoint.is_file() else checkpoint
    has_hparams = any(
        (parent / name).exists() for name in ("config.yaml", "hparams.yaml", "args.json")
    )
    if has_hparams:
        try:
            return _probe_training_loss_surface(checkpoint, n, span, device, batch_size)
        except Exception as exc:
            print(f"  [!] Training-loss probe failed ({exc}); falling back to parameter proxy")
    t1, t2, loss = _probe_parameter_proxy_surface(checkpoint, n, span, device)
    return t1, t2, loss, "parameter_proxy"


def _nearest_grid_index(axis: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(axis - value)))


def _interp_loss_at(t1: np.ndarray, t2: np.ndarray, loss: np.ndarray, x: float, y: float) -> float:
    ri = _nearest_grid_index(t1, x)
    rj = _nearest_grid_index(t2, y)
    return float(loss[ri, rj])


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Studio-compatible loss landscape NPZ")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Output .npz path")
    parser.add_argument("--grid-size", type=int, default=32, help="Grid resolution per axis")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional .pt checkpoint")
    parser.add_argument("--span", type=float, default=1.0, help="Direction span for checkpoint probe")
    parser.add_argument("--device", default="cpu", help="Torch device for checkpoint probe")
    parser.add_argument(
        "--probe-mode",
        choices=["auto", "training", "proxy"],
        default="auto",
        help="Loss probe: training forward pass, parameter-distance proxy, or auto-detect",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Synthetic instances averaged per grid point in training-loss probe (§G.5.2)",
    )
    parser.add_argument(
        "--bpc-theta1",
        type=float,
        default=None,
        help="BPC exact-solver θ₁ coordinate for landscape marker (§G.5.2)",
    )
    parser.add_argument(
        "--bpc-theta2",
        type=float,
        default=None,
        help="BPC exact-solver θ₂ coordinate for landscape marker (§G.5.2)",
    )
    parser.add_argument(
        "--distribution",
        choices=["empirical", "gamma3"],
        default="empirical",
        help="Training data distribution label bundled for Studio compare (§G.5.3)",
    )
    args = parser.parse_args()

    probe_label = "rosenbrock_demo"
    if args.checkpoint and args.checkpoint.exists():
        t1, t2, loss, probe_label = _resolve_probe(
            args.checkpoint,
            args.grid_size,
            args.span,
            args.device,
            args.probe_mode,
            args.batch_size,
        )
        source = str(args.checkpoint)
        default_bpc_t1 = 0.0
        default_bpc_t2 = 0.0
    else:
        t1, t2, loss = _demo_surface(args.grid_size)
        source = "rosenbrock_demo"
        default_bpc_t1 = 0.82
        default_bpc_t2 = 0.58

    bpc_t1 = args.bpc_theta1 if args.bpc_theta1 is not None else default_bpc_t1
    bpc_t2 = args.bpc_theta2 if args.bpc_theta2 is not None else default_bpc_t2
    bpc_loss = _interp_loss_at(t1, t2, loss, bpc_t1, bpc_t2)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        loss_grid=loss,
        theta1=t1,
        theta2=t2,
        loss_min=float(loss.min()),
        loss_max=float(loss.max()),
        bpc_theta1=np.array(bpc_t1),
        bpc_theta2=np.array(bpc_t2),
        bpc_loss=np.array(bpc_loss),
        distribution=np.array(args.distribution),
        probe_mode=np.array(probe_label),
        batch_size=np.array(max(1, args.batch_size)),
        source=np.array(source),
    )
    print(
        f"Wrote {args.output} ({args.grid_size}×{args.grid_size}) "
        f"probe={probe_label} loss_min={loss.min():.4f} "
        f"bpc=({bpc_t1:.3f},{bpc_t2:.3f}) loss={bpc_loss:.4f}"
    )


if __name__ == "__main__":
    main()
