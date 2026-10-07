"""Visualization, plotting, and interpretability suite for WSmart-Route.

This package provides a comprehensive set of tools for inspecting model
internals, projecting embeddings, generating loss landscapes, monitoring training
progress, and visualizing routing solutions. It includes tools for 2D line
charts, interactive heatmaps, neural attention maps, and geographic/cartesian
route rendering.

Attributes:
    visualize_epoch: High-level entry point for visualization during training.
    log_weight_distributions: Exports weight histograms to TensorBoard.
    plot_weight_trajectories: PCA-based visualization of weight evolution.
    project_node_embeddings: 3D projection of latent node features.
    plot_attention_heatmaps: Transformer attention weight visualization.
    plot_logit_lens: Intermediate layer logit prediction visualization.
    plot_loss_landscape: 2D/3D loss surface generation tool.
    plot_linechart: Main entry point for static 2D metric charts.
    plot_3dchart: Main entry point for static 3D metric charts.
    draw_graph: Coordinate-free graph visualization.
    plot_tsp: Renders single-vehicle tours.
    plot_vehicle_routes: Renders multi-vehicle VRP solutions.
    discrete_cmap: Utility for vehicle-indexed coloring.
    plot_attention_maps_wrapper: Heatmap generator for transformer attention.
    visualize_interactive_plot: Launcher for Plotly-based visualizations.
    plot_training_logs: Plots logged training history (mean and std dev per metric type).
    log_plot: Execution function for saving static plots.

Example:
    >>> from logic.src.utils.plotting import plot_tsp
    >>> plot_tsp(xy, tour, ax)
"""

import argparse
import os
from typing import Any, List, Union

import torch
from omegaconf import DictConfig

from logic.src.configs import Config

from . import embeddings as embeddings
from . import heatmaps as heatmaps
from . import helpers as helpers
from . import landscape as landscape

# Imports from plotting files
from .attention import plot_attention_maps_wrapper
from .charts import plot_linechart
from .charts3d import plot_3dchart

# Imports from visualization files
from .embeddings import log_weight_distributions, plot_weight_trajectories, project_node_embeddings
from .heatmaps import (
    plot_attention_heatmaps,
    plot_logit_lens,
)
from .helpers import (
    MyModelWrapper,
    get_batch,
    load_model_instance,
)
from .interactive import visualize_interactive_plot
from .landscape import (
    ImitationMetric,
    RLMetric,
    imitation_loss_fn,
    plot_loss_landscape,
    rl_loss_fn,
)

# Imports from log_visualization and routes
from .log_visualization import log_plot, plot_training_logs
from .routes import (
    discrete_cmap,
    draw_graph,
    plot_tsp,
    plot_vehicle_routes,
)

__all__ = [
    "plot_linechart",
    "plot_3dchart",
    "draw_graph",
    "plot_tsp",
    "plot_vehicle_routes",
    "discrete_cmap",
    "plot_attention_maps_wrapper",
    "visualize_interactive_plot",
    "visualize_epoch",
    "log_weight_distributions",
    "plot_weight_trajectories",
    "project_node_embeddings",
    "plot_attention_heatmaps",
    "plot_logit_lens",
    "plot_loss_landscape",
    "imitation_loss_fn",
    "rl_loss_fn",
    "ImitationMetric",
    "RLMetric",
    "MyModelWrapper",
    "get_batch",
    "load_model_instance",
    "embeddings",
    "heatmaps",
    "helpers",
    "landscape",
    "log_plot",
    "plot_training_logs",
]


def visualize_epoch(  # noqa: C901
    model: Any, problem: Any, cfg: Union[Config, DictConfig], epoch: int, tb_logger: Any = None
) -> None:
    """Main entry point for visualization during training.

    This function selectively runs different visualization routines based on
    the configuration settings in `cfg.rl.viz_modes`.

    Args:
        model: The neural model to visualize.
        problem: The problem instance for environment context.
        cfg: Root Hydra configuration.
        epoch: Current training epoch number.
        tb_logger: Optional TensorBoard SummaryWriter. Defaults to None.
    """
    rl = getattr(cfg, "rl", None) if not isinstance(cfg, dict) else cfg.get("rl")
    viz_modes: List[str] = getattr(rl, "viz_modes", []) if rl is not None else []
    if not viz_modes:
        return

    # Extract temporal horizon and other configs robustly
    model_cfg = None
    if not isinstance(cfg, dict):
        train = getattr(cfg, "train", None)
        if train is not None:
            policy = getattr(train, "policy", None)
            if policy is not None:
                model_cfg = getattr(policy, "model", None)
        if model_cfg is None:
            model_cfg = getattr(cfg, "model", None)
    else:
        train = cfg.get("train", {})
        if isinstance(train, dict):
            model_cfg = train.get("policy", {}).get("model", cfg.get("model"))
        else:
            model_cfg = getattr(getattr(train, "policy", None), "model", cfg.get("model"))

    temporal_horizon = 0
    if model_cfg is not None:
        temporal_horizon = (
            getattr(model_cfg, "temporal_horizon", 0)
            if not isinstance(model_cfg, dict)
            else model_cfg.get("temporal_horizon", 0)
        )

    log_dir: str = getattr(rl, "log_dir", "logs") if not isinstance(rl, dict) else rl.get("log_dir", "logs")
    run_name: str = getattr(rl, "run_name", "run") if not isinstance(rl, dict) else rl.get("run_name", "run")
    save_dir: str = getattr(rl, "save_dir", "outputs") if not isinstance(rl, dict) else rl.get("save_dir", "outputs")

    # Resolve graph size
    graph_size = 50
    if not isinstance(cfg, dict):
        _train = getattr(cfg, "train", None)
        if _train is not None:
            _env = getattr(_train, "env", None)
            if _env is not None:
                _graph = getattr(_env, "graph", None)
                if _graph is not None:
                    graph_size = int(getattr(_graph, "num_loc", 50) or 50)
            # fallback to train.graph
            _graph2 = getattr(_train, "graph", None)
            if _graph2 is not None:
                graph_size = int(getattr(_graph2, "num_loc", 50) or 50)
    else:
        train_dict = cfg.get("train", {})
        if isinstance(train_dict, dict):
            env_dict = train_dict.get("env", {})
            if isinstance(env_dict, dict):
                graph_dict = env_dict.get("graph", train_dict.get("graph", {}))
                if isinstance(graph_dict, dict):
                    graph_size = int(graph_dict.get("num_loc", 50) or 50)

    viz_output_dir = os.path.join(log_dir, "visualizations")
    os.makedirs(viz_output_dir, exist_ok=True)

    print(f"\n--- Visualizing Epoch {epoch} ---")

    # Move model to CPU for visualization to avoid device mismatch issues
    orig_device = next(model.parameters()).device
    model.cpu()

    # Move cost weights to CPU if they exist
    if hasattr(model, "cost_weights"):
        model.cost_weights = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in model.cost_weights.items()}

    try:
        if "distributions" in viz_modes or "both" in viz_modes:
            writer = tb_logger
            log_weight_distributions(
                model,
                epoch,
                log_dir=os.path.join(log_dir, run_name),
                writer=writer,
            )

        if "embeddings" in viz_modes:
            x_batch = get_batch(
                torch.device("cpu"),
                size=graph_size,
                batch_size=1,
                temporal_horizon=temporal_horizon,
            )
            writer = tb_logger
            project_node_embeddings(
                model,
                x_batch,
                log_dir=os.path.join(log_dir, run_name),
                writer=writer,
                epoch=epoch,
            )

        if "heatmaps" in viz_modes:
            plot_attention_heatmaps(model, viz_output_dir, epoch=epoch)

        if "logit_lens" in viz_modes:
            x_batch = get_batch(
                torch.device("cpu"),
                size=graph_size,
                batch_size=1,
                temporal_horizon=temporal_horizon,
            )
            plot_logit_lens(
                model,
                x_batch,
                os.path.join(viz_output_dir, f"logit_lens_ep{epoch}.png"),
                epoch=epoch,
            )

        if "loss" in viz_modes or "both" in viz_modes:
            plot_loss_landscape(
                model,
                cfg,
                viz_output_dir,
                epoch=epoch,
                size=graph_size,
                batch_size=4,
                resolution=10,
            )

        if "trajectory" in viz_modes:
            checkpoint_dir = save_dir
            plot_weight_trajectories(checkpoint_dir, os.path.join(viz_output_dir, "trajectory.png"))

    finally:
        # Restore model to original device
        model.to(orig_device)
        if hasattr(model, "cost_weights"):
            model.cost_weights = {
                k: v.to(orig_device) if isinstance(v, torch.Tensor) else v for k, v in model.cost_weights.items()
            }

    print("Visualization complete.\n")


def main():
    """Main execution entry point for visualization debugging."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="Path to model checkpoint")
    parser.add_argument("--checkpoint_dir", type=str, help="Directory with multiple checkpoints")
    parser.add_argument("--output_dir", type=str, default="visualizations", help="Output directory")
    parser.add_argument("--log_dir", type=str, default="logs", help="TensorBoard log directory")

    parser.add_argument("--size", type=int, default=100, help="Problem size")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument("--resolution", type=int, default=10, help="Resolution for landscapes")
    parser.add_argument("--span", type=float, default=1.0, help="Span for landscapes")
    parser.add_argument("--problem", type=str, default="ptp", help="Problem type")

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=[
            "trajectory",
            "distributions",
            "embeddings",
            "heatmaps",
            "logit_lens",
            "loss",
            "both",
        ],
        help="Visualization mode",
    )

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode == "trajectory":
        if not args.checkpoint_dir:
            raise ValueError("--checkpoint_dir required for trajectory")
        plot_weight_trajectories(args.checkpoint_dir, os.path.join(args.output_dir, "trajectory.png"))
        return  # Trajectory doesn't need model loading

    # Load model
    if not args.model_path:
        raise ValueError("--model_path required for this mode")
    model = load_model_instance(args.model_path, device, size=args.size, problem_name=args.problem)

    if args.mode == "distributions":
        log_weight_distributions(model, 0, args.log_dir)

    elif args.mode == "embeddings":
        x_batch = get_batch(device, size=args.size, batch_size=1)
        project_node_embeddings(model, x_batch, args.log_dir)

    elif args.mode == "heatmaps":
        plot_attention_heatmaps(model, args.output_dir)

    elif args.mode == "logit_lens":
        x_batch = get_batch(device, size=args.size, batch_size=1)
        plot_logit_lens(model, x_batch, os.path.join(args.output_dir, "logit_lens.png"))

    elif args.mode in {"loss", "both"}:
        # Standalone mode creates a minimal Config
        standalone_cfg = Config()
        standalone_cfg.train.policy.model.temporal_horizon = 0
        plot_loss_landscape(
            model,
            standalone_cfg,
            args.output_dir,
            size=args.size,
            batch_size=args.batch_size,
            resolution=args.resolution,
            span=args.span,
        )


if __name__ == "__main__":
    main()
