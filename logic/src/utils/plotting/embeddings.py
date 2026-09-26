"""Embedding visualization and weight trajectory analysis tools.

This module provides utilities for analyzing high-dimensional model state
and node feature representations. It supports projecting weight vectors
across training epochs via PCA to visualize learning trajectories and
interfacing with TensorBoard Projector for node embedding visualization.

Attributes:
    plot_weight_trajectories: Visualizes weight evolution across checkpoints.
    log_weight_distributions: Exports weight histograms to TensorBoard.
    project_node_embeddings: Projects 3D node features for TensorBoard analysis.

Example:
    >>> from logic.src.utils.plotting import log_weight_distributions
    >>> log_weight_distributions(model, epoch, "logs/tb/")
"""

import os

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torch.utils.tensorboard.writer import SummaryWriter


def plot_weight_trajectories(checkpoint_dir, output_file):
    """Plots the trajectory of model weights across epochs using PCA projection.

    Args:
        checkpoint_dir: Directory containing model checkpoint files (.pt).
        output_file: File path where the trajectory plot will be saved.
    """
    print("Computing Weight Trajectories...")
    # Create output dir if needed
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")]
    # Sort files by epoch number
    files.sort(key=lambda x: int(x.split("-")[1].split(".")[0]) if "-" in x and "epoch" in x else 0)

    weights = []
    epochs = []

    for f in files:
        path = os.path.join(checkpoint_dir, f)
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            state_dict = checkpoint.get("model", checkpoint)

            # More robust weight extraction: try encoder, then embed, then any model param
            weights_to_flat = []
            for k, p in state_dict.items():
                if "encoder" in k.lower() or "embed" in k.lower() or "model" in k.lower():
                    weights_to_flat.append(p.flatten())

            if not weights_to_flat:
                weights_to_flat = [p.flatten() for p in state_dict.values() if isinstance(p, torch.Tensor)]

            if not weights_to_flat:
                print(f"No valid tensors found in {f}")
                continue

            flat_weight = torch.cat(weights_to_flat).numpy()
            weights.append(flat_weight)
            epochs.append(f.replace(".pt", ""))
        except Exception as e:
            print(f"Skipping {f} due to error: {e}")

    if not weights:
        print("No weights found.")
        return

    weights = np.array(weights)  # type: ignore[assignment]
    if weights.shape[0] < 2:  # type: ignore[attr-defined]
        print("Not enough checkpoints for trajectory.")
        return

    pca = PCA(n_components=2)
    projected = pca.fit_transform(weights)

    plt.figure(figsize=(10, 8))
    plt.plot(projected[:, 0], projected[:, 1], "-o", alpha=0.6)  # pyrefly: ignore [bad-argument-type]
    for i, txt in enumerate(epochs):
        plt.annotate(txt, (projected[i, 0], projected[i, 1]), size=8)

    plt.title("Training Trajectory (PCA Projection of Model Weights)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True)
    plt.savefig(output_file)
    plt.close()
    print(f"Trajectory saved to {output_file}")


def log_weight_distributions(model, epoch, log_dir, writer=None):
    """Logs histograms of model weight distributions to TensorBoard.

    Args:
        model: The PyTorch neural network to inspect.
        epoch: Current training epoch index.
        log_dir: TensorBoard log directory (used if writer is None).
        writer: Optional existing SummaryWriter. Defaults to None.
    """
    close_writer = False
    if writer is None:
        print(f"Logging Weight Distributions to TensorBoard at {log_dir}...")
        writer = SummaryWriter(log_dir)
        close_writer = True

    for name, param in model.named_parameters():
        writer.add_histogram(name, param, epoch)

    if close_writer:
        writer.close()
        print("Distributions logged.")


def project_node_embeddings(model, x_batch, log_dir, writer=None, epoch=0):
    """Projects node embeddings to 3D space in TensorBoard Projector.

    Args:
        model: The model whose embedder will be used.
        x_batch: Input batch containing node and edge features.
        log_dir: TensorBoard log directory (used if writer is None).
        writer: Optional existing SummaryWriter. Defaults to None.
        epoch: Current training epoch index. Defaults to 0.
    """
    close_writer = False
    if writer is None:
        print(f"Projecting Node Embeddings to TensorBoard at {log_dir}...")
        writer = SummaryWriter(log_dir)
        close_writer = True

    model.eval()
    with torch.no_grad():
        # Get initial node embeddings
        nodes = model._get_initial_embeddings(x_batch)
        edges = x_batch.get("edges", None)
        dist_matrix = x_batch.get("dist", None)

        # Check if embedder accepts dist (e.g. GAT with edge embeddings)
        if getattr(model.embedder, "init_edge_embed", None) is not None:
            embeddings = model.embedder(nodes, edges, dist=dist_matrix)
        else:
            embeddings = model.embedder(nodes, edges)

        # Take the first sample in batch
        sample_embeddings = embeddings[0]  # (G, D)
        labels = [f"Node_{i}" for i in range(sample_embeddings.size(0))]
        labels[0] = "Depot"

        writer.add_embedding(sample_embeddings, metadata=labels, tag=f"Node_Embeddings_Ep{epoch}")

    if close_writer:
        writer.close()
        print("Embeddings projected.")
