"""
Graph and adjacency matrix generation utilities.

Attributes:
    OSMNX_AVAILABLE: Whether osmnx is available.
    generate_adj_matrix: Generates a random adjacency matrix.
    get_edge_idx_dist: Generates edge indices based on shortest distances in the distance matrix.
    get_adj_knn: Generates an adjacency matrix based on K-Nearest Neighbors.
    get_adj_osm: Computes an adjacency matrix via OpenStreetMap for given coordinates.

Example:
    generate_adj_matrix(size, num_edges)
    get_edge_idx_dist(dist_matrix, num_edges)
    get_adj_knn(dist_matrix, k_neighbors)
    get_adj_osm(coords, size, args)
"""

from typing import Any, Optional, Union, cast

import networkx as nx
import numpy as np

from .conversion import adj_to_idx

try:
    import osmnx as ox

    OSMNX_AVAILABLE = True
except ImportError:
    ox = None  # type: ignore[assignment]
    OSMNX_AVAILABLE = False


def generate_adj_matrix(
    size: int,
    num_edges: Union[int, float],
    undirected: bool = False,
    add_depot: bool = True,
    negative: bool = False,
    np_rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Generates a random adjacency matrix.

    Args:
        size: Number of nodes in the graph.
        num_edges: Number of edges to add to the graph.
        undirected: Whether the graph should be undirected.
        add_depot: Whether to add a depot node to the graph.
        negative: Whether to negate the adjacency matrix.
        np_rng: Random number generator.

    Returns:
        Adjacency matrix.
    """
    if np_rng is None:
        np_rng = np.random.default_rng()

    # If `num_edges` is a percentage, convert to int
    if isinstance(num_edges, float):
        num_edges = int(num_edges * (size * (size - 1)) / 2) if undirected else int(num_edges * size * (size - 1))

    max_edges = int((size * (size - 1)) / 2) if undirected else int(size * (size - 1))
    if num_edges >= 0 and num_edges < max_edges:
        adj_matrix = np.zeros((size, size), dtype=int)
        possible_edges = [(i, j) for i in range(0, size) for j in range(0, size) if i != j]

        if undirected:
            possible_edges = [(i, j) for i, j in possible_edges if i < j]

        selected = np_rng.choice(len(possible_edges), num_edges, replace=False)
        for edge_index in selected:
            i, j = possible_edges[edge_index]
            adj_matrix[i, j] = 1
            if undirected:
                adj_matrix[j, i] = 1

        if add_depot:
            adj_matrix = np.pad(adj_matrix, ((1, 0), (1, 0)), mode="constant", constant_values=1)
            adj_matrix[0, 0] = 0
    else:
        adj_matrix = np.ones((size + 1, size + 1), dtype=int) if add_depot else np.ones((size, size), dtype=int)
        np.fill_diagonal(adj_matrix, 0)

    return adj_matrix if not negative else 1 - adj_matrix


def get_edge_idx_dist(
    dist_matrix: np.ndarray, num_edges: Union[int, float], add_depot: bool = True, undirected: bool = True
) -> np.ndarray:
    """
    Generates edge indices based on shortest distances in the distance matrix.

    Args:
        dist_matrix: Distance matrix.
        num_edges: Number of edges to add to the graph.
        add_depot: Whether to add a depot node to the graph.
        undirected: Whether the graph should be undirected.

    Returns:
        Edge indices.
    """
    assert not undirected or np.allclose(dist_matrix, dist_matrix.T), (
        "Distance matrix must be symmetric for an undirected graph"
    )
    size = len(dist_matrix)

    if isinstance(num_edges, float):
        num_edges = int(num_edges * (size * (size - 1)) / 2) if undirected else int(num_edges * size * (size - 1))

    max_edges = int((size * (size - 1)) / 2) if undirected else int(size * (size - 1))
    if num_edges >= 0 and num_edges < max_edges:
        if undirected:
            upper_tri_idx = np.triu_indices_from(dist_matrix, k=1)
            upper_tri_dist = dist_matrix[upper_tri_idx]

            sorted_indices = np.argsort(upper_tri_dist)
            selected_indices = sorted_indices[:num_edges]
            edges = np.array(
                (
                    upper_tri_idx[0][selected_indices],
                    upper_tri_idx[1][selected_indices],
                ),
                dtype=int,
            )

            if add_depot:
                d_edges = [[0] * size, list(range(1, size + 1))]
                selected = np.hstack((edges + 1, d_edges, [d_edges[1], d_edges[0]])).T
            else:
                selected = edges.T
            return selected[np.lexsort((selected[:, 1], selected[:, 0]))].T
        else:
            # Mask diagonals to exclude them from edge count
            temp_dist = dist_matrix.copy()
            np.fill_diagonal(temp_dist, np.inf)
            sorted_dist = np.sort(temp_dist.flatten())
            thresh = sorted_dist[num_edges - 1]
            adj_matrix = (dist_matrix <= thresh).astype(int)

            if add_depot:
                adj_matrix = np.vstack((np.ones(size, dtype=int), adj_matrix))
                adj_matrix = np.hstack((np.ones(size + 1, dtype=int)[:, None], adj_matrix))

            np.fill_diagonal(adj_matrix, 0)
            return adj_to_idx(adj_matrix, negative=False)
    else:
        adj_matrix = np.ones((size + 1, size + 1), dtype=int) if add_depot else np.ones((size, size), dtype=int)
        np.fill_diagonal(adj_matrix, 0)
        return adj_to_idx(adj_matrix, negative=False)


def get_adj_knn(
    dist_mat: np.ndarray, k_neighbors: Union[int, float], add_depot: bool = True, negative: bool = True
) -> np.ndarray:
    """
    Generates an adjacency matrix based on K-Nearest Neighbors.

    Args:
        dist_mat: Distance matrix.
        k_neighbors: Number of neighbors to add to the graph.
        add_depot: Whether to add a depot node to the graph.
        negative: Whether to negate the adjacency matrix.

    Returns:
        Adjacency matrix.
    """
    size = len(dist_mat)

    if isinstance(k_neighbors, float):
        k_neighbors = int(size * k_neighbors)

    if k_neighbors >= size - 1 or k_neighbors == -1:
        W = np.zeros((size, size))
    else:
        W_val = np.array(dist_mat)
        W = np.ones((size, size))
        knns = np.argpartition(W_val, kth=k_neighbors, axis=-1)[:, k_neighbors::-1]
        for idx in range(size):
            W[idx][knns[idx]] = 0

    if add_depot:
        W = np.pad(W, ((1, 0), (1, 0)), mode="constant", constant_values=0)

    np.fill_diagonal(W, 1)
    return W if negative else 1 - W


def get_adj_osm(coords: Any, size: int, args: list, add_depot: bool = True, negative: bool = True) -> np.ndarray:
    """
    Computes an adjacency matrix via OpenStreetMap for given coordinates.

    Args:
        coords: Coordinates of the nodes.
        size: Number of nodes in the graph.
        args: Arguments for the OpenStreetMap graph.
        add_depot: Whether to add a depot node to the graph.
        negative: Whether to negate the adjacency matrix.

    Returns:
        Adjacency matrix.
    """
    if not OSMNX_AVAILABLE:
        raise ImportError(
            "osmnx is required for OSM graph generation. Install it with 'pip install wsmart-route[geo]'."
        )

    G, *args = args
    assert ox is not None
    assert isinstance(G, nx.MultiDiGraph)
    df = coords.copy() if coords.shape[0] == size else coords.copy().drop(index=1)
    assert df.shape[0] == size

    df["OSM_Node"] = df.apply(lambda row: ox.distance.nearest_nodes(G, row["Lng"], row["Lat"]), axis=1)
    adj_matrix = nx.to_numpy_array(  # pyrefly: ignore[no-matching-overload]  # networkx dispatcher stub rejects list[Any]
        G, nodelist=cast(list[Any], df["OSM_Node"].tolist()), dtype=int
    )

    if add_depot:
        adj_matrix = np.pad(adj_matrix, ((1, 0), (1, 0)), mode="constant", constant_values=1)

    np.fill_diagonal(adj_matrix, 0)
    return adj_matrix if not negative else 1 - adj_matrix
