"""
Facade for graph utilities.

Attributes:
    find_longest_path: Find the longest path in a DAG represented by a distance matrix.
    adj_to_idx: Converts dimensionality-2 adjacency matrix to a [2, num_edges] index array.
    idx_to_adj: Converts edge index array back to an adjacency matrix.
    tour_to_adj: Converts a sequence of nodes (tour) into an adjacency matrix.
    sort_by_pairs: Sorts edge indices by their linear index (row * size + col).
    generate_adj_matrix: Generates a random adjacency matrix.
    get_edge_idx_dist: Generates edge indices based on shortest distances in the distance matrix.
    get_adj_knn: Generates an adjacency matrix based on K-Nearest Neighbors.
    get_adj_osm: Computes an adjacency matrix via OpenStreetMap for given coordinates.
    apply_edges: Sparsifies distance matrix and computes shortest paths.
    get_paths_between_states: Constructs a nested list of paths between all pairs of bins.

Example:
    find_longest_path(dist_matrix)
    adj_to_idx(adj_matrix)
    idx_to_adj(edge_idx)
    tour_to_adj(tour_nodes)
    sort_by_pairs(graph_size, edge_idx)
    generate_adj_matrix(size, num_edges)
    get_edge_idx_dist(dist_matrix, num_edges)
    get_adj_knn(dist_matrix, k_neighbors)
    get_adj_osm(coords, size, args)
    apply_edges(dist_matrix, edge_thresh, edge_method)
    get_paths_between_states(n_bins, shortest_paths)
"""

from logic.src.utils.registry import lazy_getattr

_LAZY = {
    "find_longest_path": "logic.src.utils.graph.algorithms:find_longest_path",
    "adj_to_idx": "logic.src.utils.graph.conversion:adj_to_idx",
    "idx_to_adj": "logic.src.utils.graph.conversion:idx_to_adj",
    "sort_by_pairs": "logic.src.utils.graph.conversion:sort_by_pairs",
    "tour_to_adj": "logic.src.utils.graph.conversion:tour_to_adj",
    "generate_adj_matrix": "logic.src.utils.graph.generation:generate_adj_matrix",
    "get_adj_knn": "logic.src.utils.graph.generation:get_adj_knn",
    "get_adj_osm": "logic.src.utils.graph.generation:get_adj_osm",
    "get_edge_idx_dist": "logic.src.utils.graph.generation:get_edge_idx_dist",
    "apply_edges": "logic.src.utils.graph.network_utils:apply_edges",
    "get_paths_between_states": "logic.src.utils.graph.network_utils:get_paths_between_states",
}



def __getattr__(name):
    return lazy_getattr(globals(), _LAZY, name)
