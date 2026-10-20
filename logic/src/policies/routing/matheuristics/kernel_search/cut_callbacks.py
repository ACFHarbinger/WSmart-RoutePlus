"""Gurobi cut callbacks for the Kernel Search matheuristic.

Holds the lazy-constraint and user-cut callbacks split out of ``solver``:
route-capacity checks, DFJ subtour elimination, fractional-subtour separation
and the root-node relaxation harvest callback.
"""

import networkx as nx
from gurobipy import GRB, quicksum


def _check_route_capacity(model, G, x_vars, component):
    """
    Check if any routes from depot violate capacity constraints and add lazy cuts if needed.

    Args:
        model: Gurobi model instance
        G: NetworkX graph of active edges
        x_vars: Dictionary of edge variables
        component: Set of nodes in the current component (must contain depot 0)
    """
    if 0 not in component:
        return

    # Find the specific paths originating from the depot within this component
    depot_edges = [j for j in G.neighbors(0)]
    for start_node in depot_edges:
        # Trace the route
        route_nodes = set()
        route_edges = []
        curr = start_node
        prev = 0
        route_waste = 0.0

        while curr != 0:
            route_nodes.add(curr)
            route_edges.append((prev, curr) if (prev, curr) in x_vars else (curr, prev))
            route_waste += model._wastes.get(curr, 0.0)

            # Move to next node
            neighbors = [n for n in G.neighbors(curr) if n != prev]
            if not neighbors:
                break  # Dead end
            prev = curr
            curr = neighbors[0]

        route_edges.append((prev, 0) if (prev, 0) in x_vars else (0, prev))

        # Add a capacity cut if the route violates the limit
        if route_waste > model._capacity:
            route_vars = [x_vars[e] for e in route_edges if e in x_vars]
            if route_vars:
                model.cbLazy(quicksum(route_vars) <= len(route_vars) - 1)


def _dfj_subtour_elimination_callback(model, where):
    """
    Gurobi callback to dynamically add Dantzig-Fulkerson-Johnson (DFJ) subtour elimination cuts.

    This callback is triggered at integer solutions (MIPSOL) and uses NetworkX to detect
    connected components. If a component S does not contain the depot (node 0), it injects
    a lazy constraint enforcing that at least one edge must leave the subtour:

        sum_{i in S, j in S, i != j} x[i,j] <= |S| - 1

    Mathematical Foundation:
        The DFJ formulation provides an exponentially-sized family of constraints that
        completely eliminate all subtours. Unlike MTZ, which uses auxiliary continuous
        variables with weak LP bounds, DFJ cuts directly target the combinatorial structure
        of the tour polytope, yielding significantly tighter LP relaxations.

    Args:
        model: The Gurobi model instance (passed automatically by Gurobi).
        where: Callback code indicating when the callback is being invoked.
    """
    if where == GRB.Callback.MIPSOL:
        # Retrieve the current integer solution values
        x_vars = model._x_vars
        num_nodes = model._num_nodes

        # Build a graph from active edges (x[i,j] > 0.5 in the current solution)
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))

        for (i, j), var in x_vars.items():
            val = model.cbGetSolution(var)
            if val > 0.5:
                G.add_edge(i, j)

        # Find all connected components
        components = list(nx.connected_components(G))

        # For each component that does NOT contain the depot, add a DFJ cut
        for component in components:
            if 0 not in component and len(component) >= 2:
                # Component is a subtour - add lazy constraint
                # sum_{i,j in S} x[i,j] <= |S| - 1
                subtour_edges = []
                for i in component:
                    for j in component:
                        if i != j and (i, j) in x_vars:
                            subtour_edges.append(x_vars[(i, j)])

                if subtour_edges:
                    model.cbLazy(quicksum(subtour_edges) <= len(component) - 1)

        # Check capacity for routes connected to the depot
        for component in components:
            _check_route_capacity(model, G, x_vars, component)


def _separate_fractional_subtours(model, x_vars, num_nodes):
    """Identify and inject fractional subtour elimination cuts (User Cuts).

    Uses Max-Flow/Min-Cut to find violated directed cuts.

    Args:
        model: Gurobi model with attached _y_vars attribute.
        x_vars: Edge flow variable dictionary keyed by (i, j) tuples.
        num_nodes: Total number of nodes including the depot.
    """
    # 1. Build a capacity graph from the fractional solution
    G = nx.DiGraph()
    for (i, j), var in x_vars.items():
        val = model.cbGetNodeRel(var)
        if val > 1e-4:
            G.add_edge(i, j, capacity=val)

    # 2. Check cuts from depot (0) to each customer i
    # Requirement: min-cut(0 -> i) >= y_i
    for i in range(1, num_nodes):
        if i not in G:
            continue

        y_val = model.cbGetNodeRel(model._y_vars[i])
        if y_val < 1e-4:
            continue

        cut_value, (reachable, _non_reachable) = nx.minimum_cut(G, 0, i)

        if cut_value < y_val - 1e-4:
            # Violated cut: sum_{u in S, v not in S} x[u,v] >= y[i]
            # S is the set containing the depot (0)
            S = reachable
            cut_edges = []
            for u in S:
                if u in G:
                    for v in G.neighbors(u):
                        if v not in S:
                            cut_edges.append(x_vars[(u, v)])

            if cut_edges:
                model.cbCut(quicksum(cut_edges) >= model._y_vars[i])


def _root_node_callback(model, where):
    """Gurobi callback to harvest the root node relaxation values.

    Enhanced to add fractional subtour elimination user cuts before harvesting.

    Args:
        model: Gurobi model with _x_vars, _num_nodes, and _all_vars_list attributes.
        where: Gurobi callback code indicating the invocation point.
    """
    if where == GRB.Callback.MIPNODE:
        status = model.cbGet(GRB.Callback.MIPNODE_STATUS)
        if status == GRB.OPTIMAL and model.cbGet(GRB.Callback.MIPNODE_NODCNT) == 0:
            # 1. Separate fractional subtours at the root node
            _separate_fractional_subtours(model, model._x_vars, model._num_nodes)

            # 2. Harvest relaxation values (now reflecting the subtour cuts)
            model._node_rel = model.cbGetNodeRel(model._all_vars_list)
            model.terminate()
