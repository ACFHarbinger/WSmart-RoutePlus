# Combinatorial Optimization: A Comprehensive Survey of Problems, Algorithms, and Future Directions

> **Version**: 1.0 — April 2026  
> **Scope**: Foundational theory, problem taxonomy, algorithmic history, state-of-the-art methods, and future directions across the full field of combinatorial and routing optimization.

---

## Table of Contents

1. [Mathematical Foundations of Optimization](#1-mathematical-foundations-of-optimization)
2. [Taxonomy of Optimization Paradigms](#2-taxonomy-of-optimization-paradigms)
3. [Routing Problem Taxonomy and Formulations](#3-routing-problem-taxonomy-and-formulations)
4. [History of Exact Solvers](#4-history-of-exact-solvers)
5. [History of Classical Heuristics and Meta-Heuristics](#5-history-of-classical-heuristics-and-meta-heuristics)
6. [Mathematical Search Operators](#6-mathematical-search-operators)
7. [Acceptance Criteria in Trajectory Search](#7-acceptance-criteria-in-trajectory-search)
8. [Mandatory Node Selection for Multi-Period Problems](#8-mandatory-node-selection-for-multi-period-problems)
9. [Neural Combinatorial Optimization](#9-neural-combinatorial-optimization)
10. [Future Research Directions](#10-future-research-directions)

---

## 1. Mathematical Foundations of Optimization

### 1.1 General Formulation

Mathematical optimization is the discipline of selecting the best element from a feasible set $\mathcal{X}$ according to a well-defined criterion $f$. In its most general form:

$$\min_{x \in \mathcal{X}} f(x) \quad \text{s.t.} \quad g_i(x) \le 0,\; i=1,\dots,m,\quad h_j(x) = 0,\; j=1,\dots,p$$

The nature of $\mathcal{X}$, $f$, and the constraints fundamentally determines which algorithms are applicable and what theoretical guarantees exist.

### 1.2 Discrete vs. Continuous Domains

**Combinatorial Optimization (CO)** operates over a discrete, finite (or countably infinite) feasible set — permutations, binary vectors, graphs, or integer lattices. Because the feasible set is discrete, standard gradient-based calculus is invalid: a small perturbation from a valid integer vector does not yield another feasible integer vector.

**Continuous Optimization** operates over uncountably infinite feasible sets defined by real-valued variables. When the objective and constraints are convex, any locally optimal solution is also globally optimal — a fundamental advantage over discrete problems.

The **theoretical bridge** between the two domains is *relaxation*: replacing the integrality constraint $x \in \{0,1\}^n$ with the continuous interval $x \in [0,1]^n$. The resulting LP or QP relaxation provides a lower bound on the integer optimum. The quality of this bound — measured by the *integrality gap* — largely governs the difficulty of solving integer programs.

### 1.3 Complexity Classes

Most combinatorial optimization problems of practical interest are **NP-hard**, meaning no polynomial-time algorithm is known to solve all instances optimally. Key complexity milestones:

| Year | Result |
|------|--------|
| 1971 | Cook-Levin theorem: SAT is NP-complete |
| 1972 | Karp: 21 NP-complete problems including TSP decision version |
| 1979 | Garey & Johnson: NP-hardness of VRP |
| 1982 | Dantzig-Wolfe decomposition for large-scale LPs |
| 1990s | Branch-and-Price scales to VRP with hundreds of customers |
| 2000s | BPC (Branch-Price-Cut) achieves CVRP instances with 1000+ nodes |

### 1.4 Optimality Conditions

For a **continuous** nonlinear program, the **Karush-Kuhn-Tucker (KKT) conditions** are necessary for local optimality at $x^*$:

$$\nabla f(x^*) + \sum_{i} \lambda_i \nabla g_i(x^*) + \sum_j \mu_j \nabla h_j(x^*) = 0$$
$$\lambda_i g_i(x^*) = 0, \quad \lambda_i \ge 0 \;\forall i$$

For **integer programs**, optimality is certified by the LP relaxation bound matching the integer solution value, or by complete branch-and-bound enumeration.

---

## 2. Taxonomy of Optimization Paradigms

### 2.1 Overview Table

| Paradigm | Domain | Uncertainty | Key Property | Representative Problems |
|----------|--------|-------------|--------------|------------------------|
| **Linear Programming (LP)** | Continuous | None | Convex polytope, polynomial-time | Network flow, transportation |
| **Integer Programming (IP/MILP)** | Mixed integer | None | NP-hard, LP relaxation bound | VRP, TSP, facility location |
| **Nonlinear Programming (NLP)** | Continuous | None | Non-convex in general | Optimal power flow, trajectory |
| **Convex Programming** | Continuous | None | Global optimum = local optimum | SVM, LASSO, portfolio |
| **Semidefinite Programming (SDP)** | Continuous matrix | None | Cone constraint $X \succeq 0$ | Max-Cut relaxation |
| **Stochastic Programming (SP)** | Mixed | Distributional | Two-stage recourse | Stochastic VRP, SIRP |
| **Robust Optimization (RO)** | Mixed | Set-based | Minimax worst-case | Robust VRP, network design |
| **Distributionally Robust (DRO)** | Mixed | Ambiguity set | Wasserstein or KL ball | OOD generalization |
| **Dynamic Programming (DP)** | Sequential | Markov | Bellman's principle | Shortest path, IRP |
| **Bilevel Optimization** | Hierarchical | None | Leader-follower | Meta-learning, HPO |
| **Multi-Objective** | Mixed | None | Pareto front | Bi-criteria VRP |
| **Combinatorial Optimization** | Discrete | None | NP-hard in general | TSP, VRP, knapsack |

### 2.2 Stochastic Programming

The classic **two-stage recourse** model:

$$\min_{x \in \mathcal{X}} c^\top x + \mathbb{E}_\omega[Q(x, \omega)]$$

where the recourse function $Q(x, \omega) = \min_{y \in \mathcal{Y}(x,\omega)} q(\omega)^\top y$ captures the cost of corrective action after uncertainty $\omega$ is revealed. This directly models VRP with stochastic demands, inventory routing under uncertain consumption, and dynamic vehicle dispatch.

**Progressive Hedging (PH)** decomposes the problem across scenarios $s \in \mathcal{S}$ by relaxing non-anticipativity and adding an augmented Lagrangian penalty:

$$\min_{x_s} \left\{ p_s f_s(x_s) + w_s^{(k)\top} x_s + \frac{\rho}{2}\|x_s - z^{(k-1)}\|^2 \right\}$$

with consensus update $z^{(k)} = \sum_s p_s x_s^{(k)}$.

### 2.3 Robust Optimization

The **minimax robust counterpart** for uncertain parameter $u \in \mathcal{U}$:

$$\min_x \max_{u \in \mathcal{U}} f(x, u) \quad \text{s.t.} \quad g(x, u) \le 0 \;\forall u \in \mathcal{U}$$

The **Bertsimas-Sim** budget uncertainty set $\mathcal{U}_\Gamma = \{u : \|u\|_\infty \le 1,\, \mathbf{1}^\top u \le \Gamma\}$ allows at most $\Gamma$ parameters to deviate simultaneously, yielding a tractable LP reformulation with a single additional variable per constraint.

### 2.4 Distributionally Robust Optimization

The **Wasserstein ambiguity set** of radius $\epsilon$ centered on empirical distribution $\hat{\mathbb{P}}_N$:

$$\mathcal{B}_\epsilon(\hat{\mathbb{P}}_N) = \left\{ \mathbb{P} : W_p(\mathbb{P}, \hat{\mathbb{P}}_N) \le \epsilon \right\}$$

The DRO objective $\min_x \sup_{\mathbb{P} \in \mathcal{B}_\epsilon} \mathbb{E}_\mathbb{P}[f(x, \xi)]$ connects directly to regularized ML: for linear regression under $W_1$, the Wasserstein DRO objective is equivalent to LASSO regularization.

### 2.5 Multi-Objective Optimization

When objectives $f_1(x), \dots, f_k(x)$ conflict, the **Pareto dominance** relation defines optimality: solution $x$ dominates $y$ if $f_i(x) \le f_i(y)$ for all $i$ and strict inequality holds for at least one. The **Pareto front** $\mathcal{P}^*$ is the set of non-dominated solutions, tracing the efficient frontier between competing criteria (e.g., cost vs. environmental impact in green VRP).

---

## 3. Routing Problem Taxonomy and Formulations

### 3.1 The TSP Family

#### 3.1.1 Symmetric TSP (STSP)

The foundational routing problem. Given a complete undirected graph $G = (V, E)$ with $c_{ij} = c_{ji}$, find the minimum-cost Hamiltonian cycle. ILP formulation with binary edge variables $x_{ij} \in \{0,1\}$:

$$\min \sum_{i \neq j} c_{ij} x_{ij}$$

Subject to degree constraints ($\sum_j x_{ij} = 2$ for all $i$) and **DFJ subtour elimination** for every proper subset $S \subset V$:

$$\sum_{i \in S,\, j \notin S} x_{ij} \ge 2$$

The DFJ constraints are exponential in number but are separated lazily by solving a minimum cut problem at each LP node.

#### 3.1.2 TSP Variants

| Variant | Additional Feature | Key Constraint |
|---------|--------------------|----------------|
| **ATSP** | Directed graph, $c_{ij} \neq c_{ji}$ | Directed subtour elimination |
| **GTSP** | Visit exactly one node per cluster | $\sum_{j \in C_k} y_j = 1\; \forall k$ |
| **PCTSP** | Nodes have profits $p_i$, penalties $\gamma_i$ | $\sum p_i y_i \ge P_{\min}$ |
| **TSPTW** | Hard time windows $[e_i, l_i]$ | $e_i \le t_i \le l_i$ |
| **CTSP** | Full cluster visit before next cluster | Intra-cluster sequence + inter-cluster order |

The **Prize-Collecting TSP** objective:
$$\min \sum_{(i,j)} c_{ij} x_{ij} + \sum_{i \in V} \gamma_i (1 - y_i) \quad \text{s.t.} \quad \sum_i p_i y_i \ge P_{\min}$$

The **TSPTW** arrival-time constraint:
$$t_i + s_i + c_{ij} \le t_j + M(1 - x_{ij}), \quad e_i \le t_i \le l_i$$

### 3.2 The VRP Family

#### 3.2.1 Capacitated VRP (CVRP)

A homogeneous fleet of $K$ vehicles, each with capacity $Q$, serves customers $V_c$ with demands $q_i$. The canonical **fractional cut** inequality for any subset $S \subseteq V_c$:

$$\sum_{i \in S,\, j \notin S} x_{ij} \ge 2 \left\lceil \frac{\sum_{i \in S} q_i}{Q} \right\rceil$$

This generalizes TSP subtour elimination to enforce that enough vehicles cross any cut to carry the total demand within $S$.

#### 3.2.2 VRP Family Table

| Problem | Fleet | Constraint Added | Objective |
|---------|-------|-----------------|-----------|
| **CVRP** | Homogeneous | Capacity $Q$ | Min cost |
| **VRPTW** | Homogeneous | Capacity + time windows | Min cost |
| **OVRP** | Homogeneous | No depot return | Min cost |
| **MDVRP** | Multi-depot | Depot assignment | Min cost |
| **VRPB** | Homogeneous | Linehaul before backhaul | Min cost |
| **VRPP/OP** | Single/fleet | Budget $T_{\max}$, profits $p_i$ | Max profit |
| **TOP** | Fleet | Per-vehicle budget | Max profit |
| **SDVRP** | Homogeneous | Split deliveries allowed | Min cost |
| **HFVRP** | Heterogeneous | Per-vehicle cost $c_{ij}^k$ | Min cost |
| **PVRP** | Homogeneous | Multi-period frequency | Min cost |
| **MPVRPP** | Fleet | Multi-period profits | Max cumulative profit |
| **GVRP/EVRP** | Electric/fuel | Energy budget, recharge stations | Min cost |
| **DVRP** | Fleet | Online requests | Min expected cost |
| **SVRP** | Fleet | Stochastic demands | Min expected cost |
| **RVRP** | Fleet | Adversarial demands $q \in \mathcal{U}$ | Minimax cost |

#### 3.2.3 VRPP / Orienteering Problem (OP)

$$\max \sum_{i \in V} p_i y_i \quad \text{s.t.} \quad \sum_{(i,j)} c_{ij} x_{ij} \le T_{\max}, \quad y_i \le \sum_j x_{ji}$$

The binary variable $y_i$ activates node $i$; the budget $T_{\max}$ enforces the routing constraint. The Team Orienteering Problem (TOP) extends this to $K$ vehicles each with individual budget $T_{\max}^k$:

$$\max \sum_{i \in V} p_i y_i \quad \text{s.t.} \quad \sum_{(i,j)} c_{ij} x_{ij}^k \le T_{\max}^k \;\forall k, \quad \sum_k y_{ik} \le 1 \;\forall i$$

#### 3.2.4 Periodic VRP (PVRP)

A multi-period generalization over horizon $H$. Each customer $i$ selects a visit-day pattern $p \in P_i$:

$$\sum_{p \in P_i} y_{ip} = 1 \;\forall i, \quad z_{it} = \sum_{p \in P_i} a_{pt} y_{ip}$$

where $z_{it} \in \{0,1\}$ triggers customer $i$'s inclusion in day $t$'s CVRP subproblem.

#### 3.2.5 Multi-Period VRP with Profits (MPVRPP)

$$\max \sum_{t \in H} \sum_{i \in V} p_{it} y_{it}$$

subject to daily fleet capacity constraints and, crucially, the dependency between days: the profit $p_{it}$ may depend on the cumulative fill level $c_{it}$ which evolves as $c_{it} = c_{i,t-1} + \mu_{it} - q_{it} y_{it}$ where $\mu_{it}$ is the daily accumulation rate.

### 3.3 Stochastic and Dynamic Extensions

#### 3.3.1 Stochastic VRP (SVRP)

Demands $q_i$ are random variables. The objective minimizes expected routing cost including failure penalties:

$$\min \sum_{(i,j)} c_{ij} x_{ij} + \mathbb{E}\left[\text{Failure Cost}(x, \tilde{q})\right]$$

A **failure** occurs when a vehicle's realized demand exceeds remaining capacity mid-route, requiring a detour to the depot.

#### 3.3.2 Robust VRP (RVRP)

The minimax formulation:

$$\min_x \max_{q \in \mathcal{U}} \left\{ \text{Cost}(x, q) \;\middle|\; \text{Constraints}(x, q) \right\}$$

With the **Bertsimas-Sim** budget set, the robust CVRP admits a tractable LP reformulation.

### 3.4 Pickup and Delivery Problems

#### 3.4.1 Standard PDP

Each request is a paired pickup-delivery tuple $(i^+, i^-)$. Precedence: $t_{i^+} \le t_{i^-}$, and both must be served by the same vehicle:

$$\sum_j x_{i^+j}^k = \sum_j x_{i^-j}^k \; \forall k$$

#### 3.4.2 Dial-a-Ride Problem (DARP)

The paratransit variant adds ride-time limits $L_i$ bounding maximum in-vehicle duration:

$$t_{i^-} - t_{i^+} \le L_i$$

### 3.5 Arc Routing Problems

| Problem | Graph | Demand on | Objective |
|---------|-------|-----------|-----------|
| **CPP** | Undirected | All edges | Min-cost Euler tour |
| **RPP** | Undirected | Subset $E_R \subset E$ | Min-cost traversal of $E_R$ |
| **CARP** | Undirected | Required edges $q_e$ | Min cost, fleet capacity $Q$ |
| **PCARP** | Undirected | Multi-period frequencies | Min cost over horizon |

**Chinese Postman Problem**: If the graph is Eulerian (all nodes even degree), the tour cost equals total edge weight; otherwise, minimum-weight perfect matching over odd-degree nodes determines deadheading overhead.

**Rural Postman Problem** LP:

$$\min \sum_{(i,j) \in E} c_{ij} x_{ij} + \sum_{(i,j) \in E_R} c_{ij}, \quad \sum_j (x_{ij} - x_{ji}) = 0 \;\forall i$$

### 3.6 Location Routing Problems

#### 3.6.1 Standard LRP

Jointly optimizes depot opening decisions $z_j \in \{0,1\}$ and routing:

$$\min \sum_{j \in D} F_j z_j + \sum_k \sum_{(i,j)} c_{ij} x_{ij}^k$$

with routing-depot coupling: $\sum_i x_{ji}^k \le z_j \;\forall j \in D,\, \forall k$.

#### 3.6.2 Two-Echelon VRP (2E-VRP)

A tiered urban distribution network. The 1st echelon routes large vehicles from depots to intermediate satellites; the 2nd echelon routes smaller vehicles from satellites to customers. Flow conservation constraints couple the echelons, and satellite inventory must be replenished before downstream routes depart.

### 3.7 Inventory Routing Problems (IRP)

The IRP integrates **Vendor-Managed Inventory (VMI)** with vehicle routing, jointly optimizing delivery quantities and routes over a multi-period horizon $\mathcal{T}$.

#### 3.7.1 Deterministic IRP

$$\min \sum_{t \in \mathcal{T}} \left( \sum_{(i,j) \in A} c_{ij} x_{ijt} + \sum_{i \in V} h_i I_{it} \right)$$

**Inventory flow conservation**:
$$I_{it} = I_{i,t-1} + q_{it} - d_{it}, \quad 0 \le I_{it} \le C_i \;\forall t \in \mathcal{T}$$

where $h_i$ is the unit holding cost, $q_{it}$ the delivered quantity, and $d_{it}$ the deterministic consumption.

#### 3.7.2 IRP Variant Table

| Variant | Additional Feature | Key Modification |
|---------|--------------------|-----------------|
| **SIRP** | Stochastic demand $\tilde{d}_{it}$ | Chance constraint: $\Pr[I_{it} < 0] \le \alpha$ |
| **MCIRP** | Multiple commodities | Per-commodity tracking $q_{it}^p$ |
| **RIRP** | Adversarial demand $d \in \mathcal{U}$ | Minimax inventory feasibility |
| **IRP-LT** | Lateral transshipment | Transfer variables $f_{ijt}$ between customers |

---

## 4. History of Exact Solvers

### 4.1 Linear Programming: The Simplex Era (1947–1980)

| Year | Milestone |
|------|-----------|
| 1947 | Dantzig's Simplex Algorithm — solves LPs in practice exponentially fast |
| 1955 | Charnes & Cooper: LP for transportation and assignment |
| 1960 | Gomory: cutting planes for integer programs |
| 1963 | Dantzig, Fulkerson, Johnson: first TSP subtour elimination |
| 1979 | Khachiyan: Ellipsoid method proves LP is polynomial |
| 1984 | Karmarkar: Interior-point method — polynomial and practically competitive |

### 4.2 Branch-and-Bound (1960s)

Developed by **Land and Doig (1960)** and formalized by **Dakin (1965)**. The algorithm partitions the feasible region into subproblems (branching), computes LP relaxation lower bounds at each node, and prunes subproblems whose bound exceeds the best-known integer solution.

**Key insight**: The LP relaxation provides a lower bound because the LP optimal cost $z_{\text{LP}} \le z^*_{\text{IP}}$ (for minimization, integrality gaps arise from fractional LP solutions).

### 4.3 Cutting Planes (1960–1990)

**Gomory cuts (1960)** for pure integer programs: from a fractional LP row $\sum_j \bar{a}_{ij} x_j = \bar{b}_i$, the Gomory cut $\sum_j \{\bar{a}_{ij}\} x_j \ge \{\bar{b}_i\}$ (where $\{\cdot\}$ denotes fractional part) is valid and tightens the relaxation.

**TSP-specific cuts**:
- **Subtour Elimination Constraints (SEC)**: Dantzig-Fulkerson-Johnson, 1954
- **Comb Inequalities**: Chvátal, 1973 — exponentially tighter than SECs
- **Blossom Inequalities**: Padberg-Rao, 1982 — based on T-joins in graphs

### 4.4 Column Generation and Branch-and-Price (1960–1990)

**Dantzig-Wolfe decomposition (1960)** reformulates large LPs by generating columns (routes) on demand. Applied to VRP by **Desrochers, Desrosiers & Solomon (1992)**:

The **Restricted Master Problem (RMP)** selects routes from a generated pool:
$$\min \sum_{r} c_r \lambda_r \quad \text{s.t.} \quad \sum_r a_{ir} \lambda_r \ge 1 \;\forall i, \quad \lambda_r \ge 0$$

The **pricing subproblem** finds a route with **negative reduced cost**:
$$\bar{c}_r = c_r - \sum_{i \in V} \pi_i a_{ir} - \mu < 0$$

For VRPTW, this subproblem is an **ESPPRC** (Elementary Shortest Path with Resource Constraints) — NP-hard but solvable in practice via dynamic programming with $ng$-routes relaxation.

**Branch-and-Price** integrates column generation into the B&B tree with **Ryan-Foster branching**: branch on pairs $(i,j)$ requiring them on the same or different routes.

### 4.5 Branch-and-Price-and-Cut (BPC): State-of-the-Art Exact Method

**BPC** unifies column generation, cutting planes, and branch-and-bound:

| Component | Purpose | Key Technique |
|-----------|---------|---------------|
| Column generation | Route enumeration | Pulse algorithm, ng-routes |
| Cutting planes | Tighten LP relaxation | Subset-Row Cuts (SRC), rank-1 cuts |
| Branching | Integrality enforcement | Ryan-Foster, strong branching |
| Dominance rules | Prune pricing labels | Resource dominance, completion bounds |

**Subset-Row Cuts** (Jepsen et al., 2008): for a set of three customers $\{i,j,k\}$:
$$\sum_{r: |\{i,j,k\} \cap S_r| \ge 2} \lambda_r \le 1$$

These cuts force fractional route variables toward integrality, dramatically reducing the B&B tree size.

**Historical BPC milestones**:

| Year | Authors | Achievement |
|------|---------|-------------|
| 1992 | Desrochers et al. | First BP for VRPTW, solves 100 customers |
| 1998 | Fukasawa et al. | Rounded capacity cuts in BP |
| 2008 | Jepsen et al. | Subset-Row Cuts for CVRP |
| 2011 | Contardo & Martinelli | BPC for CVRP, instances 1000+ customers |
| 2014 | Poggi & Uchoa | Enhanced BPC with limited memory cuts |
| 2020 | Pessoa et al. | BPC for generic VRP, VRPTW 1000 customers |
| 2024 | Lam et al. | Neural-guided BPC with ML branching |

### 4.6 Other Exact Frameworks

#### Benders Decomposition (1962)

Decomposes the problem into a master (integer variables) and a subproblem (continuous variables given master solution). **Benders cuts** $\theta \ge v^\top(b - Bx)$ are iteratively added from dual subproblem solutions.

**Integer L-Shaped (Laporte & Louveaux, 1993)** extends to stochastic MILPs with integer recourse:
$$\theta \ge (Q(x^k) - L)\!\left(\sum_{i \in S^k} x_i - \sum_{i \notin S^k} x_i - |S^k| + 1\right) + L$$

#### Logic-Based Benders (LBBD)

Removes the LP duality requirement by using Constraint Programming for subproblems, generating cuts via logical inference. Particularly powerful for routing problems with complex temporal constraints.

#### CP-SAT (Constraint Programming + Boolean Satisfiability)

Integrates finite-domain constraint propagation with modern SAT engines via **lazy clause generation**. Capacity violations and routing constraints are translated dynamically into SAT clauses (nogoods), providing a competitive alternative to MIP-based exact methods on heavily constrained instances.

---

## 5. History of Classical Heuristics and Meta-Heuristics

### 5.1 Construction Heuristics (1950s–1970s)

| Algorithm | Year | Problem | Complexity | Approx. Ratio |
|-----------|------|---------|-----------|---------------|
| **Nearest Neighbor** | 1956 | TSP | $\mathcal{O}(n^2)$ | $\mathcal{O}(\log n)$ |
| **Clarke-Wright Savings** | 1964 | CVRP | $\mathcal{O}(n^2 \log n)$ | — |
| **Greedy (cheapest insertion)** | 1965 | TSP/VRP | $\mathcal{O}(n^2)$ | — |
| **Christofides** | 1976 | TSP | Poly. | $3/2$ |
| **Farthest insertion** | 1965 | TSP | $\mathcal{O}(n^2)$ | $2$ |

**Clarke-Wright Savings**: Begin with $n$ isolated routes $(0, i, 0)$. The saving from merging routes visiting $i$ and $j$:
$$s_{ij} = d_{0,i} + d_{0,j} - d_{i,j}$$
Greedily merge pairs in decreasing savings order, subject to capacity feasibility.

**Christofides Algorithm (1976)**: The only polynomial-time $3/2$-approximation for metric TSP, still standing as a major open problem whether $3/2$ is tight. It computes a minimum spanning tree, finds minimum-weight perfect matching on odd-degree vertices, combines them into an Eulerian multigraph, and extracts a Hamiltonian tour via shortcutting.

### 5.2 Local Search (1960s–1990s)

#### 2-Opt (Lin, 1965)

Remove two non-adjacent edges $(i, i+1)$ and $(j, j+1)$; reconnect as $(i, j)$ and $(i+1, j+1)$ reversing the sub-tour between them:

$$\Delta C = d_{i,j} + d_{i+1,j+1} - d_{i,i+1} - d_{j,j+1}$$

Accept if $\Delta C < 0$. Guaranteed to find a strict 2-opt local minimum in $\mathcal{O}(n^2)$ per pass.

#### Or-Opt (Or, 1976)

Relocate chains of length $L \in \{1, 2, 3\}$ to better positions within the route — strictly more powerful than 2-opt for $L \ge 2$ because it permits non-reversing moves.

#### Lin-Kernighan (LK) Heuristic (Lin & Kernighan, 1973)

A **variable-depth** heuristic that performs sequential $k$-opt exchanges, extending the chain until no further gain is achievable. LK dynamically chooses $k$ at each iteration by following a promising sequence of alternating improving and non-improving exchanges. Near-optimal on TSP instances with hundreds of nodes.

#### Lin-Kernighan-Helsgaun (LKH) (Helsgaun, 2000)

Restricts the candidate set at each LK step using **$\alpha$-nearness bounds** derived from minimum 1-trees, containing the otherwise exponential search space while enabling 5-opt and partition moves. Produces near-optimal solutions on instances with thousands of nodes and remains the dominant TSP heuristic benchmark.

### 5.3 Trajectory Meta-Heuristics

#### Simulated Annealing (Kirkpatrick, Gelatt & Vecchi, 1983)

Inspired by the physical annealing process in metallurgy. Accepts worsening moves with probability governed by the **Boltzmann criterion**:

$$P(\text{accept } s') = \begin{cases} 1 & \text{if } \Delta f \le 0 \\ \exp\!\left(-\dfrac{\Delta f}{T}\right) & \text{if } \Delta f > 0 \end{cases}$$

Temperature $T$ decreases via a geometric cooling schedule $T_{k+1} = \alpha T_k$, $\alpha \in [0.8, 0.999]$. SA converges to the global optimum in theory (logarithmic cooling) but requires prohibitively slow cooling in practice.

#### Tabu Search (Glover, 1986)

A **memory-augmented** trajectory meta-heuristic. The **Tabu List** $\mathcal{T}$ explicitly prohibits recently applied move attributes for a configurable tenure $|\mathcal{T}|$:

$$s \leftarrow \arg\min_{s' \in \mathcal{N}(s) \setminus \mathcal{T}} f(s')$$

The **Aspiration Criterion** overrides tabu status if a move achieves a strict new global best. **Reactive Tabu Search (Battiti & Tecchiolli, 1994)** dynamically adapts tenure: expands upon detecting cycling, contracts during stagnation.

#### Iterated Local Search (Lourenço, Martin & Stützle, 2003)

A simple but powerful meta-heuristic:

$$s_0 \to \text{LocalSearch}(s_0) \to \text{Perturb}(s^*) \to \text{LocalSearch}(s') \to \text{Accept?}(s', s^*)$$

The **Double-Bridge** 4-opt move is the canonical VRP perturbation: it creates a topological change that no 2-opt, 3-opt, or Or-opt can undo, ensuring genuine trajectory escape.

#### Variable Neighborhood Search (Hansen & Mladenović, 1997)

Formalizes diversification through nested, dynamically expanding neighborhoods $\mathcal{N}_1, \dots, \mathcal{N}_{k_{\max}}$:

- **Shaking**: $s' \leftarrow \text{Random}(N_k(s))$
- **Local Search**: $s'' \leftarrow \text{LocalSearch}(s')$
- **Accept / Expand**: reset to $k=1$ on improvement, else $k \leftarrow k+1$

**Variable Neighborhood Descent (VND)** is the deterministic variant using an ordered sequence of neighborhoods.

#### Guided Local Search (Voudouris & Tsang, 1999)

Escapes local optima by augmenting the objective with learned edge penalties:

$$h(s) = f(s) + \lambda \sum_{i=1}^M p_i I_i(s), \quad \text{util}(i) = \frac{c_i}{1 + p_i}$$

Upon reaching a local optimum, the feature with maximum utility is penalized, redirecting subsequent local search to unexplored topological regions.

### 5.4 Large Neighborhood Search (LNS)

**Shaw (1998)** introduced LNS as a destroy-and-repair paradigm: a **ruin operator** ejects a subset $U$ of nodes; a **recreate operator** reinserts them. The move scale is not bounded by neighborhood size, enabling topology bridges unreachable by local operators.

**ALNS (Ropke & Pisinger, 2006)** makes LNS adaptive via a **multi-armed bandit** on destroy-repair operator pairs:

$$p_j = \frac{w_j}{\sum_{k \in \Omega} w_k}, \quad w_{j,t+1} = (1 - \rho)w_{j,t} + \rho c$$

The bandit continuously biases selection toward historically successful operator pairs. **Thompson Sampling** variant maintains a Beta posterior $\text{Beta}(\alpha_i, \beta_i)$, sampling a success probability at each iteration.

**SISR (Christiaens & Vanden Berghe, 2020)** removes sequences (strings) of spatially proximate nodes, inducing capacity "slack" into remaining routes to enable tight reconstruction without infeasibility.

### 5.5 Evolutionary Algorithms

#### Genetic Algorithm (Holland, 1975)

A population of routing solutions evolved via fitness-based selection, problem-specific crossover operators (OX, PMX, ERX), and mutation (swap, inversion). Pure GAs struggle on VRP because crossover of two valid routes frequently produces invalid offspring; repair heuristics add significant overhead.

#### Memetic Algorithm

Hybridizes evolutionary population mechanics with trajectory local search: each offspring undergoes local search intensification before competing for survival. MAs consistently outperform pure GAs on VRP because the local search gradient drives offspring to the local basin floor.

#### Hybrid Genetic Search (HGS) (Vidal, 2012/2022)

A state-of-the-art memetic algorithm combining **biased fitness** to preserve population diversity in highly constrained routing spaces:

$$BF(p) = \mathit{fit}(p) + \left(1 - \frac{nc_{pop}}{N}\right)\Delta(p)$$

where $nc_{pop}$ counts population clones and $\Delta(p)$ measures normalized Hamming distance to nearest population neighbors. Every crossover offspring immediately undergoes a steepest-descent TSP solver. HGS achieves state-of-the-art results on CVRP and VRPTW benchmarks through this combination of diversity preservation and aggressive intensification.

**HGS historical milestones**:

| Year | Author | Achievement |
|------|--------|-------------|
| 2012 | Vidal et al. | MA for multi-depot, periodic, split-delivery VRP |
| 2014 | Vidal | Efficient local search in GVRP |
| 2022 | Vidal | HGS-CVRP: new CVRP benchmark records |
| 2023 | Kool et al. | HGS as baseline for NCO comparison |

### 5.6 Swarm Intelligence

#### Ant Colony Optimization (ACO) (Dorigo & Gambardella, 1997)

Inspired by pheromone-based foraging. The transition probability for ant $k$ at node $i$ selecting node $j$:

$$P_{ij}^k = \frac{[\tau_{ij}]^\alpha [\eta_{ij}]^\beta}{\sum_{l \in \mathcal{N}_i^k} [\tau_{il}]^\alpha [\eta_{il}]^\beta}$$

where $\tau_{ij}$ is the pheromone trail (updated by successful ants) and $\eta_{ij} = 1/d_{ij}$ is the heuristic desirability. **K-Sparse ACO** restricts evaluation to the $k$ nearest neighbors, reducing per-iteration complexity from $\mathcal{O}(V^2)$ to $\mathcal{O}(VK)$.

#### Particle Swarm Optimization (Kennedy & Eberhart, 1995)

$$v_i^{t+1} = w v_i^t + c_1 r_1 (P_{\text{best},i} - x_i^t) + c_2 r_2 (G_{\text{best}} - x_i^t), \quad x_i^{t+1} = x_i^t + v_i^{t+1}$$

For discrete routing, positions are decoded via random-key permutation encoding. **PSOMA** hybridizes PSO with local search (2-opt, swap) after velocity updates.

#### Other Swarm Methods

| Algorithm | Inspiration | Key Mechanism |
|-----------|-------------|---------------|
| **ABC** (Karaboga, 2005) | Honey bee foraging | Employed/onlooker/scout bees |
| **Firefly** (Yang, 2008) | Bioluminescent attraction | Brightness-based attraction |
| **Harmony Search** (Geem, 2001) | Musical improvisation | Harmony memory, HM consideration rate |
| **VPL / HVPL / AHVPL** | Volleyball leagues | League competition + local search |
| **SLC** (2017) | Football leagues | Team competition, formation changes |

### 5.7 Matheuristics

Matheuristics hybridize the speed of heuristic search with the mathematical bounding power of exact MILP, inheriting the best of both paradigms.

| Matheuristic | Heuristic Component | Exact Component | Key Property |
|-------------|--------------------|-----------------|-----------  |
| **ILS-RVND-SP** | ILS + Random VND | Set Partitioning ILP | Best known on many CVRP instances |
| **POPMUSIC** | KD-tree spatial index | Local exact solver | Scales to $10^4+$ nodes |
| **Kernel Search** | LP variable ranking | Restricted MILP | Iterative variable expansion |
| **Adaptive KS** | Adaptive fixing | MILP with variable locks | Runtime-adaptive |
| **Local Branching** | Hamming neighborhood | MILP sub-solver | Controllable search radius |
| **RINS / RENS** | LP-guided fixing | Exact sub-solver | High-quality UB rapidly |
| **Fix-and-Optimize** | Route locking | Exact sub-solver | Strongest exact finishing |
| **MIP-LNS** | LNS ruin | Exact repair | Mathematically optimal recreation |

**ILS-RVND-SP** (Subramanian et al., 2013): The **Set Partitioning** phase selects the optimal route combination from the pool generated by ILS:

$$\max \sum_{r \in \Omega} P_r x_r \quad \text{s.t.} \quad \sum_{r \in \Omega} a_{ir} x_r \le 1 \;\forall i \in V,\; x_r \in \{0,1\}$$

**Local Branching** (Fischetti & Lodi, 2003): Restricts the MIP search via Hamming-distance constraint from incumbent $\bar{x}$:

$$\sum_{j \in B_0} x_j + \sum_{j \in B_1}(1 - x_j) \le k$$

### 5.8 Hyper-Heuristics

Hyper-heuristics operate at a higher abstraction level, searching the *space of heuristics* rather than the solution space directly.

#### GP Hyper-Heuristic (GP-HH)

A generative hyper-heuristic that evolves a mathematical scoring function for constructive routing as a GP expression tree. A $k$-NN candidate list reduces complexity from $\mathcal{O}(N^2 R)$ to $\mathcal{O}(NKR)$.

#### Hidden Markov Model + Great Deluge HH (HMM-GD-HH)

Models the hyper-heuristic as an Input-Output Hidden Markov Model. Action selection balances expected improvement against transition entropy:

$$u^* = \arg\max_u \!\left[\bar{P}(u) - \alpha_t \sum_{s'} P(s'|u)\ln(P(s'|u) + \epsilon)\right]$$

Belief updates via stochastic online EM (Forward algorithm approximation).

#### Reinforcement Learning Great Deluge HH (RL-GD-HH)

Each heuristic maintains a scalar utility $u_i$. On strict improvement: $u_i \leftarrow \min(U_{\max}, u_i + r)$. Three punishment variants: subtractive (RL1), divisional (RL2: $u \leftarrow \lfloor u/2 \rfloor$), root (RL3: $u \leftarrow \lfloor\sqrt{u}\rfloor$). Gated by a linearly updating Great Deluge water level.

#### HH-ACO

Applies ACO pheromone logic to operator sequences. Ants construct operator sequences; the pheromone matrix $\tau_{ij}$ encodes the historically validated probability of improvement from executing Operator $j$ after Operator $i$.

#### HULK (Müller & Bonilha, 2022)

Governs a tripartite operator pool (Unstringing, Stringing, Local Search) via $\epsilon$-greedy adaptive operator selector with dual-decay learning:

$$\text{weight} \leftarrow (1 - \text{lr}) \cdot \text{weight} \cdot \text{decay} + \text{lr} \cdot \text{avg\_score}$$

---

## 6. Mathematical Search Operators

### 6.1 Crossover Operators

| Operator | Key Property | Best Suited For |
|----------|-------------|-----------------|
| **OX** | Preserves relative order | TSP, CVRP (sequence matters) |
| **PMX** | Preserves absolute positions | Time-window VRP |
| **CX** | Absolute positional structure | Position-critical problems |
| **SCX** | Greedy cost-guided | Cost-minimizing VRP |
| **ERX** | Maximizes edge inheritance | TSP dense instances |
| **CA-ERX** | ERX + capacity constraint | CVRP |
| **SREX** | Route-level macroscopic swap | Multi-route CVRP |
| **GPX** | Guaranteed $f(O) \le \min(f(P_1), f(P_2))$ | TSP (mathematical guarantee) |
| **Pattern & Itinerary** | Temporal/spatial decoupling | PVRP, MPVRPP |

**Generalized Partition Crossover (GPX)**: Identifies components in $G_\Delta = E(P_1) \oplus E(P_2)$ (symmetric difference), exhaustively evaluates each partition, and swaps to minimize offspring cost. Unique guarantee: the offspring is never worse than either parent.

### 6.2 Destroy Operators (LNS Ruin)

| Operator | Removal Logic | Multi-Period Variant |
|----------|--------------|---------------------|
| **Random** | Uniform PMF | Random Horizon |
| **Worst** | Max $\Delta C_i = d_{prev,i} + d_{i,next} - d_{prev,next}$ | Worst Profit Horizon |
| **Shaw** | Relatedness $R(i,j) = \phi\frac{d_{ij}}{\max d} + \chi\frac{|t_i-t_j|}{\max\Delta t}$ | Shaw Horizon |
| **Neighbor** | $k$ spatial nearest neighbors of seed | — |
| **Cluster** | K-means cluster ejection | — |
| **Route** | All nodes of a route | — |
| **String** | Contiguous route sequence length $L$ | Shift Visit |
| **Sector** | Angular wedge from depot | — |
| **Zone** | Euclidean annulus | — |
| **Pattern** | Full visit pattern $A_i = [a_{i1},\dots,a_{iT}]$ | Pattern Removal |

**Shaw Relatedness**:
$$R(i,j) = \phi \cdot \frac{d_{ij}}{\max d} + \chi \cdot \frac{|t_i - t_j|}{\max \Delta t} + \psi \cdot \frac{|q_i - q_j|}{\max \Delta q}$$

### 6.3 Recreate Operators (LNS Repair)

| Operator | Logic | Property |
|----------|-------|----------|
| **Greedy** | $\arg\min \Delta C$ | Fast, myopic |
| **Regret-$k$** | $\text{Regret}_i = \sum_{j=2}^k(\Delta C_{i,j} - \Delta C_{i,1})$ | Avoids spatial myopia |
| **Deep** | Penalizes fragmented vehicle space | Minimizes fleet size |
| **Noise** | $\Delta C_{ij} + \epsilon_{ij},\; \epsilon \sim U(-\delta, \delta)$ | Diversifies repair |
| **GENI** | Non-adjacent reconnections via Type I-IV Stringing | Sub-neighborhoods invisible to 2-opt |
| **Greedy Blink** | Skips best position with prob $p$ | Anti-cycling |
| **B&B Insertion** | Exact sub-solver for reconstruction | Mathematical optimality |

**Regret Insertion**:
$$\text{Regret}_i = \sum_{j=2}^k \bigl(\Delta C_{i,j} - \Delta C_{i,1}\bigr)$$

Prioritizes nodes that would suffer the greatest cost increase if deferred, preventing the greedy order from systematically under-serving spatially isolated nodes.

### 6.4 Improvement Operators

#### Intra-Route

| Operator | Neighborhood Size | Local Minimum Quality |
|----------|------------------|-----------------------|
| **2-Opt** | $\mathcal{O}(n^2)$ per route | 2-opt local min |
| **3-Opt** | $\mathcal{O}(n^3)$ | 3-opt local min |
| **4-Opt / Double Bridge** | $\mathcal{O}(n^4)$ | Used as perturbation |
| **LK** | Variable depth | Near-optimal in practice |
| **LKH** | $\alpha$-nearness bounded | Near-optimal, scales to $10^4$ |
| **Or-Opt** | $\mathcal{O}(n^2)$ for $L \in \{1,2,3\}$ | Strictly stronger than 2-opt for $L \ge 2$ |

#### Inter-Route

| Operator | Complexity | Description |
|----------|-----------|-------------|
| **2-opt*** | $\mathcal{O}(V^2)$ | Exchange route tails |
| **Node Relocation** | $\mathcal{O}(V^2)$ | Move single node between routes |
| **Or-opt*** | $\mathcal{O}(V^2)$ per chain length | Move chain of $L$ nodes inter-route |
| **Cross Exchange** | $\mathcal{O}(R^2 \cdot L^2)$ | Swap segments of length $\le L$ |
| **Swap-Star** | $\mathcal{O}(V \cdot |R|)$ | Optimal position-aware swap |
| **$\lambda$-Interchange** | $\mathcal{O}(R_A^\lambda \cdot R_B^\lambda)$ | All subsets up to cardinality $\lambda$ |
| **Cyclic Transfer** | $\mathcal{O}(p \cdot V)$ | Simultaneous $p$-route cascade |
| **Ejection Chain** | Depth-bounded recursion | Fleet-size minimization |

**2-opt* (Inter-Route)**:
$$\Delta C = d_{u,v} + d_{u',v'} - d_{u,u'} - d_{v,v'}$$

Unlike intra-route 2-opt, this does not reverse any sub-tour — both new segments inherit the orientation of their respective parent routes.

---

## 7. Acceptance Criteria in Trajectory Search

| Category | Criterion | Accept Condition | Key Parameter |
|----------|-----------|-----------------|---------------|
| **Greedy** | Only Improving (OI) | $f(s') < f(s)$ | None |
| **Greedy** | Improving and Equal (IE) | $f(s') \le f(s)$ | None |
| **Greedy** | Tabu Criterion (TC) | $\text{attr}(s \to s') \notin \mathcal{T}$ | Tenure $|\mathcal{T}|$ |
| **Thermodynamic** | Boltzmann-Metropolis (BMC) | Probabilistic, $\exp(-\Delta f/T)$ | Temperature $T$, cooling $\alpha$ |
| **Thermodynamic** | Cauchy SA | $1/(1+(\Delta f/T)^2)$ | Temperature $T$ |
| **Thermodynamic** | Tsallis SA | $[1-(1-q)\Delta f/T]^{1/(1-q)}$ | $q$, temperature $T$ |
| **Deterministic** | Threshold Accepting (TA) | $\Delta f \le \tau$, decreasing $\tau$ | Schedule $\tau(t)$ |
| **Deterministic** | Record-to-Record Travel (RRT) | $f(s') \le f(s^*) + \delta$ | Deviation $\delta$ |
| **Deterministic** | Great Deluge (GD) | $f(s') \le W$, decreasing $W$ | Decay rate $\Delta W$ |
| **Deterministic** | Demon Algorithm (DA) | $\Delta f \le D$; $D \leftarrow D - \Delta f$ | Initial demon energy |
| **Memory-based** | Late Acceptance (LAHC) | $f(s') \le f(s_{i-L})$ | History length $L$ |
| **Memory-based** | Step Counting (SCHC) | $f(s') \le B$, refresh every $L$ steps | Window $L$ |
| **Memory-based** | Old Bachelor (OBA) | Adaptive threshold $\tau_k$ | $\delta_{\text{tight}}, \delta_{\text{relax}}$ |
| **Structural** | Skewed VNS (SVNS) | $f(s') - \alpha \cdot \rho(s, s') < f(s)$ | $\alpha$, distance metric $\rho$ |
| **Multi-Objective** | Pareto Dominance (PD) | Strict Pareto improvement | — |
| **Multi-Objective** | $\epsilon$-Dominance | $(1-\epsilon)f_i(s') \le f_i(s)$ | $\epsilon > 0$ |
| **Ensemble** | EMA (G-AND / G-OR / G-VOT) | Logical consensus of criteria set | Ensemble composition |

**Demon Algorithm** is uniquely self-regulating: the demon energy $D$ organically emerges from the search trajectory rather than following an external schedule, making it strict after a run of improvements and permissive after a streak of deterioration.

**Late Acceptance Hill-Climbing (LAHC)**:
$$f(s') \le f(s_{i-L})$$
When the search is in a flat region, $f(s_{i-L}) \approx f(s)$ and the criterion behaves like IE; during rapid descent, the historical cost provides a looser threshold for temporary uphill steps. The array length $L$ is the sole tuning parameter.

---

## 8. Mandatory Node Selection for Multi-Period Problems

In multi-period profit-maximizing routing (MPVRPP, IRP, waste collection VRP), each operational day requires deciding *which* nodes must be visited to prevent overflow, combined with an economic prioritization of the most profitable subset. The taxonomy below organizes strategies along two axes: the information they use (temporal, spatial, statistical, economic) and the computational budget they require.

### 8.1 Overflow Prevention Strategies

| Strategy | Key Formula | Requires | Property |
|----------|------------|----------|---------|
| **Regular (Periodic)** | $t \equiv 1 \pmod{X}$ | None | Deterministic, wasteful |
| **Last-Minute** | $c_i / Q_{\max} \ge \tau$ | Fill level | Reactive, no lookahead |
| **Deadline-Driven** | $d^* = \lfloor(Q_{\max}-c)/\mu\rfloor \le H$ | $\mu$ (mean rate) | Parameter-free |
| **Look-Ahead** | $c_j + \mu_j d_{\text{next}} \ge Q_{\max}$ | $\mu$, $H$-step lookahead | Clusters co-incident collections |
| **Linear SLA** | $c + D\mu + Dk\sigma \ge Q_{\max}$ | $\mu, \sigma$, confidence $k$ | Risk-controlled |
| **Stochastic Regret** | $\mathbb{E}[\max(0, X-s_i)] > \gamma$ | Distribution $\mathcal{N}(\mu,\sigma^2)$ | Quantifies economic loss |
| **Multi-Day Overflow Prob.** | $\Pr[\sum_{k=1}^K X_k > Q_{\max}-c] > \alpha$ | Distribution | $K$-day tail risk |
| **CVaR** | $\text{CVaR}_\alpha(\max(0,F-Q_{\max})) > \epsilon$ | Distribution | Tail-risk-averse |
| **Wasserstein DRO** | Nominal expectation $+ \epsilon$ | Empirical distribution | Distribution-free robustness |
| **MIP Multiple-Knapsack** | $\min_x \sum_i(1-x_i)[\sum_s \pi_s o_i^{(s)} \hat{m}_i + P_i\Pr[\text{overflow}]]$ | Scenario tree | Exact overflow minimization |

**Stochastic Expected Overflow Regret**: Let $s_i = Q_{\max} - c_i$ be remaining capacity. With accumulation $X \sim \mathcal{N}(\mu, \sigma^2)$ and standard score $Z = (s_i - \mu)/\sigma$:

$$\mathbb{E}[\max(0, X - s_i)] = \sigma\phi(Z) + (\mu - s_i)(1 - \Phi(Z))$$

This closed-form expression directly quantifies the expected overflow loss from deferral.

### 8.2 Profit Maximization Strategies

| Strategy | Score | Property |
|----------|-------|---------|
| **Revenue Threshold** | $(c_i/Q_{\max}) V\rho R_{\text{kg}} > \tau_{\text{rev}}$ | Pure value criterion |
| **Profit-per-km (ROI)** | $r_i / (2d_{0,i}) > \epsilon$ | Routing-aware, penalizes isolated nodes |
| **Fractional Knapsack** | Sort by $r_i / m_i$, greedy pack | $\frac{1}{2}$-approximation guarantee |
| **Lagrangian Reduced Cost** | $\bar{c}_i = (r_i - \text{cost}\cdot d_i) - \lambda^* m_i > 0$ | LP shadow price alignment |

### 8.3 Spatial Synergy Strategies

| Strategy | Key Formula | Property |
|----------|------------|---------|
| **Spatial Synergy** | $S_{\text{syn}} = \{j : c_j/Q \ge \tau_{\text{syn}} \wedge d_{ij} \le R\}$ | Co-locates collections |
| **Clarke-Wright Savings** | $s_{ij} = d_{0i}+d_{0j}-d_{ij} > 0$ | Cluster formations |
| **Route-Cluster Synergy** | $\Delta C_i / r_i \le \epsilon_{\text{route}}$ | Adapts to current fleet position |
| **Set-Cover Hub** | $\min |S|$ s.t. $\forall i \in U, \exists j \in S: d_{ij} \le R$ | $\ln(|U|)$ approximation |
| **Submodular Facility** | $f(S) = \sum_i \max(0, r_i - \alpha\min_{j\in S}d_{ij})$ | $(1-1/e)$ guarantee |
| **Supermodular Synergy** | $f(S) = \sum_i r_i - 2\alpha\sum_i\min_{j\in S\setminus\{i\}}d_{ij}$ | Increasing returns clustering |

### 8.4 Simulation and Decision-Theoretic Strategies

| Strategy | Method | Property |
|----------|--------|---------|
| **One-Step Rollout** | Compare "Collect Today" vs. "Defer" via base heuristic | Optimal stopping approximation |
| **Multi-Step Rollout** | $V(S) = \mathbb{E}[\sum_{t=1}^H \gamma^t C_t(S)]$ via Monte Carlo | Captures multi-day dependencies |
| **Whittle Index (RMAB)** | Exact "subsidy for passivity" via Value Iteration | Theoretically grounded RL |
| **VFA** | $\hat{V}_\theta(s_i)$ via TD learning | Fast approximation |
| **Pareto-Front** | Non-dominated on (Urgency, Routing Efficiency) | Multi-objective without weighting |

### 8.5 Adaptive Ensemble Dispatchers

| Dispatcher | Aggregation | Key Property |
|-----------|------------|-------------|
| **Combined (OR/AND)** | $\bigcup_k S_k$ or $\bigcap_k S_k$ | Same-domain strategy fusion |
| **Portfolio Dispatcher** | Logical OR/AND over heterogeneous sources | Cross-domain complementarity |
| **Contextual Thompson Sampling** | Beta-Bernoulli posteriors over strategies | Bayesian bandit adaptation |
| **Entropy-Regularized** | $\text{Score}_k = \bar{R}_k - \beta_t H(\pi_k)$ | Prevents strategy specialization |
| **Learned Imitation** | ML classifier imitates exact solver | Fast, interpretability limited |

---

## 9. Neural Combinatorial Optimization

### 9.1 Historical Arc

| Period | Paradigm | Representative Work |
|--------|----------|---------------------|
| 2015 | Pointer Networks | Vinyals et al. (2015) |
| 2018 | Transformer-based constructive | AM — Kool et al. (2018) |
| 2019 | GCN heatmap + search | Joshi et al. (2019) |
| 2019 | Neural improvement | L2I — Chen & Tian (2019) |
| 2020 | Multi-start symmetry | POMO — Kwon et al. (2020) |
| 2021 | Asymmetric problems | MatNet — Kwon et al. (2021) |
| 2021 | Dual-aspect improvement | DACT — Ma et al. (2021) |
| 2022 | Active search | EAS — Hottung et al. (2022) |
| 2022 | Beam search + simulation | SGBS — Choo et al. (2022) |
| 2023 | Diffusion-based heuristic | DIFUSCO — Sun & Yang (2023) |
| 2023 | Neural $k$-opt | NeuOpt — Luo et al. (2023) |
| 2023 | Deep ACO | DeepACO — Liu et al. (2023) |
| 2024 | Multi-task MoE | MVMoE — Jiao et al. (2024) |
| 2024 | Hierarchical hybrid | GLOP — Ye et al. (2024) |
| 2025 | Unified foundation model | URS, RouteFinder, SHIELD |
| 2025 | LLM-designed heuristics | EoH-S, VRPAGENT, EvoReal |

### 9.2 Autoregressive Constructive Models

#### Pointer Network (Vinyals et al., 2015)

The foundational architecture introducing attention as a trainable "pointer" over variable-length inputs. LSTM encoder produces hidden states $e_j$; at decoding step $i$, alignment scores are computed:

$$u_j^i = v^\top \tanh(W_1 e_j + W_2 d_i), \qquad p(C_i = j \mid C_{<i}, \mathcal{P}) = \operatorname{softmax}(u^i)_j$$

Originally trained with supervised learning on exact solutions; later REINFORCE replaced the oracle dependence.

#### Attention Model (AM) (Kool et al., 2018)

The canonical Transformer-based routing model. Encoder: stack of MHA + feedforward layers producing node embeddings. Decoder: context vector $h_{(c)}$ (graph embedding + last node + first node + remaining capacity for CVRP) attending over unvisited nodes via a single-head attention with $\tanh$ clipping.

**Training**: REINFORCE with rollout baseline (cost of greedy rollout from best previous-epoch model):

$$\nabla_\theta \mathcal{J}(\theta) = \mathbb{E}_{p_\theta(\pi|s)}\!\left[\bigl(L(\pi) - b(s)\bigr) \nabla_\theta \log p_\theta(\pi \mid s)\right]$$

#### POMO (Kwon et al., 2020)

Exploits the $N$-fold rotational symmetry of routing problems: $N$ rollouts launched in parallel from each of the $N$ starting nodes. The multi-start reward mean serves as a self-supervised, instance-specific baseline:

$$\nabla_\theta \mathcal{J}(\theta) = \frac{1}{N} \sum_{n=1}^N \bigl(L(\pi_n) - \bar{L}\bigr) \nabla_\theta \log p_\theta(\pi_n), \quad \bar{L} = \frac{1}{N}\sum_{n=1}^N L(\pi_n)$$

Achieves near-LKH3 quality on CVRP-100 at inference time.

#### MatNet (Kwon et al., 2021)

Designed for asymmetric problems. Encodes the full cost matrix $C \in \mathbb{R}^{N \times N}$ via bilinear cross-attention reading directed edge cost $c_{ij}$ directly into the attention logit:

$$a_{ij} = \frac{\exp\!\left(q_i^\top k_j + w \cdot c_{ij}\right)}{\sum_l \exp\!\left(q_i^\top k_l + w \cdot c_{il}\right)}$$

The preferred architecture for heterogeneous fleet variants and ATSP where $c_{ij} \neq c_{ji}$.

#### LEHD (Luo et al., 2023)

Inverts the encoder/decoder capacity budget: shallow encoder, deep autoregressive decoder that **re-reads** instance features at every decoding step conditioned on the current partial solution state. Replaces the fixed context vector with a dynamic, step-aware graph representation.

### 9.3 Non-Autoregressive Constructive Models

#### GCN Edge Heatmap + Beam Search (Joshi et al., 2019)

A GCN trained via supervised learning against exact solutions outputs edge inclusion probabilities:

$$P_{ij} = \operatorname{Sigmoid}\!\left(\operatorname{MLP}([h_i \parallel h_j \parallel e_{ij}])\right)$$

Classical beam search then assembles a feasible tour following highest-probability edges. The GCN runs **once** — inference scales far beyond autoregressive models.

#### DIFUSCO (Sun & Yang, 2023)

Applies **denoising diffusion probabilistic models** to CO. Diffuses the binary edge-solution matrix $x_0$ forward with Gaussian noise; a graph transformer learns the reverse process:

$$p_\theta(x_{t-1} \mid x_t, G) = \mathcal{N}(x_{t-1};\, \mu_\theta(x_t, t, G),\, \Sigma_\theta(x_t, t, G))$$

The iterative denoising naturally integrates global tour consistency, outperforming GCN heatmaps on TSP-500/1000 without fine-tuning.

#### BQ-NCO (Drakulic et al., 2023)

Recasts solution construction as Q-value estimation. A transformer encodes both the "left" and "right" partial tour simultaneously (bi-directional). A Q-value head scores each candidate next node based on expected solution quality from that state. Bi-directionality anchors search from both ends, reducing compounding error.

### 9.4 Improvement and Local Search Models

| Model | Year | Mechanism | Key Innovation |
|-------|------|-----------|---------------|
| **L2I** | 2019 | RL on operator selection | First end-to-end neural improvement for VRP |
| **N2S** | 2022 | Continuous scoring over swaps | Adaptive effective neighborhood radius |
| **DACT** | 2021 | Dual-aspect transformer | Decoupled spatial vs. sequential features |
| **NeuOpt** | 2023 | Autoregressive $k$-opt | Learned Lin-Kernighan, finds long-range improvements |

**NeuOpt** factorizes the joint probability over the $k$-opt exchange tuple:

$$p(E_\text{swap} \mid G, S_t) = \prod_{i=1}^k p(e_i \mid G, S_t, e_1, \dots, e_{i-1})$$

Effectively a **learned Lin-Kernighan** heuristic capable of discovering long-range improving chains that 2-opt or 3-opt miss.

### 9.5 Meta-Heuristic Augmented Models

#### DeepACO (Liu et al., 2023)

Replaces the static desirability $\eta_{ij} = 1/d_{ij}$ with a learned GNN, enabling instance-adaptive, constraint-aware guidance:

$$p_{ij}^k = \frac{[\tau_{ij}]^\alpha \cdot [\operatorname{MLP}_\theta(h_i, h_j, e_{ij})]^\beta}{\sum_{l \in \mathcal{N}_i^k} [\tau_{il}]^\alpha \cdot [\operatorname{MLP}_\theta(h_i, h_l, e_{il})]^\beta}$$

#### GFlowNet Ant Colony System

Trains the pheromone policy via the **Trajectory Balance loss**:

$$\mathcal{L}(\theta) = \left(\log \frac{Z_\theta \cdot P_F(\tau)}{R(x) \cdot P_B(\tau)}\right)^2$$

The result: a colony that samples solutions **proportionally to their reward** rather than collapsing to a single local optimum — overcoming the mode-collapse failure of standard RL-trained policies.

#### DR-ALNS

Replaces ALNS's roulette-wheel operator selection with a DQN or PPO agent that learns instance-dependent dynamics. State $s_t$ encodes optimality gap, iteration count, recent operator performance, and fleet utilization. The neural network learns that "relatedness removal + greedy repair" works well early; "worst removal + SISR repair" near convergence.

#### GLOP (Ye et al., 2024)

Hierarchical hybrid: a high-level RL policy partitions the instance into subproblems (spatial clustering), assigns each to a neural constructive model, then hands partial state to LKH3 for local refinement. The partition policy is trained end-to-end on the combined objective, learning optimally-sized subproblems for the downstream solver.

### 9.6 Predict-Once Solvers

Predict-once solvers decouple the neural model from search: the DNN runs a single forward pass and outputs guidance (edge heatmaps, candidate lists, branching priors) consumed by an independent OR solver.

| Method | Neural Output | OR Component | Inference |
|--------|--------------|--------------|---------|
| **GCN heatmap** | Edge probabilities | Beam search | $\mathcal{O}(N)$ GCN once |
| **DIFUSCO** | Denoised edge matrix | Greedy / MCTS | $T$ denoising steps once |
| **Neural Candidate for LKH** | Sparse $k$-candidates per node | LKH3 unchanged | GNN once |
| **ML-guided B&B** | Branching variable ranking | B&B tree | GNN at root |

**Neural Candidate Generation**: Replaces LKH3's $k$-nearest-neighbor candidates with a learned $k$-candidate list per node, improving solution quality on CVRP and VRPTW by 1–3% at equal iteration budgets.

### 9.7 Predict-Many Solvers

Predict-many solvers call the neural model **repeatedly** during search, consuming current search state and outputting updated guidance.

| Method | Neural Calls | Adaptation |
|--------|-------------|-----------|
| **DPDP** | At every DP stage (beam pruning) | Sees progressively more complete solutions |
| **Neural MCTS** | At every tree expansion + rollout | Concentrates on most promising subtrees |
| **Iterative Graph Sparsification** | $R$ rounds of pruning + re-prediction | Corrects early over-pruning errors |

**SGBS (Simulation-Guided Beam Search)** (Choo et al., 2022): At each beam expansion, candidates are evaluated not by immediate log-probability but by **expected total cost under $T$ stochastic rollouts**. Converts any constructive model into a vastly more powerful inference-time planner without retraining.

### 9.8 Active Search and Fine-Tuning

| Method | What is Updated | Speed vs. Full Active Search |
|--------|----------------|-------------------------------|
| **Active Search** (Bello et al., 2016) | Full model parameters per instance | Very slow |
| **EAS** (Hottung et al., 2022) | Adapter layer only (frozen backbone) | 20-50× faster |
| **COMPASS** (Luo et al., 2023) | Parallel policies with async improvement | Model-agnostic wrapper |

**EAS Mechanism**: A lightweight adapter layer is injected into the pretrained AM or POMO model. At test time, only adapter weights are optimized via REINFORCE on the specific instance.

### 9.9 Multi-Task and Generalist Models

| Model | Year | Variants | Mechanism |
|-------|------|---------|-----------|
| **MVMoE** | 2024 | 16+ VRP variants | Sparse Mixture-of-Experts gating |
| **URS** | 2025 | 100+ VRP variants | Unified Data Representation + Mixed Bias Module |
| **RouteFinder** | 2025 | Universal VRP | Global Attribute Embeddings, Mixed Batch Training |
| **SHIELD** | 2025 | Compositional variants | State-Decomposable MDPs, basis policy experts |

**MVMoE Gating**: A gating network routes each input token to one of $E$ expert FFN modules. Variant identity is inferred from instance features — not provided explicitly. Achieves competitive performance with single-task models across 16 variants using equal parameter budget.

**SHIELD State Decomposition**: Models complex VRPs as compositions of "basis" problems. The state space decomposes as a Cartesian product of basis states; basis policies (Capacity Expert, Time Window Expert, etc.) are mixed via a hierarchical gating mechanism — only relevant experts activate per instance.

### 9.10 IRP and Multi-Period NCO Models

| Architecture | Approach | Key Insight |
|-------------|---------|------------|
| **L2D** | Decoupled dispatch + routing | Separates inventory decision from geometry |
| **BGN** | Bipartite Graph Network | Explicit supply-demand separation |
| **Hierarchical Temporal Attention** | Period-level + within-period attention | Neural counterpart to Benders decomposition |

**Hierarchical Temporal Attention**: Period-level transformer captures inter-day inventory dynamics (replenishment wave propagation); within-period transformer handles spatial routing. The two levels are trained jointly with a shared reward signal, so the schedule level learns to create daily workloads that the routing level executes efficiently.

### 9.11 LLM-Driven Heuristic Design (2025–2026)

The most disruptive 2025–2026 development: LLMs used as **meta-solvers** that design algorithms rather than just solve instances.

#### Evolution of Heuristic Sets (EoH-S)

Uses an evolutionary framework where the population consists of LLM-generated Python code snippets. "Complementary population management" maintains heuristics that perform well on instances where the current elite set fails, automating algorithm portfolio design.

#### VRPAGENT (2025)

LLMs write "Destroy" and "Repair" operators for LNS. A genetic algorithm refines these operators based on solution quality contribution. Discovered novel operators outperforming handcrafted baselines on CVRP and VRPP, running on a single CPU core.

#### EvoReal (2026)

Addresses the "Synthetic Gap" (models fail on real-world data). An LLM-guided evolutionary module synthesizes VRP instances matching real-world statistical fingerprints (clustering, depot centrality). A neural solver trains on a curriculum of evolved instances via progressive adaptation from random to realistic topologies.

---

## 10. Future Research Directions

### 10.1 Generalization and Distribution Shift

The most critical open problem in NCO: models trained on uniform random instances fail catastrophically on real-world clustered or structured data. Key research directions:

- **Foundation model for routing**: A single model capable of zero-shot generalization to any VRP variant, analogous to GPT for text. URS and RouteFinder are early steps; bridging to instances of thousands of nodes remains open.
- **Cross-distribution training curricula**: Progressive learning schemes (as in EvoReal) that gradually shift training distributions from synthetic to realistic, using domain randomization or data-augmentation by LLM-generated instance families.
- **Invariant feature learning**: Architectures that explicitly learn distribution-invariant representations — e.g., via causal discovery or domain-adversarial training — separating structural routing patterns from distributional artifacts.

### 10.2 Scalability to Large Instances

Classical methods (LKH3, HGS) still significantly outperform NCO at $N > 500$ nodes. Key directions:

- **Hierarchical sparse attention (Scale-Net)**: U-Net-style coarsening/refinement to break the $\mathcal{O}(N^2)$ attention bottleneck, enabling end-to-end neural models for $N > 1000$.
- **Predict-once + classical local search**: GLOP-style neural partitioning followed by LKH3 local refinement, learning partition policies that create subproblems optimally sized for the downstream solver.
- **Progressive graph sparsification**: Iterative rounds of GNN prediction and edge pruning converging to $\mathcal{O}(N)$ candidate edges, scaling to TSP-10000 and VRP-5000.

### 10.3 Uncertainty and Stochastic Environments

Most NCO work addresses static, deterministic instances. Critical gaps:

- **Neural stochastic VRP**: End-to-end RL for SVRP and SIRP where demand is revealed online. Current work relies on classical stochastic programming; neural approaches must handle partial observability and recourse actions.
- **Distributionally robust NCO**: Training neural solvers that explicitly optimize worst-case Wasserstein-ball expected cost, bridging DRO theory and end-to-end learning.
- **Online and dynamic VRP**: Real-time re-routing as customer requests arrive; NCO models that update solutions incrementally with sub-second latency as the fleet moves.

### 10.4 Multi-Objective and Pareto-Aware Optimization

Real-world logistics balances competing objectives (cost, emissions, service quality, equity):

- **Pareto-conditioned NCO**: Models that take a scalarization weight vector as input and produce Pareto-diverse solutions — training once, serving all tradeoff preferences at inference.
- **Multi-objective GDPO / PMOCO extensions**: Applying gradient-decomposition and PCGrad-style conflict resolution within transformer decoders to prevent gradient interference between objectives.
- **Bi-criteria CVRP and green VRP**: Jointly minimizing travel cost and CO₂ emissions via neural architectures with explicit energy-depletion state tracking (EVRP).

### 10.5 Hierarchical and Multi-Period Routing

The MPVRPP and IRP represent an underexplored frontier for NCO:

- **Temporal-spatial joint optimization**: Architectures that simultaneously learn *when* to visit (temporal schedule) and *how* to route (spatial sequence) in a unified latent space, without the decoupling that currently limits L2D and BGN.
- **Meta-RL for periodic routing**: Manager-worker hierarchical RL where the manager sets multi-day collection schedules and the worker executes daily routing — with both levels trained jointly via shared reward signals.
- **Neural inventory management**: Integrating fill-level dynamics directly into the attention encoder so that the model naturally learns to balance urgency, spatial efficiency, and profit maximization across the planning horizon.

### 10.6 Hybrid Neural-Classical Synergies

The most immediate practical impact may come from tighter neural-classical integration:

- **ML-guided BPC**: Using GNNs to predict strong branching scores, variable selection priorities, and cutting plane separation in BPC, reducing B&B tree sizes by orders of magnitude on instances drawn from the training distribution.
- **Neural operator design for ALNS**: LLM-generated destroy/repair operators (VRPAGENT) combined with meta-bandit selection, automatically discovering domain-specific LNS operators that outperform hand-crafted counterparts.
- **Neuro-exact IRP**: Combining approximate DP via trained value function approximators (VFA) with exact Benders refinement for SIRP, trading optimality guarantees for dramatic speedups on large-horizon instances.

### 10.7 Foundation Models and LLM Integration

The 2025–2026 emergence of LLM-driven algorithm design opens profound research directions:

- **Algorithm evolution with MCTS**: Replacing genetic algorithm search over code (EoH) with MCTS + "Tree-Path Reasoning," enabling backtracking and deeper exploration of radical design changes.
- **Universal heuristic portfolios**: LLM-evolved diverse heuristic sets (EoH-S) that cover the problem space complementarily, combined with contextual meta-selectors (Thompson Sampling dispatchers) that route instances to the most appropriate heuristic at runtime.
- **Instance-adaptive fine-tuning at scale**: Scaling EAS-style test-time adaptation to large transformer routing models, where the adapter budget is proportional to instance difficulty rather than fixed.

### 10.8 Theoretical Guarantees and Explainability

A major gap in NCO is the absence of approximation guarantees:

- **Learned approximation algorithms**: Training NCO models to provably approximate NP-hard problems within known ratios (e.g., $(1+\epsilon)$-optimal with high probability), bridging algorithm theory and machine learning.
- **Formal verification of neural solvers**: Certifying that neural routing policies satisfy capacity constraints, time windows, and other hard feasibility conditions for all inputs — critical for safety-critical applications.
- **Explainable attention routing**: Connecting Transformer attention weights to classical graph-theoretic quantities (spanning trees, $\alpha$-nearness candidates), providing interpretable explanations for why neural models select specific nodes.

### 10.9 Real-World Integration

- **CO + LLM hybrid systems**: Using LLMs as high-level planners for semantic constraints ("driver must visit pharmacy before hospital") while delegating geometric routing to specialized NCO models.
- **Digital twin optimization**: Real-time routing optimization synchronized with digital twin city models, incorporating live traffic, bin fill sensors, and dynamic demand signals.
- **Federated learning for routing**: Training NCO models on private operational data (from logistics operators) without sharing sensitive route or customer information, using federated learning across competing companies to build stronger shared models.
- **Carbon-aware routing**: Integrating real-time carbon intensity data from electricity grids into the EVRP state representation, enabling dynamic routing decisions that minimize lifecycle emissions rather than just distance.

---

## Summary: Key Equations Reference

| Equation | Context |
|----------|---------|
| $\min \sum c_{ij}x_{ij}$ s.t. DFJ cuts | Symmetric TSP ILP |
| $\max \sum p_i y_i$ s.t. $\sum c_{ij}x_{ij} \le T_{\max}$ | VRPP / Orienteering |
| $I_{it} = I_{i,t-1} + q_{it} - d_{it}$ | IRP inventory flow |
| $P(s \to s') = \exp(-\Delta f/T)$ | Simulated Annealing |
| $w_{j,t+1} = (1-\rho)w_{j,t} + \rho c$ | ALNS bandit update |
| $\text{Regret}_i = \sum_{j=2}^k(\Delta C_{i,j} - \Delta C_{i,1})$ | Regret insertion |
| $BF(p) = \mathit{fit}(p) + (1-nc_{pop}/N)\Delta(p)$ | HGS biased fitness |
| $\nabla_\theta\mathcal{J} = \mathbb{E}[(L(\pi)-b(s))\nabla_\theta\log p_\theta]$ | REINFORCE routing |
| $\frac{1}{N}\sum_n(L(\pi_n)-\bar{L})\nabla_\theta\log p_\theta(\pi_n)$ | POMO multi-start |
| $\mathbb{E}[\max(0,X-s_i)] = \sigma\phi(Z) + (\mu-s_i)(1-\Phi(Z))$ | Stochastic overflow regret |
| $f(s') - \alpha\cdot\rho(s,s') < f(s)$ | Skewed VNS acceptance |
| $u^* = \arg\max_u[\bar{P}(u) - \alpha_t H(P(\cdot|u))]$ | HMM-GD hyper-heuristic |

---

*This report synthesizes content from the WSmart+ Route project's `docs/moon/markdown/` documentation suite and `docs/moon/reports/` research archive. For implementation details of the algorithms described here, consult the corresponding modules in `logic/src/policies/` and `logic/src/models/`.*
