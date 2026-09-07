# Comprehensive Review of VRPP & CTOP Heuristic Operators

**Project**: WSmart+ Route
**Date**: August 27, 2026
**Purpose**: Theoretical evaluation of profit-aware (economic exploitation) versus purely structural (geometric exploration) operator mechanics across VRPP and CTOP (Capacitated Team Orienteering Problem) solvers.

To satisfy reviewers at top-tier AI and Operations Research venues (e.g., NeurIPS, INFORMS, Transportation Science), routing algorithms must maintain a rigorous balance between **exploitation** (optimizing the profit objective within capacity and shift boundaries) and **exploration** (escaping local optima via unconstrained structural moves).

---

## 1. Repair (Insertion) Operators

_Goal: Reconstruct destroyed routes to maximize the net objective function._

### The "Essential" Tier (Highly Justified for VRPP & CTOP)

These operators perfectly map to the dual objective of maximizing collected rewards while minimizing routing costs.

- **Greedy & Greedy Blink:** Direct gradient steps on the objective function. Evaluating immediate marginal profit ($\Delta P = R \cdot w_i - C \cdot \Delta d$) is the most fundamental heuristic.
- **Regret-$k$:** Directly models **opportunity cost**. Prioritizes nodes that will suffer a massive profit loss if not inserted into their best current position.
- **Savings:** The Clarke-Wright savings metric evaluates the financial benefit of merging routes (synergy) versus serving a node on a dedicated route.

### The "Clever" Tier (Strongly Justified)

- **Deep Insertion:** VRPP/CTOP is heavily constrained by the knapsack problem (vehicle capacity $Q$) and shift budget ($T_{\max}$). Deep insertion explicitly balances spatial efficiency with knapsack and time efficiency by penalizing routes that exhaust capacity or shift hours too quickly.

### The "Questionable" Tier (Requires Redesign or Low Weight)

- **Farthest Insertion:** **Flawed for pure VRPP profit.** Farthest insertion is a TSP construction heuristic designed to trace the convex hull of a route. In VRPP, going to the _farthest_ possible node actively destroys profit unless balanced by massive fill.
  - _Guideline:_ Maintain farthest insertion as a low-probability "Spatial Diversification Operator" in ALNS pools specifically to escape dense clusters near the depot.

---

## 2. Destroy (Removal) Operators

_Goal: Dismantle sub-optimal parts of the solution. Must balance removing bad economic decisions with blind geometric shuffling._

### The "Essential" Tier (Must Make Profit-Aware)

These operators evaluate the "quality" of a node's assignment, which in VRPP is strictly economic.

- **Worst Removal:** Evaluates the lowest (or negative) marginal profit contribution of a node ($\Delta P_i = R \cdot w_i - C \cdot \Delta d_i$), rather than raw distance alone.
- **Shaw (Related) Removal:** Similarity metric incorporates an **economic equivalence** term alongside geographic distance and demand:
  $$R(i, j) = \phi_1 \cdot \frac{d_{ij}}{d_{\max}} + \phi_2 \cdot \frac{|w_i - w_j|}{w_{\max}} + \phi_3 \cdot \frac{|\Delta P_i - \Delta P_j|}{\Delta P_{\max}}$$

### The "Clever Systemic" Tier (Strongly Justified)

- **Route Removal:** Specifically calculates the **Net Profit Margin** of non-mandatory routes. Scraps entire routes that operate at a net loss or fail time efficiency.
- **Historical Knowledge Removal:** Tracks the historical _profitability_ of specific node pairs or route assignments, rather than just raw distance costs.

### The "Leave Them Alone" Tier (Do NOT Make Profit-Aware)

- **Neighborhood, Cluster, Sector, and String Removal:** These are purely **spatial and structural diversification** operators. Their job is to rip out geographic chunks of the map completely blind to cost/profit. If made profit-aware, they collapse into redundant, localized versions of "Worst Removal."
- **Random Removal:** Must remain purely uniform random.

---

## 3. Stringing / Unstringing Operators (Block Relocation)

_Goal: Move contiguous sequences of nodes to escape the "Synergy Trap" of single-node evaluation._

- **The Synergy Trap:** Single-node Worst Removal often ignores highly unprofitable distant clusters because removing just _one_ node from the cluster yields almost no distance savings.
- **The Solution:** Dedicate specific operators (e.g., US Unstringing Variant IV) to "Sub-tour Profit Amputation", allowing the algorithm to evaluate and amputate entire unprofitable branches at once.
- **The Constraint:** Keep Variants I, II, and III purely structural (blind) to preserve sequence-shuffling diversity in the operator pool.

---

## 4. Perturbation Operators

_Goal: Violently cross fitness valleys to escape deep local optima. Structural destruction must remain blind, but reconstruction must be profit-aware._

### The "Broken" Anti-Pattern (Resolved)

- **Evolutionary Perturbation (Micro-GA):** Evaluating TSP distance alone actively strips away highly profitable, slightly out-of-the-way nodes.
  - _Resolution:_ Genetic crossover operators (OX, Swap) remain structural, but the selection fitness function evaluates total **Route Profit** ($P_{\text{total}} = \text{Revenue} - \text{Cost}$).

### The "Hybrid" Tier (Theoretically Sound)

- **Kick (Ruin-and-Recreate):** Randomly sampling nodes to destroy (blind exploration) followed by greedy profit reinsertion (exploitation) is theoretically robust.
- **Genetic Transformation:** Locking historical elite edges while wiping and greedily reinserting the rest balances structural memory and economic greed.

### The "Do Not Touch" Tier (Leave Entirely Blind)

- **Double Bridge (4-opt):** Shatters tightly wound, greedy clusters by reconnecting segments non-sequentially ($A \to C \to B \to D$). Must remain blind to force the algorithm out of local maxima.
- **Random Perturb:** Multi-swaps must remain chaotic; the outer acceptance criterion (e.g., Simulated Annealing) decides acceptance.

---

## 5. CTOP Temporal Duality: Shift-Time vs. Distance

_Goal: Enforce working shift time budgets ($T_{\max}$) alongside vehicle capacity ($Q$)._

In the Capacitated Team Orienteering Problem (CTOP, integrated 2026-08-27):
1. **Marginal Time Efficiency:** The insertion metric evaluates marginal profit per hour:
   $$\eta_i = \frac{\Delta P_i}{\Delta t_i} = \frac{R \cdot w_i - C \cdot \Delta d_i}{\frac{\Delta d_i}{v_{\text{avg}}} + t_{\text{service}}}$$
2. **Dual Feasibility Enforcement:** Any candidate move must satisfy both $\sum w_i \le Q$ and $t(\text{route}) \le T_{\max}$.
3. **Vectorized Evaluation:** Vectorized operators in `logic/src/policies/vector/` evaluate batches of candidate insertions against both capacity and shift duration tensors in parallel on GPU.
