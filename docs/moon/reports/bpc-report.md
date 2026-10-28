# Branch-and-price-and-cut in WSmart-Route

Date: 2026-10-27. Read-only account of the solver on `main` and of the manuscript `assets/papers/BPC-for-the-Capacitated-PTP-with-Resource-Constraints-and-Time-Windows/paper.tex`. No production files were changed to produce this note.

Branch-and-price-and-cut, as this repository and its manuscript define it, is a branch-and-bound tree whose node bound is a column-generation linear program, tightened by cuts whose duals are pushed back into the pricing problem. The manuscript designs that machine for the capacitated profitable tour problem with resource constraints and time windows (CPTP-RCTW). The code that runs today is the same skeleton on a smaller problem: a capacitated profitable tour with no clock.

## Three problems that share a name

The manuscript `assets/papers/BPC-for-the-Capacitated-PTP-with-Resource-Constraints-and-Time-Windows/paper.tex` defines CPTP-RCTW in Section “Problem Statement.” A route is a path from a depot copy \(0\) through distinct customers to a copy \(n+1\). The solver selects at most \(K\) such routes. The objective is collected prize minus travel cost,

\[
\max \sum_{i\in N} p_i y_i - \sum_{(i,j)\in A} c_{ij} x_{ij}.
\]

Each customer has a prize \(p_i\), a demand \(q_i\), a service time \(s_i\), and a window \([a_i,b_i]\). Each arc has a cost and a travel time. A mandatory set \(M\) must be visited exactly once. Optional customers are visited at most once. Load on a route is at most \(Q\). Waiting before \(a_i\) is allowed, and service must start by \(b_i\). Extra cumulative resources \(\rho\in\mathcal{R}\) (the text names duration, length, and a stop limit) stay within \(U^\rho\). Time windows and resources define which routes exist. They do not appear as rows of the extended master. Pricing with zero duals is the one-vehicle problem, and the paper reduces that case to the TSP, so it is strongly NP-hard even with one vehicle, no extra resources, and every window equal to the depot window.

The learning environment `logic/src/envs/cptptwtc.py` is a different object, CPTPTWTC. It is one vehicle and many trips. Each trip has a capacity and a shift (driving plus service). The day clock does not reset at the depot. An early vehicle waits, service must start by `tw_late`, and the vehicle must be home before the depot window closes. The objective used in routing is kilograms times a revenue weight minus kilometres times a cost weight. The generator does not build a mandatory set. CPTPTWTC is registered as a learning problem and is not in the released set (`PTP`, `CPTP`, `CPTPTC`, `CVRP`). The BPC engine never reads its time-window or shift tensors.

The simulator paper’s daily model is smaller again: profit is \(R\) times collected waste minus \(C\) times distance, the only feasibility check is load at most \(Q\), and the fleet is not limiting. There is no window and no shift. The DHRL paper adds a route duration and a fleet bound \(m\), and it treats time windows as optional. Those three mathematical programs should not be collapsed into one.

Special cases the BPC manuscript draws are useful as a map. Drop windows and extra resources and the problem is the CPTP. Infinite capacity is a multi-vehicle profitable tour with time windows. Full coverage and zero prizes is the VRPTW. The team orienteering problem maximizes prize under a length limit. The prize-collecting TSP minimizes cost plus penalties subject to a minimum prize, and the manuscript marks it as related rather than a special case.

## Where branch-and-bound sits

Branch-and-bound searches a tree of restrictions of one integer program. Each node stores a partial decision and a bound on the best completion. A node is discarded when its bound cannot beat the incumbent, when the restriction is infeasible, or when the bound is itself an integer solution. The order of the nodes and the choice of the split decide whether the search reuses the parent linear program or chases the best bound.

In this code the live walk is `search_bpc_tree` in `logic/src/policies/helpers/solvers_and_matheuristics/search/bpc_tree.py`. Both `run_bpc` and `run_ms_bpc_sp` call it. A `BranchNode` stores only the constraints added at that node, the parent pointer, the LP value, the \(\lambda\) map, and a basis snapshot. Columns live in one shared master. Selecting a node resets every column upper bound to 1 and then sets the upper bound to 0 on routes that violate the path from the root. The tree is a maximization tree. The incumbent is a global lower bound. There is no stored global upper bound. The gap uses the best LP value among open nodes.

Default node selection is a stack, so the search is depth-first. The classes `BestFirstSearch` and `HybridSearchStrategy` exist and are not called. `"best_first"` is the one alternative that does run: a heap on the LP value. Depth-first search is there so a child can start from the parent’s simplex basis.

The split, in the order the loop actually uses, is:

1. Optional-customer visit, on by default. The most fractional visit \(y_v\) is forced to 1 in one child and to 0 in the other. Strong branching, off by default, scores a few candidates by the product of the two restricted LP drops and does not re-price.
2. Fleet size, if \(\sum\lambda\) is fractional: \(\sum\lambda\le\lfloor\sum\lambda\rfloor\) against \(\sum\lambda\ge\lceil\sum\lambda\rceil\).
3. Divergence. The two most fractional routes are walked from the depot. At the first vertex where they separate, outgoing arcs are split by cumulative flow into a high-flow set and its complement. Both children forbid one of those sets. The manuscript asks for a polar-angle split in the spirit of Barnhart, Hane, and Vance (2000). The code splits by flow and can add arcs whose direction is close to the heaviest successor.
4. Ryan–Foster (1981). A customer pair whose co-occurrence is most fractional is either forced together or forced apart by deleting columns. The master does not add the equality \(\sum_{k\ni r,s}\lambda_k=1\). The filter is the whole mechanism.
5. A single arc whose flow is nearest one half. The must-use side keeps only routes that contain that arc.

An integer node is one whose \(\lambda\) values sit within \(10^{-6}\) of integers. The plain BPC engine accepts that, including a cyclic column at value 1. The multi-stage engine also rejects a selected route that repeats a customer and a customer that appears in two selected routes. The incumbent updates only when the new value beats the old one by more than \(10^{-6}\). Open nodes whose LP value cannot beat the incumbent are dropped.

Budgets are hard-coded around the config. The root node receives at most 40 percent of the time limit. Each Farkas pricing call is capped at 5 seconds. Three pricing timeouts abort the node. Plain BPC then stops the tree. The multi-stage engine tries a restricted integer solve and then stops.

## From branch-and-bound to branch-and-price-and-cut

A route formulation has one variable \(\lambda_r\) per feasible route. For the CPTP-RCTW that set is exponential, because feasibility already includes capacity, windows, and the other resources. Branch-and-bound on the full extended formulation is impossible to write down. Column generation solves the linear relaxation of a restricted master and prices the missing routes.

The master in `VRPPMasterProblem` is a maximization of route profit. Profit of a column is revenue minus travel cost. Revenue is a per-kilogram price times waste, unless node prizes are passed, which the BPC engine does not do. Mandatory customers are equalities \(\sum_k a_{ik}\lambda_k=1\). Optional customers are packing inequalities \(\sum_k a_{jk}\lambda_k\le 1\). If a fleet limit \(K\) is present, \(\sum_k\lambda_k\le K\). There is no convexity row. Capacity \(Q\) is not a master row. It is enforced inside pricing, and it reappears as the right-hand side of rounded capacity cuts.

That linear program is the branch-and-bound node bound. Branch-and-price is branch-and-bound in which every node re-solves this linear program by column generation under the branching constraints. Branch-and-price-and-cut adds inequalities at the node and puts their duals into the reduced cost, so the next pricing call respects the cut. The engine docstring maps this sequencing to Barnhart, Hane, and Vance (2000): Phase I, then profit pricing, then a Lagrangian bound only after pricing has stopped, then cuts kept on one master that is never rebuilt. The paper they cite is an origin-destination multicommodity flow paper. This code does not solve that flow model. It copies the sequencing onto a route set-partitioning master. The manuscript is explicit about the same substitution.

The historical line the manuscript actually cites, each for one claim, is:

| Source | Role in the manuscript |
| --- | --- |
| Dantzig and Wolfe (1960), Gilmore and Gomory (1961) | Decomposition and delayed column generation |
| Desrochers, Desrosiers, and Solomon (1992) | Column generation for the VRPTW |
| Barnhart et al. (1998) | Branch-and-price as a general integer-programming framework |
| Barnhart, Hane, and Vance (2000) | Cuts whose duals return to pricing, and the pattern for divergence branching |
| Lübbecke and Desrosiers (2005), Desaulniers, Desrosiers, and Solomon (2005), Costa, Contardo, and Desaulniers (2019) | Surveys of column generation and of exact BPC for routing |
| Geoffrion (1974) | The set-packing LP bound is the Lagrangian dual of the coupling rows |
| Irnich and Desaulniers (2005), Feillet et al. (2004) | Labelling for the elementary resource-constrained shortest path |
| Baldacci, Mingozzi, and Roberti (2011) | ng-route relaxation |
| Christofides, Mingozzi, and Toth (1981) | \(q\)-routes |
| Righini and Salani (2009, 2008), Boland et al. (2006) | Decremental state-space relaxation |
| Jepsen et al. (2008), Pecin et al. (2017) | Subset-row and limited-memory rank-1 cuts |
| Lysgaard, Letchford, and Eglese (2004) | Rounded capacity cuts |
| Letchford, Eglese, and Lysgaard (2002) | Multistar inequalities |
| Ryan and Foster (1981) | Pair branching |
| Held and Karp (1971), Fisher (1981) | Subgradient steps for the Lagrangian pre-prune |
| Wentges (1997) | Dual smoothing |
| Irnich et al. (2010) | Reduced-cost arc fixing |

The manuscript’s own claim is that no published exact method takes CPTP-RCTW as its primary object. Sections “Methodology,” “Implementation,” “Computational Results,” and “Conclusions” are placeholders. The figures are one synthetic 18-customer instance, and the appendix says that toolkit is separate from the solver in the algorithm section.

## Pricing, including Farkas pricing

A column’s reduced cost, as the manuscript writes it, charges the prize and the visit dual on the arrival vertex and subtracts the fleet dual on the return. The code matches that shape and then adds cut terms. On a customer arc the label adds revenue of the head, subtracts travel cost and the visit dual, and subtracts a dual whenever the arc crosses a capacity set, uses an edge-clique edge, completes a subset-row pair, traverses an LCI arc, or crosses a multistar set. The fleet dual is subtracted once, on the return.

Phase I is classical Farkas pricing, not a Big-M artificial phase. The master asks Gurobi for a Farkas ray and negates it, because an uncovered mandatory row comes back negative and pricing wants a positive weight. It runs when the node LP is infeasible. If any customer is mandatory, the engine starts in Phase I, with every column objective wiped to zero. The first optimal LP restores profits and switches to Phase II. A positive combination of node weights on a new route is the certificate that the ray is not a proof of infeasibility. If no such route exists, the node is infeasible.

The ray that is actually priced is partial. Subset-row and edge-clique components are extracted and then cleared. Lifted-cover and multistar Farkas duals are not read. Acceptance ignores the route’s own reduced cost and keeps a column when the sum of node weights exceeds \(10^{-6}\). Each Farkas call returns at most five columns and is capped at five seconds.

Phase II labelling is a best-first label correction. A label carries reduced cost, node, load, the visited set, ng-memory, a Ryan–Foster mask, and a subset-row state. Load above \(Q\) rejects the arc. A repeat of a visited customer rejects the arc. There is no arrival time, service time, window, or route duration on the label. `route_budget` is a config field and is dropped before it reaches pricing.

The pricing problem is therefore one capacitated profitable tour for every problem name. A CVRP-style instance is the same model with every customer mandatory. The optional CPTP pricer (`pricing_engine="cptp"`) replaces the dynamic program by the external branch-and-cut in `logic/submodules/BPC-CPTP` when the matrix is symmetric and the node has no Farkas ray, no must-use arc, no Ryan–Foster pair, no subset-row cut, no arc LCI, and no multistar. That oracle separates its own subtour, rounded-capacity, and multistar cuts on the edge variables. Those cuts never become master rows. With time windows, the manuscript says labelling is the only pricer. The code has no window test to make that distinction. It simply never builds a clock.

Heuristic pricing, on by default, truncates each node to a nearest-neighbour successor list and, above 40 customers, may skip the dynamic program after a Clarke–Wright-style pass. Plain BPC confirms a zero-column result with one exact pass. The multi-stage engine treats an unfinished pricing call as an incomplete node and stops the tree. A node bound is a proved dual bound only when that exact pass really exhausted the elementary paths. The implementation below does not guarantee that.

## Cuts

Cuts are separated only after a column-generation inner loop has stopped, and only on an optimal LP. If any cut is added, pricing runs again. A relative objective change below \(10^{-5}\) stops further separation. The comment in `column_generation.py` attributes that order to Barnhart et al. (1998). Child nodes see the cuts because the master object is shared. `GlobalCutPool` archives them and has no replay into a rebuilt model. The only rows deleted between nodes are the local prize-collecting subtour forms.

With `cutting_planes="all"` the composite engine runs rounded capacity cuts, 3-subset-row inequalities, edge cliques, fleet cover, physical lifted covers, saturated-arc lifted covers, a knapsack cover, rounded multistars, and a limited-memory rank-1 engine. Ten cuts per round are split across those engines, so each family contributes about one cut.

What each family does in code:

- **Rounded capacity.** For a customer set \(S\), the row is the number of route crossings of the cut \(\delta(S)\), at least \(2\lceil\text{demand}(S)/Q\rceil\). Separation grows nearest-neighbour sets and also reads min-cuts from the depot. The engine then drops any set that is not entirely mandatory, because an optional customer can be skipped and the cut would be invalid. That restriction is also what the manuscript states, citing Lysgaard, Letchford, and Eglese (2004).
- **Prize-collecting subtours.** Forms in the style of Fischetti, Salazar González, and Toth (1997): \(x(\delta(S))\ge 2\), and two local forms that subtract visit variables. Their duals are not extracted, so pricing does not see them. New columns receive a crossing count only.
- **Rounded multistar.** The Letchford, Eglese, and Lysgaard (2002) form, stored as a nonpositive combination of columns. The label subtracts an arc coefficient that depends on whether the arc leaves, enters, or stays inside \(S\). The CPTP pricer cannot represent this dual and falls back to labelling.
- **3-subset-row.** \(\sum_k \lfloor |S\cap\text{route}_k|/2\rfloor\lambda_k\le 1\) for \(|S|=3\), following Jepsen et al. (2008). The label keeps a counter in \(\{0,1,2\}\) and subtracts the dual on the step from 1 to 2. The manuscript says the same, and it declines the weakened dominance Jepsen allows. Customers in an active subset-row cut are forced into every ng-set.
- **Edge clique.** Routes through an edge whose flow exceeds 1 are limited by 1. The dual is subtracted on that undirected edge.
- **Lifted covers.** Two mechanisms. If \(K=1\), a cover of the node knapsack \(\sum q_i y_i\le Q\) is lifted with the Gu, Nemhauser, and Savelsbergh coefficients and written on the route variables. For a saturated arc, routes that use the arc are covered and the dual is subtracted only on that directed arc. The manuscript cites Balas (1975), Gu et al. (1998), and Barnhart, Hane, and Vance (2000, §6) for these two.
- **Fleet cover and the knapsack-cover engine.** Both are weaker than their comments. The fleet engine writes \(\sum_k\lambda_k\le K\), which the fleet row already says. The knapsack engine writes a crossing inequality whose right-hand side is \(K\), not \(\sum_{j\in C}\lambda_j\le |C|-1\).

Limited-memory rank-1 cuts are enumerated for 3-subsets and 5-subsets with multipliers \(1/2\), citing Pecin, Pessoa, Poggi, and Uchoa (2017). The master has no method that stores the memory set. The fallback is the 3-subset-row routine, which rejects every set whose size is not 3. Five-customer cuts are dropped. Three-customer cuts become ordinary subset rows.

Comb inequalities exist as a data class. Nothing separates them. The flag `enable_comb_cuts` is stored and never read. Hypotours, GK cuts, template inequalities, strong-degree cuts, \(k\)-path cuts, and 2-path cuts are absent. No master cut reads travel time. The manuscript says this in one sentence: none of the current families uses the time windows or the resources in \(\mathcal{R}\). The inequalities that would exploit selection, windows, and resources are the empty “Methodology” section.

Inside the CPTP submodule the pricing IP has its own subtour elimination, a modulus form of the rounded capacity cut, and a generalized large multistar. Comb, rounded GLM, and the set-infeasibility cuts (a set whose cheapest tour exceeds the upper bound cannot be taken together) are implemented and off by default. Those cuts tighten the pricing integer program. They do not change the master reduced cost except through the one route the pricer returns.

## Relaxations

**Linear relaxation of the master.** \(\lambda\) is continuous on \([0,1]\). This is the number the tree prunes with. It bounds the original problem only together with a pricing proof that no positive reduced-cost elementary route remains, under the cuts currently in the master.

**Lagrangian relaxation of capacity.** For one vehicle the code dualizes \(Q\). The dual function is

\[
L(\lambda)=\max_{A}\Big[\sum_{i\in A}(R-\lambda)w_i - C\cdot\mathrm{dist}(A)\Big]+\lambda Q,\qquad\lambda\ge 0.
\]

The inner problem is an uncapacitated orienteering MIP: flow, degree, and lazy DFJ subtours, with no capacity row. The subgradient is \(g=Q-\sum w\), the step is the Polyak step \(\alpha=\theta\max(L-\mathrm{LB},10^{-8})/g^2\) with \(\theta=1\), and \(\lambda\) is projected onto the nonnegative reals. The bound is used to discard a node before column generation, and only after an incumbent exists. Plain BPC multiplies it by the fleet size, which is a valid but weak fleet bound when every orienteering solve is optimal. A timed-out maximization returns an incumbent that can lie below the true \(L(\lambda)\), so the recorded upper bound can be too small and the prune is not safe under the five-second orienteering limit. The manuscript’s version of this argument needs the triangle inequality, concatenates \(K\) routes into one tour, allows load \(KQ\), and drops time windows. Waiting is not counted. The code’s orienteering model has no clock at all.

**A second Lagrangian-style bound, after pricing.** Once the pricer reports exhaustion,

\[
z_{UB}=z_{LP}+K\cdot\max(0,\bar c^\star).
\]

If this is below the incumbent, the node is pruned. The manuscript states the same inequality. Exhaustion in code is “best reduced cost at most the optimality gap,” after a pricer that can hide elementary routes. The bound is then not a proof.

**ng-routes, as the manuscript defines them.** The neighbourhood \(\mathcal{N}_j\) is \(j\) and its nearest customers. Memory updates by \(\Pi'=(\Pi\cap\mathcal{N}_j)\cup\{j\}\). A customer may be repeated if the previous visit has fallen out of memory. The resulting bound is valid and weaker than elementary pricing. Decremental state-space relaxation, cited from Righini and Salani and from Boland et al., puts a repeated vertex back into the neighbourhoods for a fixed number of rounds, until the priced paths are elementary. Subset-row customers and Ryan–Foster endpoints are forced elementary. The default neighbourhood size in both the paper’s parameter table and `BPCConfig` is 12. The multi-stage config uses 8.

**ng-routes, as the labeller implements them.** The memory update is the usual one. Extension still rejects every vertex already in `visited`. Memory is always a subset of the visited set, so the extra memory test never forbids an arc that elementarity would allow. ng changes only dominance: a label can be deleted because its memory is not contained in the survivor’s memory, even when its visited set is not contained either. The search can therefore miss the best elementary route. It also cannot emit the cyclic columns that the decremental repair in `column_generation.py` is written to detect. `enable_dssr` and `dssr_max_iters` are never read. The docstring citation of Baldacci et al. (2011) describes a relaxation this labeller does not run.

**Other relaxations that do run.** A resource-free completion bound is a long walk of at most \(n\) arcs on prize, distance, visit duals, and capacity-cut duals. It legally skips a hopeless return to the depot. It does not prove that a label has no feasible completion under capacity. Reduced-cost arc fixing uses those same bounds and runs whenever an incumbent exists. The config switch is not consulted. The completion bounds ignore capacity, so the test is a relaxation. Heuristic successor lists are a graph restriction, not a bound. The multi-stage knapsack pre-selection deletes customers before pricing. Unless `exact_mode` is set or the instance has at most 12 nodes, the tree bound is a bound on a smaller instance. There is no \(q\)-route routine anywhere under `logic/src/policies`.

## What the two engines return

`run_bpc` builds the master, seeds it with a greedy column if asked, chooses Phase I when some customer is mandatory, and walks the tree. If the tree never finds an integer node, it solves the restricted master as an integer program for a short time and keeps the better of that solution and the greedy start.

`run_ms_bpc_sp` adds stages around the same tree. A knapsack first drops optional customers. The tree then runs with stricter integer recognition and stops when pricing is incomplete. After the tree, a regret-2 insertion rebuilds a route on the original instance, and a set-partitioning MIP recombines the column pool. Those two stages can raise the profit. They are not part of the branch-and-bound certificate. The issue-90 notes in the workspace memory record cases where this multi-stage path finished below a brute-force optimum. That is consistent with a heuristic node filter plus a truncated tree, not with a finished exact BPC.

## Parameters

The manuscript’s Table of parameters and `BPCConfig` agree on the intended defaults: depth-first search, visit branching first, divergence as the routing branch, all cut families, orthogonality 0.7, 10 cuts per round, 50 columns per pricing call, 150 column-generation iterations, relative gap 0.5 percent, reduced-cost tolerance \(10^{-6}\), ng-size 12, 8 decremental rounds, Lagrangian pre-pruning on, 40 subgradient iterations, arc fixing on, greedy warm start on. The paper says these defaults are not tuned. Wentges’ smoothing weight is drawn as \(0.5\) on one figure and is not in the table.

Defaults that change the run in practice:

| Knob | Plain BPC | Multi-stage |
| --- | --- | --- |
| Time limit | 60 s | same family of fields |
| Node cap | 2000 | 1000 |
| Gap | 0.005 | \(10^{-4}\) |
| Column-generation iterations | 150 | 50 |
| Cuts | `"all"` | saturated-arc LCI only |
| ng-size | 12 | 8 |
| Lagrangian pre-prune | on | off |
| Pricing engine | labelling or CPTP | labelling only |

Several switches do nothing when flipped: `enable_dssr`, `dssr_max_iters`, `max_cut_iterations`, `use_spatial_partitioning`, and `enable_reduced_cost_arc_fixing`. Arc fixing still runs. Dual smoothing is described on the master class and has no update. `route_budget` never becomes a resource.

## How the same machine specializes

The master and the labeller are one formulation. The problem name changes which rows are equalities and which resources the label carries.

- **Prize-collecting tour, one vehicle.** Optional rows are packing constraints. Mandatory rows, if any, are equalities and trigger Farkas pricing. Capacity is the label load. This is what `run_bpc` solves today.
- **Team version.** The fleet row and the fleet branch appear when `vehicle_limit` is set. The Lagrangian pre-prune scales by \(K\). Nothing in the learning-environment instance dicts carries \(K\). Those environments model one vehicle and several trips, which is a different coupling: the shift resets per trip and the day clock does not.
- **Full-coverage VRP.** Every customer is mandatory. Rounded capacity cuts become legal on every customer set. Pricing is still a profitable tour whose prizes happen to be waste. A pure min-cost CVRP would need the prize term removed and the objective sign flipped. The code does not have that mode.
- **VRPTW.** The manuscript’s reduction is: set \(M=N\) and prizes to zero, and keep the windows. The labeller would have to grow an arrival resource \(t'=\max(a_j,\,t+s_i+t_{ij})\) and reject \(t'>b_j\). Dominance would grow a “no later arrival” test. That resource is absent, so the VRPTW reduction is not available in the engine.
- **Orienteering.** The survey and the manuscript treat orienteering as prize maximization under a length limit. The Lagrangian subproblem is an uncapacitated orienteering MIP. The routing engine itself maximizes prize minus cost under capacity, which is the profitable-tour objective, not the orienteering objective.
- **CPTP branch-and-cut as a pricer.** Jepsen et al. (2014), as used here, solve one symmetric capacitated profitable tour and return one route. The manuscript limits it to instances with no windows and no extra resources. The code limits it to nodes whose dual structure the edge formulation can express.

## A research program for a BPC on the capacitated PTP with time windows and time constraints

The manuscript already names three items in the conclusions, and the code shows why they are first. The rest follow from holes that make a windowed bound either missing or unsafe.

**Put the clock on the label, and make the bound honest.** The extension the manuscript writes is the right one: reduced cost plus arc profit minus cut duals, load \(u+q_j\), time \(w'=\max\{a_j,\,w+s_v+t_{vj}\}\), and each extra resource \(r^\rho+\tau^\rho_{vj}\), with rejection on capacity, the window, the resource, the ng-test, and the return to the depot. Service time has to sit in that same arc, in the forward label and in the completion bound. For CPTPTWTC the shift is a second clock that resets at the depot, while the window clock does not. A label that cannot separate those two clocks will accept a trip the environment masks, or reject a trip the environment allows. Dominance must add “arrival no later” and “resources no larger.” Until `visited`-set dominance is restored, or ng-routes are changed so that repeats outside memory are truly allowed, adding time makes the existing miss larger: the surviving label can be feasible for a different continuation than the discarded one.

**Implement ng-routes as a relaxation, then repair them.** Allow a revisit when the customer has left \(\Pi\). The master bound stays valid and becomes weaker, which is the Baldacci–Mingozzi–Roberti tradeoff the docstring cites. Decremental state-space relaxation should read `dssr_max_iters` and add each repeated vertex to the neighbourhoods for that many rounds, which is the paragraph already in the manuscript. Forcing subset-row customers and Ryan–Foster endpoints into every neighbourhood is the right companion and is partly present in `_prepare_node_relaxation`. A \(q\)-route, which the manuscript uses only as a completion bound in a figure, is the other classical relaxation: drop elementarity, keep load and time, and accept a weaker bound that is cheap to compute. Either relaxation is a legitimate bound. The current hybrid, elementary extension with ng-dominance, is not.

**Completion bounds that keep the windows.** The manuscript lists this as future work, and the arc-fixing caption says the illustrated bounds relax windows and would delete more arcs if they kept them. Irnich et al. (2010), already cited, is the path-reduced-cost template: a forward and a backward bound. Those bounds have to carry load and time, or both the depot-close test and arc fixing will delete arcs that a windowed optimum uses. Bidirectional labelling with dynamic half-way points (Tilk et al., 2017) is the other item the conclusions name. It pays for itself only after the resources exist, because the halfway point is defined on the resource that grows fastest, which on this problem is usually time.

**Price the Farkas ray completely, then smooth the duals.** A windowed master is infeasible for new reasons: a mandatory customer can be unreachable inside the depot window. Phase I has to stay a pure Farkas phase, which the manuscript insists on and the code already attempts. The ray should include subset-row, clique, lifted-cover, and multistar components, with one sign convention on every arc. Wentges smoothing, \(\tilde\pi=\alpha\hat\pi+(1-\alpha)\pi\), is specified and would matter more once time makes the duals oscillate. The weight \(\alpha\) needs a parameter. The figure’s \(0.5\) is not a study.

**Cuts that see time, selection, and the shift.** The empty methodology section is the paper’s own opening. Families that are standard on the VRPTW and absent here are the natural first candidates, used only where they survive optional customers:

- Infeasible-path elimination and its conflict version: a sequence that violates a window, the shift, or a resource cannot appear in any column. With optional customers the inequality is on the visits that the sequence would require, in the spirit of the submodule’s set-infeasibility cuts.
- Two-path cuts: if \(S\) cannot be served by one vehicle under capacity and the windows, at least two routes must cross \(\delta(S)\). The mandatory-set restriction used for rounded capacity cuts is the validity condition to copy.
- A rounded capacity cut that also knows the shift. Demand may fit in \(Q\) while service plus travel does not fit in the trip budget. The right-hand side then comes from a small bin-packing problem in two resources, not from \(\lceil\text{demand}/Q\rceil\) alone.
- Limited-memory rank-1 cuts stored as themselves. The separator already enumerates the \((\tfrac12,\ldots,\tfrac12)\) sets of size up to five. The label needs a memory reset outside \(M\), which is the Pecin et al. (2017) point of the cut. Falling through to a 3-subset-row throws the strengthening away.
- Subset-row dominance weakened as Jepsen et al. (2008) allow, instead of requiring identical states. The manuscript already flags this as stricter than the citation.
- A cover on the shift knapsack for \(K=1\), parallel to the physical lifted cover on load. For \(K>1\) the same cover is not valid as a global row, which is why the physical LCI already refuses to fire when the fleet is larger than one.

Prize-collecting structure changes every one of these. A cut derived under “everyone in \(S\) is served” is valid here only for \(S\subseteq M\), or it has to be written with the visit variables the way the prize-collecting subtours already are.

**A Lagrangian pre-prune that knows which constraint was dropped.** Dropping windows still gives a valid upper bound on the more constrained problem, and only when the orienteering MIP is solved to optimality. The code should refuse to prune on a timed-out incumbent. For a duration limit the manuscript already proposes consumption \(s_i+t_{ij}\) and a budget \(KU^T\), without counting waiting. That number is a relaxation bound, and it should be labelled as one. Soft window penalties are different: if the penalty depends on arrival, leaving it out of \(L(\lambda)\) destroys the upper bound. The penalty has to enter both the column profit and the reduced cost, and time stays in the dominance state because two labels at the same node can face different future penalties.

**Match the coupling to the application.** A BPC for CPTPTWTC has to decide whether \(K\) is a concurrent fleet or a sequence of trips of one vehicle. The manuscript is the first. The environment is the second. Multi-trip pricing adds a depot reset of load and shift in the middle of a column, while the day clock and the windows keep accumulating. The master then has one variable per daily schedule, not per trip, or it keeps trip variables and adds a linking constraint on the shared clock. Those are different extended formulations. The simulator’s multi-day stock, overflow allowance, and “visit when the bin exceeds a fill threshold” are the stochastic multi-period extension the conclusions already list. They do not belong inside the deterministic pricing label. They belong in a master that chooses which customers are mandatory on each day, with the deterministic CPTP-RCTW as the daily pricing problem.

**Certificate hygiene, before new components.** A windowed BPC will be believed only if the node bound is either proved or marked incomplete. The pieces to fix, because they already affect the capacitated problem, are: ng-dominance versus elementarity, SEC duals omitted from the reduced cost, multistar duals omitted from the pool’s reduced-cost check, the partial Farkas ray, the unsafe Lagrangian prune under a time limit, and the multi-stage knapsack that deletes customers and then still reports a tree bound. The manuscript’s rule is the right one: heuristic pricing may propose columns, and only an unrestricted labelling call certifies the node.

The shortest path through that list is a forward labeller with load, shift, and time windows, elementary dominance, Farkas Phase I on the mandatory rows, rounded capacity cuts on mandatory sets, 3-subset-row cuts, and a completion bound that carries the same three resources. ng-routes, decremental repair, limited-memory rank-1 cuts, bidirectional search, and the new window-dependent inequalities are the strengthening steps the manuscript has already named, in that order.
