# Adapting and Optimizing Branch-Cut-and-Price (BCP) for the Vehicle Routing Problem with Profits and its Multi-Period Variant: A Comprehensive Research Survey

## Introduction

The Vehicle Routing Problem with Profits (VRPP), introduced systematically in the seminal survey of Feillet, Dejax & Gendreau (2005, *Transportation Science*) and consolidated for the multi-vehicle case in the chapter by Archetti, Speranza & Vigo (2014) in Toth & Vigo's *Vehicle Routing*, is a class of routing problems sharing a defining feature: **node selection is itself a decision variable**. Each customer carries a prize/revenue, vehicles have capacity and route-duration/length budgets, and the planner maximizes collected prize minus travel cost (the *profitable* objective), or maximizes prize subject to budget (the orienteering objective), or minimizes cost subject to a quota of prize (the prize-collecting TSP/PCTSP objective). The most-studied multi-vehicle members of the family are the Team Orienteering Problem (TOP), the Capacitated Team Orienteering Problem (CTOP), the Capacitated Profitable Tour Problem (CPTP), and their time-windowed extensions (TOPTW, CTOPTW). The Multi-Period VRPP (MP-VRPP / PVRPP), formalized by Tricoire, Romauch, Doerner & Hartl (2010) and studied heuristically by Zhang, Lam & Sleigh (2013) as the mVRPP, adds a planning horizon over which optional customers may be visited under additional inter-period constraints.

While Branch-Price-and-Cut (BPC) has, since Pecin, Pessoa, Poggi & Uchoa (2017a,b), Pecin, Contardo, Desaulniers & Uchoa (2017), Sadykov, Uchoa & Pessoa (2021) and Pessoa, Sadykov, Uchoa & Vanderbeck (2020 — VRPSolver), become the *de facto* state of the art for classical VRPs, its adaptation to VRPP requires non-trivial modifications because (i) the master problem covering constraints become *less than or equal* (or implicit), (ii) sparse optimal solutions destabilize column generation duals, (iii) the prize structure modifies the reduced-cost objective of pricing, and (iv) the family of valid inequalities and branching rules must accommodate the visit/no-visit dimension. The current frontier — Boussier, Feillet & Gendreau (2007) for TOP; Archetti, Bianchessi & Speranza (2013) for CTOP/CPTP; Poggi, Viana & Uchoa (2010) for the BCP of TOP; Keshtkaran, Ziarati, Bettinelli & Vigo (2016) for enhanced TOP; Jepsen, Petersen, Spoorendonk & Pisinger (2014) for CPTP; Li, Zhu, Peng et al. (2024) for TOP with interval-varying profits — solves classical TOP/CTOP benchmarks with up to ~100 customers, but lags far behind capacitated VRP exact methods which now solve 400+ customer instances. This survey synthesizes the techniques available, distinguishes (a) what is well-established, (b) what is partially attempted, and (c) what is open, with concrete proposals for new directions.

---

## 1. Master Problem Adaptations

### 1.1 Set Partitioning, Set Covering and Set Packing for VRPP

In the classical CVRP/VRPTW BPC of Desrochers, Desrosiers & Solomon (1992), Fukasawa et al. (2006), Baldacci, Christofides & Mingozzi (2008), and Pecin et al. (2017), the master is a Set Partitioning Problem (SPP) with constraints `Σ_{r∈Ω} a_{ir} λ_r = 1 ∀i ∈ N`. For VRPP, customer i need not be served, so the natural formulation replaces equality by `Σ_{r∈Ω} a_{ir} λ_r ≤ 1` (a **Set Packing Problem, SPaP**), and the prize is collected only if customer i is on at least one selected route. This is the formulation used by Boussier, Feillet & Gendreau (2007), Archetti, Bianchessi & Speranza (2013) and Poggi et al. (2010). Equivalently one writes the master as

  max Σ_{r∈Ω} (p_r − c_r) λ_r
  s.t. Σ_r a_{ir} λ_r ≤ 1 ∀i ∈ N (covering / packing)
     Σ_r λ_r ≤ K (fleet size)
     λ_r ∈ {0,1} ∀r ∈ Ω,

where p_r = Σ_{i∈r} π_i is the prize collected by route r and c_r its travel cost. For the CPTP and prize-collecting TSP the constraint Σ a_{ir} λ_r ≤ 1 is sometimes replaced by an *implicit* covering constraint together with an explicit y_i ∈ {0,1} variable indicating whether i is visited; this gives the **hybrid extended formulation** used by Jepsen, Petersen, Spoorendonk & Pisinger (2014) for the CPTP. Set covering (`≥ 1`) is equivalent for binary integer feasible solutions with non-negative prizes (one can always shrink a route covering i twice).

**Tightness ranking (LP relaxations):**

- **Set Partitioning** (= 1) is generally *tighter* than Set Covering (≥ 1), which is generally tighter than Set Packing (≤ 1) when the corresponding problem is feasible. For VRPP, however, partitioning is infeasible because customers may be unvisited. Among the *valid* options for VRPP, the literature consensus (Costa, Contardo & Desaulniers 2019, *Transportation Science* survey) is that **Set Packing with explicit y_i variables and linking equalities y_i = Σ_r a_{ir} λ_r** yields the same LP bound as the implicit pure packing formulation but exposes useful primal variables for cutting and branching.
- **(a) Well-established:** the equivalence of LP bounds between explicit and implicit packing forms.
- **(c) Open:** no comprehensive empirical comparison of partitioning-with-dummy-routes vs. pure packing for VRPP exists. A promising direction is to introduce a "skip-route" of zero length and zero prize for each unvisited customer to convert ≤ into = (preserving SPP machinery), inspired by the dummy-shift trick used in crew scheduling.

### 1.2 Stabilization Implications of Sparse Optimal Solutions

A defining feature of VRPP optima is **sparsity**: only a fraction of customers are visited. In LP terms, the dual variables π_i associated with unvisited customers naturally take value 0 (since the constraint Σ a_{ir} λ_r ≤ 1 is slack), and the dual polyhedron has an enlarged set of degenerate vertices. This causes pronounced "dual oscillation" or zig-zagging during column generation — well known and worse than for classical CVRP/VRPTW. Consequently, **dual stabilization is more critical** for VRPP than for partitioning VRPs.

Established techniques (most documented for classical VRP) include:
- **Boxstep / proximal point** of du Merle, Villeneuve, Desrosiers & Hansen (1999).
- **Bundle methods** of Briant, Lemaréchal, Meurdesoif, Michel, Perrot & Vanderbeck (2008) and Frangioni & Gorgone (2014).
- **In-out separation** and **dual price smoothing** of Pessoa, Sadykov, Uchoa & Vanderbeck (2013, 2018, *INFORMS J. Comput.*) — the auto-regulating smoothing of Pessoa et al. (2018) is now the default in VRPSolver and is particularly suitable for VRPP because it tunes itself when duals are unstable.
- **Interior-point warm starts** (Rousseau, Gendreau & Feillet 2007).
- **Penalty-based stabilization**: linear penalty around an estimate of the dual optimum (Ben Amor, Desrosiers & Frangioni 2009).

**(b) Mixed results:** Briant et al. (2008) showed that pure proximal bundle can hurt total CPU because it requires many exact pricing solves. Auto-smoothing with an additional in-out direction (Pessoa et al. 2018) is the most robust generic choice. **(c) Open:** No paper has tested *VRPP-specific* dual estimates as the smoothing centre, e.g. setting the initial dual estimate of customer i to π_i (the prize value), exploiting the natural lower bound π_i = 0 for the customer's dual variable in the packing form. This is a high-payoff research direction: with prizes as an a-priori dual centre, the in-out direction immediately points into the feasible dual region, as observed in the analogous "Lagrangian pricing" literature (Löbel 1998).

### 1.3 Restricted Master Heuristics (RMH) for VRPP

Archetti, Bianchessi & Speranza (2013) explicitly introduced an RMH at every B&B node specifically to obtain primal bounds for CTOP/CPTP. Because VRPP solutions are sparse, the IP solution of the restricted master typically has a much smaller support than for VRPTW, and modern MIP solvers solve the restricted master quickly. **(a) Well-established:** RMH is critical in VRPP — it is the principal source of primal bounds. **(c) New direction:** *prize-guided RMH* — instead of solving the IP on all current columns, only allow columns containing the highest-prize customers (top-K by π_i) at each call; this exploits the structure that high-π customers should appear in optima.

### 1.4 Column Pool Management and Aging

Because in VRPP many columns generated early in a branch are "fragments" (high-cost, low-prize routes generated with poor duals), aging columns out of the pool aggressively is more damaging than for classical VRPs (where any column might still be useful). Established practice (Vanderbeck 2005, Sadykov, Vanderbeck, Pessoa, Tahiri & Uchoa 2019) is to keep a pool of all non-dominated columns observed thus far. **(c) Open:** *prize-density-based pool management*: keep all columns with prize-per-unit-resource above a threshold, since these are "structurally promising" regardless of current duals.

### 1.5 Multi-Commodity Flow / Path-Flow Embedded Formulations

Poggi, Viana & Uchoa (2010) introduced for the TOP an **extended formulation** with edge variables x_{ij}^t indexed by the time t at which arc (i,j) is placed in the route, dualized by Dantzig–Wolfe over time-indexed flow conservation. They obtained better root bounds than Boussier et al. (2007). The equivalent of Letchford & Salazar-González's (2019) hybrid CVRP formulation that combines Set Partitioning with path-flow, projecting out arc variables, has yet to be fully developed for VRPP — **(c) open**.

---

## 2. Column Generation / Pricing Subproblem

### 2.1 ESPPRC as the VRPP Pricing Subproblem

The canonical pricing subproblem in BPC for VRPP is the **Resource-Constrained Elementary Shortest Path Problem (ESPPRC)** of Feillet, Dejax, Gendreau & Gueguen (2004, *Networks* 44). For VRPP with route duration T_max and capacity Q, a column corresponds to an elementary path P = (0, i_1, …, i_k, n+1) with `Σ d_{i_l} ≤ Q` and `Σ τ_{i_l i_{l+1}} ≤ T_max`. The **modified arc cost** for a given dual vector (π, σ) of the master is
  c̃_{ij} = c_{ij} − π_j ; (sink/source modifications absorb σ, the fleet dual).
For VRPP, the prize π_j_master appears *additively* via the master objective (max p_r − c_r), so the pricing problem solves
  min { Σ_{(i,j)∈P} c_{ij} − Σ_{j∈P\{0,n+1\}} (π_j_value + π_j^{dual}) },
i.e., the **modified arc costs absorb both the dual prices and the original prize values**: c̃_{ij} = c_{ij} − π_j_value − π_j^{dual}, with the convention that an unvisited customer simply does not contribute. This means the pricing graph almost always has many negative arcs, making the underlying SPPRC harder than for classical VRPTW.

Equivalently, one may keep the "min total reduced cost minus prize" pattern: search for routes with negative reduced cost = travel cost − collected prize − duals, and only routes with negative value are added. A subtlety unique to VRPP is that **the trivial empty route (depot → depot) has reduced cost 0**, so column generation terminates legitimately when no route with strictly negative reduced cost exists; this is *not* a failure to find a route — it is a proof of LP optimality. Boussier et al. (2007) emphasize this: in VRPP the pricing subproblem may correctly answer "no profitable route".

### 2.2 Label-Setting and Label-Correcting Algorithms

The standard ESPPRC label-setting algorithm of Feillet et al. (2004) extends partial-path labels with components (cost, resource, visited-set bitmask). For VRPP this becomes:
  L = (v, c̃, q, t, V, p)
where p is the *cumulative prize* (handy for ordering and dominance, though redundant with cost when prizes are folded into arc costs) and V ⊆ N is the set of visited (or unreachable) customers. Dominance: L dominates L' if v(L) = v(L'), c̃(L) ≤ c̃(L'), q(L) ≤ q(L'), t(L) ≤ t(L'), V(L) ⊆ V(L'). Boland, Dethridge & Dumitrescu (2006, *Operations Research Letters*) and Righini & Salani (2008, *Networks*) made label-setting practical by **bidirectional labeling** and **state-space relaxation**.

Bidirectional labeling (Righini & Salani 2006, 2008; Tilk, Rothenbächer, Gschwind & Irnich 2017 for *asymmetric* dynamic half-way points) is by now standard. For VRPP, the natural critical resource is route duration T (or capacity Q); the **half-way criterion** is t(L) ≤ T_max/2 for forward labels and analogously for backward labels, and forward/backward labels are joined whenever the union resource consumption is feasible. The **modification needed for VRPP** is that the join condition must also check that the merged path's prize minus cost is negative (otherwise the column is not improving). Righini & Salani (2009, *Computers & Operations Research*) explicitly applied bounded bidirectional DP with DSSR to the OPTW.

For TOP and CTOP, Boussier et al. (2007) used a forward-only label-setting algorithm with prize-as-resource; Archetti, Bianchessi & Speranza (2013) introduced acceleration techniques: **completion bounds** on prize collected by extension, **reachability-based dominance**, and **2-cycle elimination**. Keshtkaran et al. (2016) employed bidirectional DP with two-phase dominance relaxation, closing 17 previously open TOP instances.

### 2.3 Heuristic Pricing

A well-established acceleration is **heuristic pricing** before exact pricing:
- **Tabu search** for ESPPRC (Desaulniers, Lessard & Hadjar 2008; Boussier et al. 2007).
- **Restricted graph pricing** (Chabrier 2006).
- **k-shortest paths with prizes** — for VRPP a natural variant is to enumerate the k best paths in the *non-elementary* relaxation and filter for elementarity.
- **ng-route heuristics** (Baldacci, Mingozzi & Roberti 2011) — the dominant paradigm; see §3.
- **Pulse algorithm** (Lozano & Medaglia 2013, Lozano, Duque & Medaglia 2016, *Transportation Science*): a depth-first branch-and-bound search that uses bounding rather than full dominance. Duque, Lozano & Medaglia (2014) specifically developed pulse for the OPTW. Cabrera, Medaglia, Lozano & Duque (2020) extended to bidirectional pulse for the constrained shortest path. Pulse is competitive with label setting on stand-alone OPs but inside CG it primarily uses **rollback pruning** rather than full dominance.

### 2.4 Multiple Pricing

Generating multiple columns per pricing call is well-established. For VRPP-specific tactics:
- **Diverse-coverage multiple pricing**: among the columns with negative reduced cost, retain those whose visited sets V have small pairwise intersection. This counters dual oscillation due to sparse optima.
- **(c) Open**: a *prize-stratified* multiple pricing rule: at each iteration, return the best route from each "prize stratum" (e.g., quartiles of total prize), encouraging diversity in the column pool.

### 2.5 Resource Extensions Specific to VRPP

In addition to the standard duration/capacity resources, the VRPP literature uses:
- **Prize-as-resource** (Boussier et al. 2007), useful when a minimum-prize-per-route constraint is present (CPCVRP).
- **Negative-reduced-cost-as-resource** for completion bounds (Archetti, Bianchessi & Speranza 2013).
- **Time-window resources** for OPTW/TOPTW with possibly multiple windows (Tricoire et al. 2010 — the multiple-time-window feasibility check is itself a solvable subproblem).

### 2.6 Multi-Period Pricing

For MP-VRPP, the typical design is **one ESPPRC per period d**, each with its own dual vector π^d on customer-period covering rows. The pricing graph is (V_d, A_d) — possibly identical across days but with day-dependent durations and prizes — and the master couples through Σ_d λ_r^d ≤ K_d (per-day fleet caps) plus customer-frequency constraints. A more aggressive variant (used by Athanasopoulos & Minis 2013 in a heuristic context, and conceptually adopted in Pirkwieser & Raidl 2009b for PVRPTW) is to enumerate **multi-day route patterns** in a single pricing graph layered by day; this trades a larger SPPRC for fewer cross-period coupling constraints.

---

## 3. Relaxation Strategies for the Pricing Subproblem

### 3.1 Why Elementarity Matters More in VRPP

In classical VRPTW, non-elementary q-routes (with 2-cycle elimination) can be priced in pseudo-polynomial time and the dual bound loss is moderate (Desrochers, Desrosiers & Solomon 1992). In VRPP, however, **a non-elementary route can spuriously collect the prize of a customer multiple times**: if customer i is allowed to be visited twice in a relaxed pricing path, the path picks up prize π_i twice, leading to an inflated estimate of the most profitable column. Even when the master constraint Σ_r a_{ir} λ_r ≤ 1 caps the total visit fraction, the LP gets fooled because the route's *reduced cost* is overestimated. Hence VRPP requires either elementary pricing or carefully controlled relaxations.

### 3.2 ng-Route Relaxation

Baldacci, Mingozzi & Roberti (2011, *Operations Research* 59) introduced the ng-route relaxation: each customer i has a neighborhood N_i ⊆ N of its closest |N_i| customers; cycles through customer i are forbidden only inside the "memory" set. For VRPP, two adaptations are documented:
- Park, Tae & Kim (2017) explicitly used ng-route relaxation for the CTOPTW.
- Cortés-Murcia, Prodhon, Afsar & Cattaruzza (2022) used ng-paths in BPC for a profitable-tour electric variant.
- Roberti & Mingozzi (2014, *Transportation Science*) developed **dynamic ng-path** that grows N_i adaptively until elementarity holds for the LP optimum.

**ng-bound tightness in VRPP:** Empirically (Park et al. 2017; Bulhões, Sadykov & Uchoa 2018 for the minimum latency problem; Pessoa et al. 2020) the ng-bound is *slightly* weaker than the elementary bound but the pricing speedup more than compensates. **(c) Open question:** For VRPP the bound deterioration may be *more severe* than for CVRP because prize double-counting on small cycles is amplified. A natural extension is **prize-aware ng-routes**: each i's neighborhood is the set of *high-prize* nearby customers (where double counting is most damaging), rather than just nearest-by-distance. To our knowledge this calibration rule has not been published.

### 3.3 Decremental State-Space Relaxation (DSSR)

Boland, Dethridge & Dumitrescu (2006) and independently Righini & Salani (2008, 2009) developed DSSR: start from a fully relaxed (q-route) pricing problem and iteratively expand the "critical" set Θ ⊆ N of customers that must be elementary, until the resulting path is elementary. Righini & Salani (2009) explicitly used DSSR for the OPTW; Keshtkaran et al. (2016) used it for TOP. DSSR converges in few iterations because optima are sparse — exactly the VRPP regime.

**(b) Mixed:** the choice of *trigger* (which Θ to expand on each iteration) matters. The strategies HMO (most-overvisited), ALL, and FIRST were compared by Righini & Salani (2009) and showed no clear dominance. **(c) Open:** *prize-weighted DSSR triggers*: insert into Θ the customer with the highest π_i × overvisit count; the rationale is that double-counting a high-prize customer is more damaging.

### 3.4 ng-DSSR Combination

The combination of ng-routes with DSSR — call it "ng-DSSR" — is now embedded in VRPSolver (Pessoa et al. 2020). The ng-relaxation provides cheap initial routes; DSSR-style expansion is triggered by detecting the smallest set of customers visited multiple times. **(c) Promising:** for VRPP one could initialize DSSR's critical set Θ with the top-K highest-prize customers (cardinal to optimal solutions) — a *seeded DSSR* — rather than starting empty.

### 3.5 k-Cycle Elimination

Irnich & Villeneuve (2006, *INFORMS J. Comput.*) showed that increasing k in k-cycle elimination from 2 toward elementarity considerably tightens bounds for VRPTW. For VRPP, the trade-off is shifted: the detrimental effect of small-k cycles (revisiting i) is *prize-amplified*. **(b) Mixed:** Boussier et al. (2007) used 2-cycle elimination in their TOP pricing; later works moved to ng-routes which subsume k-cycle elimination as |N_i| grows.

### 3.6 Completion Bounds

A completion bound is an over-estimate of the contribution that an extension of a partial label can add. Lozano, Duque & Medaglia (2016) and Cabrera et al. (2020) deploy strong cost-based bounds. For VRPP the natural completion bound is **Lagrangian or LP-based prize-budget bounds**:
  ub_complete(L) = solution of a knapsack with items = remaining customers, weights = min residual time/capacity, profits = π_j + π_j^{dual},
which can be precomputed once per pricing call. Archetti, Bianchessi & Speranza (2013) used a similar idea to aggressively prune labels in the CTOP. **(c) New direction:** *prize-Lagrangian completion bounds*: compute a Lagrangian multiplier on the duration constraint and use the resulting unconstrained but penalized prize as a completion bound, leveraging Beasley & Christofides (1989).

### 3.7 State-Space Augmentation for Cuts

When subset-row cuts (SRCs) are added (§4), the labels must carry an additional state per active SRC (the modulo-2 count of visited customers in the SRC's defining set). This is the "non-robust cut" issue of Jepsen et al. (2008), Desaulniers, Lessard & Hadjar (2008) and Pecin et al. (2017a, 2017b — limited memory). **(a) Established:** for VRPP, limited-memory rank-1 cuts of Pecin, Pessoa, Poggi, Uchoa & Santos (2017) carry over directly because they are defined over master variables; the labelling overhead is identical. **(c) Open:** computational study of limited-memory R1Cs *combined with prize-stratified ng-routes* — no paper has exhaustively benchmarked the interaction for VRPP variants.

### 3.8 Alternative Pricing Formulations

- **Time-expanded DAG pricing**: Poggi, Viana & Uchoa (2010) used a time-indexed graph where the SPPRC becomes a shortest path in a DAG with O(n × T_max) nodes; viable when T_max is moderate, useful for OPTW.
- **Pseudo-polynomial pricing on bucket graphs**: Sadykov, Uchoa & Pessoa (2021, *Transportation Science*) introduced bucket-graph labelling, which has been integrated with VRPSolver and applies *mutatis mutandis* to VRPP variants.
- **Pulse algorithm** for stand-alone pricing (Lozano et al. 2016, Cabrera et al. 2020).
- **Constraint-Programming-based pricing**: Rousseau, Gendreau, Pesant & Focacci (2004) and Tae & Kim's CP-TOPTW (2017, *Computers & Operations Research*) — competitive when many side constraints are present.
- **(c) Open:** **branch-cut-and-price *over* the pricing problem**, i.e., solving the ESPPRC in VRPP via its own BCP, exploiting that for VRPP the ESPPRC has a non-trivial polyhedral structure (Da, Zheng & Tang 2017, polyhedral study of ESPPRC).

---

## 4. Cutting Planes and Valid Inequalities

### 4.1 Classical Inequalities Adapted for VRPP

#### Rounded Capacity Inequalities (RCIs)
For CVRP, RCIs are `Σ_{(i,j)∈δ(S)} x_{ij} ≥ 2⌈d(S)/Q⌉` (Lysgaard, Letchford & Eglese 2004; Naddef & Rinaldi 2002). Pessoa, Poggi de Aragão & Uchoa (the "robust BCP" paradigm) showed that RCIs are robust because their dual prices simply modify arc costs in pricing.

For VRPP the *amount of demand inside S that is actually served* is fractional: `Σ_{i∈S} y_i d_i` instead of d(S), where y_i = Σ_r a_{ir} λ_r. A direct adaptation is the **"served-RCI"**:
  Σ_{(i,j)∈δ(S)} x_{ij} ≥ 2⌈(Σ_{i∈S} y_i d_i)/Q⌉,
which is non-linear because of the ceiling on a fractional argument. Two practical linearizations are documented:
- **Lifted Rounded Capacity Inequalities for CPTP**: Jepsen et al. (2014) introduced "rounded multistar inequalities" specific to CPTP, lifting the standard multistar of Letchford, Eglese & Lysgaard (2002) to account for the y_i variables.
- **Conditional RCIs**: write the RCI conditionally on a fixed visit pattern y, for each "candidate" served subset T ⊆ S: `Σ_{δ(T)} x ≥ 2⌈d(T)/Q⌉`. For TOP one uses **min-cut inequalities** (Poggi, Viana & Uchoa 2010) which are unconditional and very effective.

**(b) Mixed:** RCIs work very well for CPTP/CTOP (Jepsen et al. 2014; Bianchessi, Mansini & Speranza 2018 for TOP) but the literature is split on whether the served-RCI is worth the lifting overhead. **(c) Open:** *probabilistic RCIs* — in VRPP, given an LP solution, one can compute the *probability* that a served subset of S violates a capacity bound and aggregate cuts accordingly.

#### Subset-Row Cuts (SRCs)
The seminal SRCs of Jepsen, Petersen, Spoorendonk & Pisinger (2008, *Operations Research* 56) are Chvátal–Gomory rank-1 cuts on triplets (or, for higher rows, k-tuples) of the master partitioning constraints. Pecin et al. (2017a, 2017b) generalized to **limited-memory R1Cs** with arbitrary multipliers (up to 5 rows; see Pecin, Pessoa, Poggi, Uchoa & Santos 2017 for the polyhedral study).

For VRPP, the master constraints are **packing inequalities (≤ 1)** rather than partitioning. The SRCs derivation is altered: starting from `Σ_r a_{ir} λ_r ≤ 1` for i ∈ S with |S|=3, the (1/2, 1/2, 1/2) Chvátal–Gomory rounding gives `Σ_r ⌊(Σ_{i∈S} a_{ir})/2⌋ λ_r ≤ ⌊3/2⌋ = 1`, which is a valid inequality. **The ≤ 1 form yields a standard SRC; no additional weakening is needed.** Park et al. (2017) used 3-row SRCs in CTOPTW. The labelling-side complication (additional resource per SRC) is identical to the partitioning case.

**(c) Open:** **prize-weighted SRCs**. A natural higher-rank cut would weight rows by π_i so that customers with larger prize contribute more to the cut's right-hand side; this is unexplored.

#### k-Path Cuts and Generalized k-Path Cuts
Kohl, Desrosiers, Madsen, Solomon & Soumis (1999) introduced 2-path cuts; Desaulniers, Lessard & Hadjar (2008) generalized to k-path. For VRPP these adapt by replacing the lower bound on number of vehicles entering S (which is ⌈k(S)⌉) with a *served-fraction-aware* lower bound. The generalized k-path inequalities have not been systematically tested for VRPP.

#### Clique and Triangle-Clique Cuts
Pessoa, Poggi & Uchoa (2007) introduced triangle-clique cuts; these were ported to TOP by Poggi, Viana & Uchoa (2010) with strong computational results: combined with min-cut inequalities, they were typically the most effective cuts in their BCP.

#### Cover Inequalities for the Node-Selection Component
Because VRPP has explicit y_i ∈ {0,1} variables (in the explicit master), classical 0-1 cover inequalities apply directly:
  for any cover C ⊆ N s.t. Σ_{i∈C} d_i > Q, `Σ_{i∈C} y_i ≤ |C|−1`.
Lifted cover inequalities are also valid. To our knowledge the use of lifted covers on y_i has not been combined with R1Cs on λ_r in any published VRPP BCP, although it is implemented in the CPTP branch-and-cut of Jepsen et al. (2014).

### 4.2 Inequalities Specific to VRPP Structure

Beyond the literature, the prize structure suggests several genuinely *new* inequality families:

#### Prize-Flow / Budget-Flow Inequalities
Connect the number of vehicles dispatched to the maximum prize they can collect. If at most K vehicles serve customers in S, an aggregate budget-flow valid inequality is
  Σ_{i∈S} π_i y_i ≤ K · max_route p_max(S, T_max, Q),
where p_max(S, ...) is the maximum prize a single route inside S can collect under the duration/capacity budget. p_max can be precomputed by solving a single-vehicle OP. A weaker but separable cut is
  Σ_{i∈S} y_i ≤ K · n_max(S, T_max, Q),
with n_max the maximum number of customers reachable.

#### Co-Visit / Implicit Conflict Inequalities
If customers i and j cannot both be visited by any single feasible route (due to combined demand or duration), then either both routes containing them are different vehicles or one customer is unvisited:
  y_i + y_j ≤ 1 + (Σ_r a_{ir} λ_r and a_{jr} λ_r split appropriately),
or in the explicit form
  y_i + y_j ≤ x^{cross}_{ij} + 1, where x^{cross}_{ij} = 1 if i and j are on different routes.

These are conflict-graph cliques on a customer conflict graph; in the static (capacity-only) case they reduce to classical clique cuts on infeasible pairs.

#### Depot-Degree Inequalities for VRPP
In CVRP the depot has degree 2K. In VRPP, vehicles need not be deployed (an empty solution might be optimal if all prizes are too low), so the depot degree is `Σ_{(0,j)∈A} x_{0j} ≤ K` and `≥ 0`. **Empty-route exclusion**: explicitly require that any used vehicle visits at least one customer of positive prize, which gives `x_{0,n+1} ≤ 0` (forbid the trivial route) — already standard in Boussier et al. (2007).

#### Route Length / Prize Ratio Inequalities (new)
Combining the budget T_max with prize collected on a route, valid for each r ∈ Ω with strictly positive λ_r:
  c_r ≤ T_max ⟹ p_r ≥ p_min(c_r) where p_min is a non-decreasing concave hull constructed from previously enumerated routes. As a global cut, these contribute "prize threshold" cuts of the form
  Σ_r (p_r − ρ c_r) λ_r ≥ 0 for ρ = (min-cost-per-prize among optimal routes),
which are dynamically separable.

### 4.3 New Inequality Ideas to Explore (Open Research Directions)

#### "Prize-Feasibility Cuts"
*Formal statement.* If a set S of customers cannot all be visited by a single route (capacity or duration infeasibility for the subset) and the combined prize Σ_{i∈S} π_i exceeds a threshold P*, then the LP-relaxed solution must split S over at least k(S) ≥ 2 routes:
  Σ_{(i,j) ∈ δ(S)} x_{ij}/2 ≥ k(S) · 1_{Σ_{i∈S} π_i y_i ≥ P*}.
After linearization (using big-M or McCormick), this yields a cut family that strengthens the LP precisely where the prize structure forces vehicle multiplicity.

#### Interdiction-Style Cuts
For a subset S that is "geometrically committing" — i.e., serving any customer in S with a feasible route necessarily uses some arc in a small set A_S — derive a cut of the form
  Σ_{i∈S} y_i ≤ |S| · Σ_{(i,j) ∈ A_S} x_{ij}/c_S,
where c_S is a precomputed connectivity factor. This is a generalization of strengthened comb / multistar to a prize setting.

#### Dual-Guided Cutting
Use the LP dual at the current node to identify customer groups being "split fractionally" across many low-fraction routes. Specifically, for groups G with Σ_r a_{ir} λ_r small but Σ_r λ_r among routes touching G large, generate cuts of the form Σ_{i∈G} y_i ≤ Σ_{r: r∩G≠∅} λ_r. This is an explicit form of the implicit y-x-λ linkage that can become violated under fractionality.

#### Multi-Route Synergy Cuts (MP-VRPP)
If a customer's revisit on day d' carries a modified prize (e.g., π_i' < π_i), the temporal interaction admits cuts like
  Σ_{d} π_i^d y_i^d ≥ aggregate-revenue-bound,
along with monotonic cuts encoding spacing constraints.

#### Temporal Cover Cuts (MP-VRPP)
Cover cuts on the visit-frequency dimension. If `Σ_{d∈D} y_i^d ≤ f_i^max` is required, then for any subset D' ⊆ D with `|D'| > f_i^max`,
  Σ_{d∈D'} y_i^d ≤ f_i^max,
plus lifted versions strengthening with prize differentials per day.

#### Layered Graph Cuts
In a time-/day-expanded graph for MP-VRPP, valid inequalities on the per-layer flow:
  Σ_{(i,j)∈A_d} x_{ij}^d ≤ K_d , Σ_{d∈D} x_{ij}^d ≤ K_total · capacity-fraction,
strengthen the LP at each time slice. These are analogues to Pessoa, Uchoa, Poggi & Rodrigues' (2010) arc-time-indexed inequalities for parallel-machine scheduling.

---

## 5. Branching Strategies

### 5.1 Classical Branching Rules Adapted for VRPP

#### Arc/Edge Branching (`x_{ij}`)
Branching on arc fractionalities is robust (does not break the pricing) but for VRPP it is **systematically weaker** because many arcs have small fractional values when customers are optional. Both Boussier et al. (2007) and Archetti, Bianchessi & Speranza (2013) explicitly note this weakness. They report that pure arc branching leads to deep, unbalanced trees. Pecin et al. (2017b) made strong arc branching the default for CVRP/VRPTW, but in VRPP it must be combined with structural branching to scale.

#### Ryan–Foster Branching (`together / separate`)
Originally introduced by Ryan & Foster (1981) for set partitioning. In its standard form: for two customers i, j with fractional co-visit fraction in (0,1), create child 1 (i and j together on the same route) and child 2 (i and j on different routes). For VRPP, customers may be unvisited, requiring a **trichotomy**:
  Branch 1: i and j together on the same route (forces y_i = y_j = 1 and same route).
  Branch 2: i and j separated (different routes; forces y_i = y_j = 1 with arc constraints).
  Branch 3: at least one of i, j is unvisited.
Pessoa, Sadykov, Uchoa & Vanderbeck (2020) generalized Ryan–Foster via the "packing set" concept for arbitrary VRP variants; their VRPSolver natively supports the trichotomy when the equality master is replaced by ≤ 1.

#### Node-Visit Branching (`y_i = 0 / y_i = 1`)
A *highly natural* and *strong* branching for VRPP: pick a fractional y_i and create two children. Boussier et al. (2007) and Archetti, Bianchessi & Speranza (2013) explicitly recommend this and apply it in their TOP/CTOP B&P. Setting y_i = 0 simply removes customer i from the pricing graph; setting y_i = 1 forces every feasible solution to include i (handled by adding a column-feasibility constraint or by a restricted master with i compulsory).

**Why y-branching is strong:** because it directly tackles the most VRPP-specific decision, it commits a fundamental structural choice early and dramatically tightens the LP at the child node.

#### Vehicle-Count Branching
Branch on `K_used = ⌊Σ_r λ_r⌋` vs. `⌈Σ_r λ_r⌉`. Boussier et al. (2007) use this as one branching dimension. In VRPP, K is an upper bound; the active vehicle count can vary.

### 5.2 New Branching Ideas for VRPP

These are largely unexplored and constitute fertile open ground:

#### Prize-Group Branching
Identify a *clique* C of mutually-compatible high-prize customers (i.e., a set that can plausibly all be visited together). Branch on:
  Branch 1: all of C visited (`Σ_{i∈C} y_i = |C|`),
  Branch 2: at least one of C unvisited (`Σ_{i∈C} y_i ≤ |C|−1`),
or a finer ternary partition. This is a *structural* branching that exploits the prize structure to commit early to high-value subsets.

#### Fractional-Prize Branching
For a fractionally-visited customer i with `Σ_r a_{ir} λ_r = α ∈ (0,1)`, branch on whether the *prize collected* π_i α is above or below π_i/2. While at first sight equivalent to y-branching, threshold-α branching with α ≠ 0.5 can be more balanced (closer to the prize-weighted barycenter).

#### Route-Profit Threshold Branching
Branch on whether any single route in the support of the LP solution achieves profit ≥ some threshold ρ*: specifically `max_r (p_r − c_r) λ_r > ρ*` vs. `≤ ρ*`. Concretely realized by adding a constraint to the master and re-pricing in each child. Useful when the LP solution has very thin fractional routes.

#### Combined Node-Arc Branching
Simultaneously branch on a node y_i = 1 and an incident arc x_{ij} = 1, reducing symmetry and committing the partial structure of the route through i. A 4-way branching: (y_i = 0); (y_i = 1, x_{ij} = 1, x_{ik} = 0 for k ≠ j); (y_i = 1, x_{ij} = 0, x_{ik} = 1); (y_i = 1, x_{ij} = 1, x_{ik} = 1, etc.).

#### Day-Assignment Branching for MP-VRPP
Branch on which day d ∈ D customer i is visited. For y_i^d fractional, child d* fixes y_i^{d*} = 1 and y_i^d = 0 for d ≠ d*. This is the natural "day-by-day" decomposition branching for MP-VRPP and is analogous to the depot branching used by Baldacci & Mingozzi (2009) for the MDVRP.

#### Pseudo-Cost Branching Calibrated for VRPP
Standard MIP pseudo-cost branching estimates score from past LP-bound improvements per unit branching. For VRPP, calibration must distinguish prize-cost contributions: maintain pseudo-costs separately for "prize-loss" (difference in prize collected) and "cost-saving" (difference in cost incurred). This is unexplored. The "two-level branching with branching history" of da Silva & Schouery (2025, *INFORMS J. Comput.*, dynamically tuned 2LBB) for cutting stock is a relevant template.

### 5.3 Search Strategies for the B&B Tree

#### Best-first vs. Depth-first vs. Best-estimate
- **Best-first** is risky for VRPP because the LP relaxation gives loose upper bounds when the prize structure is salient — many open nodes can have similar bounds.
- **Depth-first** (with backtracking) supports diving and limited discrepancy search heuristics (Joncour, Michel, Sadykov, Sverdlov & Vanderbeck 2010; Sadykov, Vanderbeck, Pessoa, Tahiri & Uchoa 2019).
- **Best-estimate** combining LP bound and a learned correction (Achterberg's pseudo-cost) is what modern VRPSolver uses.

#### Diving Strategies Guided by Prize
Standard column-generation diving (Sadykov et al. 2019) selects columns by rounding fractional λ_r. **(c) New idea:** prize-prioritized diving: at each diving step, select the column with maximum p_r λ_r (rounded up) among the fractional columns; this dives toward solutions committing to high-prize routes early.

#### Limited Discrepancy Search and Beam Search
LDS (Joncour et al. 2010) is generic. Beam search has not been formally combined with the BPC tree for VRPP.

#### Warm-Starting at Each B&B Node
Standard practice (Vanderbeck 2005): inherit the parent's column pool and restart column generation with the parent's dual prices. For VRPP, the additional consideration is that y_i = 0 branches *invalidate* every column containing i — these must be filtered out, and the remaining pool may be sparse, requiring more column generation in the child.

---

## 6. Lagrangian Relaxation and Dual Methods

### 6.1 Standard Lagrangian Relaxation

For the master `max Σ (p_r − c_r) λ_r s.t. Σ a_{ir} λ_r ≤ 1, Σ λ_r ≤ K`, dualizing the covering constraints with multipliers π ≥ 0 yields the Lagrangian
  L(π) = Σ_i π_i + max Σ_r (p_r − c_r − Σ_i a_{ir} π_i) λ_r,
which decomposes into the same SPPRC pricing. Subgradient methods of Held & Karp / Polyak with Polyak step size or modified Polyak (Bertsekas, *Nonlinear Programming*) are standard. **Deflection** (d'Antonio & Frangioni 2009) further improves convergence.

### 6.2 Multiplier Interpretation for Optional Nodes

When customer i is unvisited at the LP optimum, its dual variable π_i = 0 (*by complementary slackness* on Σ_r a_{ir} λ_r ≤ 1). This has the interpretation that **the prize π_i_value is "absorbed"** into the decision not to serve, and the dual contributes nothing. Operationally, this means:
- During pricing in VRPP, the modified arc cost into customer i depends only on the prize value (π_i_value) when the dual is zero; this signals to the pricing graph that customer i is "free profit" per unit cost.
- Stabilization via prize-anchored dual estimates (§1.2) is consistent with this interpretation.

### 6.3 Combining Lagrangian Relaxation with Column Generation

**Lagrangian-based pricing** (Mingozzi, Roberti & Toth 2013, Boschetti, Mingozzi & Ricciardelli 2008) uses Lagrangian dual values as estimates of LP duals to feed into the SPPRC. For VRPP this is particularly attractive because the SPP packing structure is well-suited to the dual ascent of Boschetti, Mingozzi & Ricciardelli (2008). The three-stage scheme — H1: q-route Lagrangian bound; H2: column-and-cut with Lagrangian master; H3: classical column-and-cut to close gap — is a paradigm originally established by Baldacci, Christofides & Mingozzi (2008) and adopted (in modified form) by Archetti, Bianchessi & Speranza (2013) for CTOP/CPTP.

### 6.4 Bundle and Proximal Bundle Methods

Briant, Lemaréchal, Meurdesoif, Michel, Perrot & Vanderbeck (2008, *Mathematical Programming*) compared bundle and classical column generation across cutting stock, vehicle routing, lot sizing and TSP, finding bundle convergence to be more stable but per-iteration slower. Frangioni's generalized bundle methods (Frangioni 2002, *SIAM J. Optim.*; Frangioni & Gorgone 2014) are state-of-the-art for nonsmooth dual problems. For VRPP, where dual oscillation is severe, **proximal bundle is theoretically attractive but underexplored**.

### 6.5 Augmented Lagrangian / ADMM for MP-VRPP

ADMM decompositions by period — i.e., separate the multi-period master into one subproblem per day, with an augmented Lagrangian penalty on coupling fleet/customer constraints — have been proposed in inventory routing (Coelho, Cordeau & Laporte 2014) but **not yet applied to MP-VRPP**. This is a clear open direction: the per-day pricing remains an ESPPRC; the periodic coupling (visit-frequency, fleet share) is enforced via quadratic penalties and dual updates. Such an ADMM procedure can serve as a heuristic warm-start within an exact BCP.

### 6.6 Volume Algorithm

The Volume algorithm (Barahona & Anbil 2000) recovers approximate primal solutions from a Lagrangian dual without solving an LP. For VRPP, it provides a quick way to estimate the master LP solution within a Lagrangian-driven scheme; tested in set covering (Boschetti et al. 2008) but not for VRPP.

### 6.7 Dual Ascent for Set-Covering LPs

Mingozzi (2002) and Boschetti, Mingozzi & Ricciardelli (2008) introduced parametric and Lagrangian dual ascent procedures for the set-partitioning LP, which scale to enormous columns implicitly. For VRPP, the natural target is the set-packing LP; the dual ascent must respect non-negativity of dual variables (since the master is ≤). **(c) Open:** a dedicated dual ascent for set-packing LPs with embedded prize structure has not been published.

### 6.8 Surrogate Relaxation

Surrogate-relaxing duration and capacity simultaneously (combined into a single weighted constraint) can simplify the SPPRC pricing into a simpler one-resource constrained shortest path. This was explored by Baldacci, Hadjiconstantinou & Mingozzi (2004) for CVRP via two-commodity flow surrogates. For VRPP it has not been done; given that pricing is computationally dominant, this is a promising acceleration.

---

## 7. Multi-Period VRPP Specific Adaptations

### 7.1 Master Problem Structure

For MP-VRPP with planning horizon D and per-day fleet K_d, the natural master is:
  max Σ_{r,d} (p_r^d − c_r^d) λ_r^d
  s.t. Σ_r a_{ir}^d λ_r^d ≤ y_i^d ∀i, d (per-day visit linking)
     Σ_d y_i^d ≤ f_i^max (max-frequency)
     Σ_d y_i^d ≥ f_i^min (min-frequency, if any)
     Σ_r λ_r^d ≤ K_d ∀d (per-day fleet)
     λ_r^d ∈ {0,1}, y_i^d ∈ {0,1}.

Pirkwieser & Raidl (2009b) and Cacchiani, Hemmelmayr & Tricoire (2014) developed the corresponding column-generation master for the periodic VRPTW; Baldacci, Bartolini, Mingozzi & Roberti (2011a) provided the only exact approach for the pure PVRP, using three relaxations with route enumeration. Archetti, Pillac, Hemmelmayr & Doerner type work on selective PVRP variants is partial. **No published exact BCP exists for MP-VRPP** in its full prize-collecting form — this is the *single biggest open gap* in the literature.

### 7.2 Period-Indexed Pricing

The natural design is one ESPPRC per period d (with period-specific durations, prizes, time windows) sharing dual prices via the y-linking and fleet constraints. Acceleration: solve all D pricings concurrently, share dominance information (a dominated label in day d may dominate in day d' if profit and resource consumption permit).

### 7.3 Cross-Period Cuts

Cuts spanning multiple periods are essential when the prize structure has temporal dependencies:
- **Frequency-cover cuts**: `Σ_{d∈D'} y_i^d ≤ f_i^max` for any subset D' larger than f_i^max.
- **Inter-visit-spacing cuts**: if customer i must have ≥ s days between visits, then `y_i^d + y_i^{d+1} + ... + y_i^{d+s-1} ≤ 1`.
- **Cumulative-prize cuts**: `Σ_{d} π_i^d y_i^d ≤ aggregated upper bound`.
- **Layer-flow cuts** (§4.3) on day-indexed arc variables.

Layered SRCs and limited-memory R1Cs across periods are unexplored.

### 7.4 Temporal Lagrangian Decomposition

Decompose by period d, dual-price the y-linking and fleet constraints; solve each resulting per-period LP (a single-day VRPP) by classical BCP; coordinate via subgradient or bundle. This **temporal Lagrangian decomposition** has not been applied to MP-VRPP exactly, only heuristically (Cacchiani et al. 2014 set-covering matheuristic). It is the natural exact-approach analogue.

### 7.5 Rolling Horizon Inside BCP

Tricoire, Romauch, Doerner & Hartl (2010) used variable neighborhood search heuristically with rolling horizon; integrating rolling-horizon decisions inside an exact BCP — i.e., solving day-by-day with the residual prize structure absorbed into cuts — could provide strong primal solutions and warm-starts.

### 7.6 Prize Dynamics

If π_i^d depends on d (time-dependent prizes), the pricing graph changes per period; the column structure must encode the day index. If π_i depends on previous visits (e.g., a discount on the second visit), **state augmentation** is needed: each label carries a counter of prior visits to i.

### 7.7 Synchronization Constraints

For MP-VRPP with vehicle returns to depot between days and possible inventory replenishment, the master gains additional resource-replenishment constraints. Boland, Christiansen, Nygreen & Sørensen (2014) and Stalhane, Andersson, Christiansen & Fagerholt (2012) develop similar scheduling-routing coupling for periodic ship routing; analogous formulations apply here.

### 7.8 Multi-Day vs. Single-Day Columns

Two design choices in MP-VRPP BCP:
- **Single-day columns** (one λ_r^d per day) coupled by visit-frequency constraints: smaller pricing problem, tighter master (more constraints). This is the design used by Pirkwieser & Raidl (2009b).
- **Multi-day columns** (a column is a vehicle's full schedule across D days): pricing solves a larger ESPPRC over a layered graph; master has fewer constraints. Athanasopoulos & Minis (2013) used this design heuristically.

The single-day-with-coupling design is computationally preferred because it preserves the per-day pricing structure and exposes more cutting-plane families. **(c) Open:** no exact comparison of the two paradigms exists in the prize-collecting setting.

---

## 8. Synthesis: What is Established, Mixed, or Open

### Established (a):

- ESPPRC with bidirectional labeling is the dominant pricing scheme (Feillet et al. 2004; Righini & Salani 2006, 2008; Tilk et al. 2017).
- ng-routes (Baldacci, Mingozzi & Roberti 2011) and DSSR (Boland, Dethridge & Dumitrescu 2006; Righini & Salani 2008) are the standard relaxations; combination ng-DSSR is the workhorse.
- Limited-memory rank-1 cuts (Pecin et al. 2017a, 2017b) work in the packing master with no modification.
- Y-branching (node visit) and arc branching are both used; y-branching is strong for VRPP specifically.
- Dual price smoothing with auto-regulation (Pessoa et al. 2018) is the recommended stabilization.
- For TOP/CTOP, BCP solves up to ~100-customer benchmarks (Boussier et al. 2007; Poggi et al. 2010; Keshtkaran et al. 2016; Bianchessi, Mansini & Speranza 2018).
- For CPTP, the Jepsen et al. (2014) branch-and-cut with rounded multistar inequalities scales to 800 nodes — a 4× advantage over column-generation-based methods of comparable era.

### Mixed Results (b):

- Bundle methods are theoretically superior for stabilization but per-iteration costly; smoothing has emerged as the practical winner.
- Pulse vs. labelling in pricing: Pulse outperforms labelling on stand-alone OPs (Lozano et al. 2016) but gives mixed results inside CG due to limited dominance.
- Triangle-clique cuts (Pessoa et al. 2007) help in TOP (Poggi et al. 2010) but their interaction with R1Cs is not systematically studied for VRPP.
- ng-route vs. fully elementary: the prize-doubling effect makes the deterioration of ng less acceptable than for CVRP, but no consensus on when to favour elementary.

### Open / Promising (c):

1. **Prize-anchored dual stabilization**: initialize the smoothing centre at π_i_value (the prize as natural dual estimate). Predicted to strongly accelerate VRPP column generation.
2. **Prize-aware ng-route neighborhoods**: select N_i by combination of distance and prize π_j, focusing elementarity precisely on high-payoff cycles.
3. **Prize-weighted DSSR triggers**: insert into Θ the customer maximizing π_i × overvisit count.
4. **Prize-Lagrangian completion bounds**: Lagrangize duration and use the resulting penalty-prize as completion bound.
5. **Prize-feasibility cuts**: cuts forcing extra vehicles when an infeasible high-prize subset is fractional.
6. **Dual-guided cutting**: targeted cuts on customer groups split fractionally.
7. **Prize-group branching**: clique-based branching on whether high-prize subsets are jointly visited.
8. **Combined node-arc branching**: simultaneous y_i and x_{ij} branching to reduce VRPP symmetry.
9. **Prize-prioritized diving**: dive on columns with maximum p_r λ_r.
10. **ADMM by period for MP-VRPP**: temporal Lagrangian decomposition with augmented Lagrangian, embedded in BCP as warm-start heuristic.
11. **Layered graph cuts and temporal cover cuts for MP-VRPP**: limited-memory R1Cs across periods.
12. **First exact BCP for the full MP-VRPP**: this is genuinely missing from the literature. Single-day columns coupled by visit-frequency constraints, with cross-period cuts and a temporal Lagrangian master, would constitute a major advance.
13. **Extended-formulation reformulations for VRPP**: pseudo-polynomial (Letchford & Salazar-González 2019) and arc-time-indexed formulations have given strong bounds for CVRP; their VRPP analogues, especially with prize variables, are unexplored.
14. **Machine-learning-supported branching**: 2LBB (da Silva & Schouery 2025) for VRPP, calibrating pseudo-costs separately for prize-loss and cost-saving.
15. **Surrogate Lagrangian relaxations** of duration + capacity into one combined resource for VRPP, simplifying pricing.
16. **Constraint-programming-based pricing** for very-tight-time-window VRPP variants (TOPTW, MC-TOP-MTW), per Tae & Kim (2017).

---

## 9. Concluding Remarks

The Vehicle Routing Problem with Profits sits at the intersection of two algorithmic paradigms: classical Branch-Cut-and-Price for VRPs (where the master is partitioning and the pricing is ESPPRC) and prize-collecting / orienteering algorithms (where node selection is the dominant decision). The literature has successfully imported core BPC tools — ng-routes, DSSR, limited-memory R1Cs, smoothing stabilization, hierarchical strong branching — into TOP, CTOP, CPTP and the OPTW. State-of-the-art exact algorithms now solve TOP benchmarks with up to ~100 customers and CPTP up to 800 nodes, but lag behind the 400+ customer frontier of CVRP.

The principal opportunities for advancing VRPP BCP/BPC lie in:

(i) **Exploiting the prize structure as a-priori dual information** — through prize-anchored stabilization centres, prize-weighted DSSR/ng-relaxations, and prize-Lagrangian completion bounds. The prize values are *natural* dual estimates that the literature has so far underexploited.

(ii) **Developing genuinely VRPP-specific cuts and branching rules** — prize-feasibility cuts, prize-group branching, threshold-based fractional-prize branching — that exploit the optional-customer structure rather than just adapting partitioning-based tools.

(iii) **Closing the MP-VRPP gap** — no exact BCP exists for the multi-period prize-collecting VRP. A single-day column formulation with layered graph cuts, temporal cover cuts, and a Lagrangian-decomposed master coupled by visit-frequency constraints, embedded in the modern VRPSolver (Pessoa et al. 2020) packing-set framework, would constitute a substantial advance.

(iv) **Integrating modern BCP innovations** — bucket-graph labelling (Sadykov, Uchoa & Pessoa 2021), generalized arc-memory R1Cs (Pecin et al. 2017a), automatic stabilization (Pessoa et al. 2018), enumeration-based pricing — uniformly across VRPP variants.

The methodological building blocks are mature. What remains is the careful combination, calibration, and computational validation of these blocks specifically for the prize-collecting setting, and the formal extension into the multi-period domain.
