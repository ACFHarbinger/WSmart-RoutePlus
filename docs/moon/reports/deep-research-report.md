# Branch-Price-and-Cut: Theory, Origins, Components, and a Research Blueprint for Capacitated Profitable Tour Problems with Time Windows

## Executive synthesis and terminology

**Branch-Price-and-Cut (BPC)**—also written **Branch-and-Price-and-Cut**, **Branch-Cut-and-Price**, or **Branch-Cut-and-Price (BCP)**—is best understood not as one algorithm but as an exact optimization architecture combining three ideas:

\[
\boxed{\text{Branch-and-Bound}+\text{Column Generation}+\text{Cutting Planes}}
\]

The names “branch-price-and-cut” and “branch-cut-and-price” normally describe the same fundamental machinery; different papers emphasize a different ordering of operations. Fukasawa et al.'s influential CVRP algorithm uses the term *robust branch-and-cut-and-price*, whereas Jepsen et al. describe a *branch-and-cut-and-price* algorithm for the VRPTW, and more recent routing papers commonly use BPC or BCP interchangeably. citeturn15search0turn15search2turn22search23

At a high level:

\[
\text{BPC}
=
\underbrace{\text{tree search}}_{\text{branch}}
+
\underbrace{\text{dynamic variables}}_{\text{price}}
+
\underbrace{\text{dynamic constraints}}_{\text{cut}}.
\]

That compact definition hides most of the difficulty. A serious BPC solver also needs a strong Dantzig–Wolfe reformulation, an efficient resource-constrained shortest-path engine, dual stabilization, feasibility/Farkas pricing, specialized dominance rules, route relaxations such as \(ng\)-routes, a separation engine for robust and nonrobust inequalities, pricing-compatible branching rules, primal heuristics, variable fixing, preprocessing, numerical safeguards, and careful coordination of all these components. The success of generic frameworks such as VRPSolver demonstrates how mature this architecture has become for vehicle-routing problems, while also showing that the difficult part is no longer merely “doing column generation”: it is orchestrating all of these ingredients efficiently. citeturn19search0turn22search4

For your intended problem, there is one **very important modeling distinction**.

The classical **Profitable Tour Problem (PTP)** chooses a subset of vertices and a tour so that collected profit is traded against routing cost. Dell'Amico, Maffioli, and Värbrand introduced the name in 1995 in a prize-collecting-tour context. citeturn16search1 The **Capacitated Profitable Tour Problem (CPTP)** adds capacity restrictions; Archetti et al. studied the CPTP together with the capacitated team-orienteering problem. citeturn16search2 There is also a direct branch-and-cut treatment of the CPTP by Jepsen et al. citeturn16search3 Sun et al. later considered a particularly close relative of your proposed target: a **time-dependent capacitated profitable tour problem with time windows and precedence constraints**. citeturn22search0

A classical single-vehicle CPTP is therefore conceptually different from a classical VRP:

\[
\text{VRP: choose a collection of routes serving all required customers}
\]

versus

\[
\text{PTP: choose which customers are worth serving and how to connect them in a tour}.
\]

This matters because **route-based BPC is most naturally powerful when many complete routes must be chosen from an exponentially large set**. If your problem literally contains one vehicle and exactly one tour, a direct branch-and-cut or a specialized elementary resource-constrained path/tour algorithm may be simpler. If instead you have several vehicles, several tours, multi-trip operation, shifts, route-duration limits, heterogeneous resources, or assignment decisions, then BPC becomes exceptionally natural. The route-based decomposition can simultaneously decide **which profitable customers to serve** and **which feasible routes should serve them**. Exact BPC has been successful precisely on rich routing models in which the compact arc formulation becomes weak or unwieldy. citeturn17search11turn19search0turn21search19

For the remainder of this report I therefore treat your proposed target as a generalized **CPTP-TW-TC**:

\[
\text{Capacitated Profitable Tour/Routing Problem with Time Windows and Time Constraints},
\]

where customers may be optional, each carries a profit, vehicles have finite capacity, service must satisfy time windows, and routes may additionally face a maximum duration, working-time limit, time-dependent travel, precedence requirements, or related temporal constraints. I distinguish throughout between the **single-tour** and **multi-route** cases.

The central conclusion of the literature is:

> A promising CPTP-TW BPC should not simply copy a VRPTW BPC and add customer profits. Its most interesting research opportunity is to make **customer selection, route pricing, temporal feasibility, cuts, and branching interact explicitly**.

In particular, I would build the new method around five ideas: a **selection-aware set-packing/set-partitioning master**, **profit-aware ESPPRC pricing with adaptive \(ng\)-route/DSSR relaxations**, a **two-tier robust/nonrobust cut system**, **selection-first followed by cluster/arc branching**, and a **pricing-for-integrality/primal-search mechanism** designed for optional customers. Recent developments in cluster branching, learning-to-branch, pricing for integrality, resource-window reduction, and learned column prefilling make this a particularly promising direction in 2026. citeturn17search1turn22search7turn17search3turn15search23turn22academia26


## Historical lineage from branch-and-bound to branch-price-and-cut

BPC is easiest to understand chronologically because virtually every one of its modern components arose as a response to a specific weakness of an earlier exact method.

| Period | Development | Why it matters for BPC |
|---|---|---|
| 1958–1960 | Gomory cutting planes; Land–Doig branch-and-bound; Dantzig–Wolfe decomposition | Established the three intellectual roots: dynamic constraints, tree search, and decomposition |
| 1961–1963 | Gilmore–Gomory cutting stock; Little et al. TSP B&B | Demonstrated practical generation of missing columns and combinatorial branching |
| 1980s–1990s | Polyhedral routing, Lagrangian routing methods, route-based formulations | Developed stronger bounds and resource-constrained path subproblems |
| late 1990s | Modern branch-and-price and VRPTW cuts | Column generation became integrated with integer search |
| 2000s | Robust BCP, ESPPRC algorithms, subset-row cuts | BPC became a state-of-the-art exact routing paradigm |
| 2010s | \(ng\)-routes, limited-memory cuts, stronger pricing, generic solvers | Large improvement in pricing/cutting efficiency |
| 2020s | Bucket graphs, reduced-cost resource windows, enumeration, learning, cluster branching | Focus shifted toward component interaction, integer solutions, and adaptive algorithms |

**Branch-and-Bound.** The historical starting point is the observation that an integer optimization problem can be recursively partitioned into smaller subproblems while using relaxations to avoid enumerating everything. Land and Doig's 1960 work is generally credited with the first LP-based branch-and-bound procedure for general mixed-integer programming; Little, Murty, Sweeney, and Karel's 1963 traveling-salesman algorithm was an early and highly influential combinatorial branch-and-bound method. citeturn20search1turn20search8

Suppose we solve a maximization problem. At a node \(N\), let

\[
UB(N)=\text{optimal value of a relaxation of }N
\]

and let

\[
LB=\text{value of the best known feasible integer solution}.
\]

Then:

* if the relaxation is infeasible, \(N\) can be discarded;
* if \(UB(N)\le LB\), \(N\) cannot improve the incumbent and can be discarded;
* if the relaxed optimum is feasible for the original integer problem, it becomes an incumbent candidate;
* otherwise a fractional/disjunctive decision is selected and the node is partitioned into children.

For minimization, the interpretation of the bounds is reversed: LP relaxation values are lower bounds and incumbents provide upper bounds.

This simple mechanism is the exactness skeleton of every BPC algorithm. Pricing and cutting do **not replace** branch-and-bound; they make its node bounds much stronger or much cheaper to obtain.

One can view classical B&B as:

\[
\boxed{
\text{relax} \rightarrow
\text{bound} \rightarrow
\text{prune or branch} \rightarrow
\text{repeat}
}
\]

The fundamental danger is a weak relaxation. If the LP solution is far from the integer optimum, the tree becomes enormous.

**Cutting planes and Branch-and-Cut.** Cutting-plane methods attack precisely that weakness. A valid inequality

\[
\alpha^\top x \le \beta
\]

is added when it is satisfied by every feasible integer solution but violated by the current fractional relaxation solution.

Rather than explicitly include potentially millions of inequalities in the original model, a branch-and-cut algorithm searches for violated inequalities dynamically. Modern CVRP branch-and-cut work illustrates how rich such separation can become: Lysgaard, Letchford, and Eglese used capacity, framed-capacity, generalized-capacity, strengthened-comb, multistar, partial-multistar, extended-hypotour, and Gomory mixed-integer cuts in one algorithm. citeturn15search7 Multistar and partial-multistar inequalities themselves were developed specifically as polyhedral strengthening devices for capacitated routing. citeturn15search3

Conceptually:

\[
\boxed{
\text{Branch-and-Cut}
=
\text{B\&B}
+
\text{separate violated rows dynamically}.
}
\]

Cuts reduce the **vertical weakness** of the formulation: the model has the right variables but too much fractional freedom.

**Dantzig–Wolfe decomposition and Column Generation.** A different problem arises when a strong model naturally has an astronomical number of variables. Dantzig and Wolfe's 1960 decomposition principle showed how a large structured LP could be represented through extreme points/rays of subproblems coordinated by a master problem. citeturn20search6 Gilmore and Gomory's cutting-stock work then demonstrated the computational power of solving an LP without explicitly listing its immense collection of pattern variables: only promising columns were generated when required. citeturn20search3turn20search11

If a master LP is

\[
\min \sum_{r\in\mathcal R} c_r\lambda_r
\]

with an exponentially large set \(\mathcal R\), the Restricted Master Problem (RMP) contains only

\[
\mathcal R'\subset\mathcal R.
\]

After solving the RMP, its dual variables reveal whether some omitted column could improve the objective. The subproblem searching for that column is the **pricing problem**.

For a generic master

\[
\begin{aligned}
\min \;& \sum_r c_r\lambda_r\\
\text{s.t. }&
\sum_r a_{ir}\lambda_r = 1 && i\in V,\\
&\lambda_r\ge0,
\end{aligned}
\]

with dual \(\pi_i\), the reduced cost of route \(r\) is

\[
\bar c_r=c_r-\sum_i\pi_i a_{ir}.
\]

Pricing solves

\[
\min_{r\in\mathcal R}\bar c_r.
\]

If

\[
\min_r\bar c_r\ge0,
\]

the current restricted solution is optimal for the full LP.

For vehicle routing, \(r\) is a complete feasible vehicle route, so the column itself implicitly enforces complicated combinations of capacity, time, sequence, and resource restrictions. That is why route-based Dantzig–Wolfe formulations can be much stronger than straightforward compact arc formulations. Vanderbeck formalized this integer-programming perspective on Dantzig–Wolfe decomposition and the associated branching problem. citeturn13search4

**Branch-and-Price.** Column generation alone solves an LP relaxation. To enforce integrality it must be embedded in B&B:

\[
\boxed{
\text{Branch-and-Price}
=
\text{B\&B}
+
\text{column generation at B\&B nodes}.
}
\]

Barnhart, Johnson, Nemhauser, Savelsbergh, and Vance's 1998 paper became the canonical modern synthesis of this approach for huge integer programs. citeturn13search13

The key complication is subtle: after branching, the pricing problem has to obey the branch decisions. Branching directly on a route variable such as

\[
\lambda_r=0 \quad\text{or}\quad \lambda_r=1
\]

is usually unattractive because pricing may later regenerate an equivalent route. Instead, successful branch-and-price methods branch on **structural properties**—arcs, customer pairs, cutsets, assignments, or other attributes that can be enforced within pricing.

**Branch-Price-and-Cut.** BPC arises because route-based models have both problems simultaneously:

1. too many route variables to enumerate;
2. an LP relaxation that can still be too fractional.

The resulting node solution process is therefore approximately

\[
\boxed{
\text{solve RMP}
\leftrightarrow
\text{price columns}
\leftrightarrow
\text{separate cuts}
}
\]

followed by branching only after the node bound is sufficiently certified.

Fukasawa et al.'s 2006 **Robust Branch-and-Cut-and-Price for the Capacitated Vehicle Routing Problem** was a landmark because it combined route-based pricing with strong CVRP cutting planes while carefully controlling the effect of those cuts on the pricing problem. citeturn15search0 Jepsen et al. subsequently introduced **subset-row inequalities** into a BPC for VRPTW, explicitly using a set-partitioning master and an elementary shortest-path problem with resource constraints as pricing. citeturn15search2 Baldacci, Mingozzi, and Roberti's 2011 work introduced the influential **\(ng\)-route relaxation** and new pricing strategies for CVRP and VRPTW. citeturn21search0turn22search22 Pecin et al. later sharpened limited-memory subset-row inequalities for VRPTW BPC. citeturn22search23

Thus the historical trajectory can be summarized as

\[
\begin{aligned}
\text{B\&B} &\rightarrow \text{manage integrality},\\
\text{cuts} &\rightarrow \text{strengthen weak bounds},\\
\text{Dantzig–Wolfe} &\rightarrow \text{strong structured formulation},\\
\text{column generation} &\rightarrow \text{handle exponentially many variables},\\
\text{branch-and-price} &\rightarrow \text{make CG exact for integer problems},\\
\text{BPC} &\rightarrow \text{simultaneously generate variables and constraints}.
\end{aligned}
\]

That is why BPC is not merely “B&B plus two tricks.” It is the convergence of several decades of integer-programming research.


## Mathematical anatomy of a BPC solver

A route-based formulation makes the architecture particularly transparent.

Let

\[
V=\{1,\dots,n\}
\]

be customers and let \(\mathcal R\) denote the set of every feasible route with respect to vehicle capacity, time windows, route duration, precedence constraints, and any other route-local restrictions.

For a standard VRP define

\[
a_{ir}=
\begin{cases}
1 & \text{if route }r\text{ serves customer }i,\\
0 & \text{otherwise.}
\end{cases}
\]

A classical route master is

\[
\begin{aligned}
\min \quad & \sum_{r\in\mathcal R}c_r\lambda_r\\
\text{s.t.}\quad
&\sum_{r\in\mathcal R}a_{ir}\lambda_r=1
&&\forall i\in V,\\
&\sum_{r\in\mathcal R}\lambda_r\le K,\\
&\lambda_r\in\{0,1\}.
\end{aligned}
\]

The first rows are set-partitioning constraints and the second limits the fleet. This decomposition is central to exact VRP algorithms; Jepsen et al.'s VRPTW BPC explicitly identifies a set-partitioning master with an ESPPRC pricing subproblem. citeturn15search2

For a **profitable routing problem**, however, customers are optional. A particularly useful model introduces selection variables

\[
y_i=
\begin{cases}
1 & \text{if customer }i\text{ is served},\\
0 & \text{otherwise}.
\end{cases}
\]

With customer profit \(p_i\),

\[
\begin{aligned}
\max\quad
&\sum_i p_i y_i-\sum_r c_r\lambda_r\\
\text{s.t.}\quad
&\sum_r a_{ir}\lambda_r=y_i &&\forall i,\\
&\sum_r\lambda_r\le K,\\
&y_i\in\{0,1\},\quad\lambda_r\in\{0,1\}.
\end{aligned}
\tag{MP}
\]

An equivalent formulation absorbs the customer profits into each route,

\[
P_r=\sum_i p_i a_{ir}-c_r,
\]

and uses the set-packing model

\[
\begin{aligned}
\max\quad &\sum_rP_r\lambda_r\\
\text{s.t.}\quad
&\sum_ra_{ir}\lambda_r\le1 &&\forall i,\\
&\sum_r\lambda_r\le K,\\
&\lambda_r\in\{0,1\}.
\end{aligned}
\tag{SP}
\]

This model matches the basic CPTP objective—balancing collected profits and routing costs—studied in the profitable-tour literature. citeturn16search1turn16search2

For your problem I prefer the explicit \(y_i\) version because customer selection then becomes available to cutting, branching, preprocessing, and business-side constraints.

**Reduced-cost pricing.** Assume the set-packing maximization formulation for simplicity. Let \(u_i\ge0\) be customer-row duals and \(v\ge0\) the fleet dual. Ignoring extra cuts momentarily, route \(r\) has reduced profit

\[
\bar P_r
=
P_r-\sum_i u_i a_{ir}-v.
\]

Pricing seeks

\[
\max_{r\in\mathcal R}\bar P_r.
\]

A column is improving when

\[
\bar P_r>0.
\]

The more common minimization convention says exactly the same thing with signs reversed and searches for a column with negative reduced cost.

Because

\[
P_r=\sum_{i\in r}p_i-c_r,
\]

pricing can be interpreted as finding a feasible route maximizing an **adjusted customer reward**

\[
p_i-u_i
\]

minus routing cost and other dual charges.

This is one reason profitable routing fits naturally into route pricing: choosing whether to visit a customer is already part of a resource-constrained path problem.

With active cuts \(h\in H\), having route coefficient \(\alpha_{hr}\) and corresponding dual \(\mu_h\),

\[
\bar P_r
=
P_r
-\sum_i u_i a_{ir}
-v
-\sum_{h\in H}\mu_h\alpha_{hr},
\]

subject to the sign convention of each master row. Thus every newly inserted cut can alter the pricing problem. That interaction is one of the defining difficulties of BPC.

**Pricing as an ESPPRC.** In VRPTW-type problems the pricing problem is usually an **Elementary Shortest Path Problem with Resource Constraints (ESPPRC)** or its profitable/longest-path sign equivalent. Jepsen et al. use precisely this decomposition; recent complexity work confirms that pricing in set-partitioning VRP formulations is itself strongly NP-hard for important routing classes. citeturn15search2turn21search11

A label for your CPTP-TW pricing algorithm might contain

\[
L=(i,q,t,d,\mathcal M,\mathbf z,\rho),
\]

where

- \(i\) is the current vertex;
- \(q\) is load;
- \(t\) is service/start time;
- \(d\) is elapsed route duration;
- \(\mathcal M\) stores elementarity or \(ng\)-memory;
- \(\mathbf z\) stores states required by nonrobust cuts or side constraints;
- \(\rho\) is reduced cost/reduced profit.

If the current customer is \(i\) and we extend to \(j\),

\[
q'=q+d_j\le Q,
\]

and with service time \(s_i\) and travel time \(\tau_{ij}\),

\[
t'
=
\max\{e_j,\;t+s_i+\tau_{ij}\},
\]

with feasibility condition

\[
t'\le l_j.
\]

For time-dependent travel,

\[
t'
=
\max\{e_j,\;t+s_i+\tau_{ij}(t+s_i)\}.
\]

A maximum route/working duration \(H\) adds

\[
t_{\text{return}}-t_{\text{departure}}\le H.
\]

Sun et al.'s time-dependent CPTP with time windows and precedence constraints is especially relevant because it shows that profitable customer selection, capacity, temporally dependent travel, and TW/precedence structure can coexist within one exact routing model. citeturn22search0

The pricing engine should use **dominance** aggressively. At the same endpoint, a label \(L^1\) can eliminate \(L^2\) if the first has resources and reduced cost at least as favorable, with a memory/state relationship guaranteeing that every feasible completion of \(L^2\) can also be constructed from \(L^1\). The hard part is that elementarity and nonrobust cuts enlarge the state space, making dominance weaker and label counts explode. Modern bucket-graph labeling was developed precisely to reduce dominance-comparison overhead in RCSP pricing for state-of-the-art BPC. citeturn17search16

**Farkas pricing.** Ordinary pricing assumes the RMP is feasible and searches for an objective-improving column. But branching or column deletion can create a different situation:

\[
\text{RMP infeasible}
\]

while

\[
\text{full master potentially feasible}.
\]

Declaring the B&B node infeasible at this point would be wrong: perhaps the omitted route needed to satisfy a new branch constraint has simply not been generated yet.

By Farkas' lemma, the LP solver can return a dual ray certifying infeasibility of the **restricted** master. Farkas pricing searches for a missing column that violates that certificate and may restore feasibility. SCIP's pricer interface explicitly distinguishes reduced-cost pricing, which creates objective-improving variables after a feasible LP solve, from Farkas pricing, which generates columns capable of reducing LP infeasibility after an infeasible solve. citeturn18search2turn18search14

This distinction is fundamental:

\[
\boxed{
\begin{array}{ll}
\text{ordinary pricing:} & \text{Can an omitted column improve the objective?}\\[2mm]
\text{Farkas pricing:} & \text{Can an omitted column restore feasibility?}
\end{array}}
\]

In implementation, objective costs effectively cease to be the primary criterion during the Farkas phase. The pricing objective comes from the Farkas multipliers. If no such column exists, infeasibility has been proved for the full master node rather than merely for its current restricted representation. SCIP exposes the Farkas values separately for precisely this reason. citeturn18search2turn18search14

For a profitable-tour BPC this becomes especially relevant after branching on

\[
y_i=1.
\]

The child requires customer \(i\) to be served, yet the inherited route pool might contain no route serving \(i\). The RMP appears infeasible, but Farkas pricing can construct the necessary route.

**Exactness conditions.** Four conditions are non-negotiable for a BPC solver that claims a proof of optimality.

First, before pruning a node on its bound, pricing must certify that no omitted improving column remains. A heuristic pricing algorithm may be used first, but a failure of heuristic pricing is not an optimality certificate.

Second, all inserted cuts must be globally or locally valid for the intended feasible set. Failure to find a violated cut weakens the bound but does not invalidate it; inserting an invalid cut can destroy exactness.

Third, branching disjunctions must cover the complete integer feasible set and be respected by future pricing.

Fourth, any relaxed route family—\(q\)-routes, \(ng\)-routes, partial elementarity, etc.—must be handled in a way consistent with the original problem. In a maximization CPTP, allowing infeasible non-elementary routes enlarges the relaxation and therefore may produce a valid **upper bound**, but such routes cannot simply be accepted as an integer feasible solution. Exactness eventually requires certification against the true route space or a mechanism that eliminates all offending relaxed routes. The 2011 \(ng\)-route work deliberately treats such routes as relaxations used to obtain tight bounds efficiently. citeturn21search8turn22search22


## Pricing, route relaxations, cutting planes, and branching in depth

The most important practical insight is that the components of a BPC cannot be designed independently.

A very strong cut is not necessarily useful if it makes pricing ten times slower. A very tight pricing model is not necessarily useful if millions of labels are needed. A strong branching rule is not useful if every branch adds a constraint that destroys the structure of the pricing problem.

The correct metric is approximately

\[
\frac{\text{bound/integrality improvement}}
     {\text{total induced computational burden}}.
\]

That viewpoint explains most of the development of modern routing BPC.

**Elementary versus relaxed route pricing.** Complete elementarity requires

\[
\text{each customer is visited at most once}.
\]

Keeping a visited-customer bitset in every label yields strong routes but greatly reduces dominance. Feillet et al. developed an exact dynamic-programming approach for the elementary resource-constrained shortest-path problem, which became a basic building block of exact route pricing. citeturn21search2

A sequence of progressively weaker alternatives emerged because exact ESPPRC pricing can dominate total solution time.

A **\(q\)-route relaxation** permits customer repetition while accumulating capacity consumption. It is much easier than full elementarity but can admit unrealistic cyclic routes, producing weak master bounds.

**\(k\)-cycle elimination** prevents short cycles while allowing longer repeated-node structures.

**Partial elementarity** enforces elementarity only on a selected set of nodes. Desaulniers, Lessard, and Hadjar showed how partial elementarity and heuristic ESPPRC pricing could substantially accelerate VRPTW algorithms, alongside generalized \(k\)-path inequalities. citeturn21search24

**Decremental State-Space Relaxation (DSSR)** begins with a weak state space and progressively adds elementarity information when repeated customers are detected. In other words, rather than carrying the full visited set immediately, it discovers which memory is actually needed. Later work continues to describe DSSR as an iterative relaxation in which nodes are progressively added to the elementarity state until cyclic solutions disappear. citeturn21search1turn21search5

The **\(ng\)-route relaxation** of Baldacci, Mingozzi, and Roberti provides one of the most important compromises. Each customer \(i\) is given a local neighborhood

\[
NG_i\subseteq V.
\]

Instead of remembering every previously visited customer, a label remembers only customers that remain relevant through these neighborhoods. Informally, a repeated visit is forbidden when the earlier occurrence has remained in the path's local \(ng\)-memory; repetitions can become possible after the customer has been “forgotten.” Consequently,

\[
\text{weak non-elementary routes}
\;\subsetneq\;
\text{\(ng\)-routes}
\;\subsetneq\;
\text{elementary routes}
\]

in strength, while pricing usually remains dramatically cheaper than full elementarity. Baldacci et al. introduced this relaxation specifically to obtain tight lower bounds for CVRP/VRPTW route formulations without paying the full elementary pricing cost. citeturn21search0turn21search8

The neighborhood size is therefore a fundamental algorithmic parameter:

\[
|NG_i|\uparrow
\Longrightarrow
\begin{cases}
\text{stronger relaxation},\\
\text{more label state},\\
\text{weaker dominance},\\
\text{more pricing time}.
\end{cases}
\]

For your CPTP-TW, I would not choose neighborhoods solely by Euclidean distance. A more interesting **profit/time-aware \(ng\)-neighborhood** would combine:

\[
\text{spatial closeness}
+
\text{TW compatibility}
+
\text{profit interaction}
+
\text{dual co-activity}.
\]

That is a genuine research opportunity discussed later.

**Lagrangian relaxation.** Lagrangian relaxation comes from a different direction but is closely connected to decomposition.

For a minimization problem

\[
\begin{aligned}
\min\quad & c^\top x\\
\text{s.t.}\quad &Ax=b,\\
&x\in X,
\end{aligned}
\]

relax the difficult linking constraints \(Ax=b\) with multipliers \(u\):

\[
L(u)
=
\min_{x\in X}
\left\{
c^\top x+u^\top(Ax-b)
\right\}.
\]

For every \(u\),

\[
L(u)\le z^\star
\]

for the minimization problem, so

\[
\max_uL(u)
\]

gives the strongest bound obtainable from this Lagrangian relaxation.

For maximization, the bound direction reverses.

Routing algorithms used Lagrangian relaxation well before today's BPC implementations. Desrosiers, Sauvé, and Soumis developed Lagrangian methods for minimum-fleet multiple TSP with time windows. citeturn19search13 Kohl and Madsen's VRPTW algorithm relaxed customer-service constraints, optimized the multipliers using subgradient/bundle ideas, and obtained shortest-path-with-time-window-and-capacity subproblems. citeturn19search3 Fisher, Jörnsten, and Madsen likewise investigated a Lagrangian decomposition producing semi-assignment and resource-constrained shortest-path subproblems. citeturn19search8

The deep connection is that **Dantzig–Wolfe master dual variables play the role of prices on the same linking structure that a Lagrangian approach dualizes**. This suggests several uses in a new CPTP algorithm: dual initialization, pricing bounds, reduced-cost fixing, approximate dual ascent before exact column generation, and bounding routes that cannot become attractive.

Recent work on **DeLuxing—Deep Lagrangian Underestimate Fixing** shows that Lagrangian information remains an active avenue for accelerating modern column-generation algorithms through fixing decisions. citeturn17search23turn22search24

**Reduced-cost resource-window reduction.** Temporal pricing admits another powerful form of relaxation/tightening. Instead of merely using original time windows

\[
[e_i,l_i],
\]

one can derive stronger forward/backward windows and eliminate arcs or states using the incumbent gap and reduced costs. Recent work shows that arc-specific directional resource-window reduction can materially accelerate path-based BPC for VRPTW. citeturn15search23

For a profitable-tour problem, this mechanism is particularly attractive because a bound on the **remaining attainable profit** can be combined with remaining time:

\[
\text{current reduced profit}
+
\text{maximum achievable future reduced profit}
\le \text{pricing threshold}
\]

implies that the label can safely be eliminated.

That suggests extending ordinary RCSP dominance into **profit-aware completion bounding**.

**Robust versus nonrobust cuts.** This is arguably the central concept linking pricing and cutting.

Suppose cut \(h\) has master coefficient \(\alpha_{hr}\) for route \(r\). If

\[
\alpha_{hr}
=
\sum_{(i,j)\in r}\gamma^h_{ij},
\]

then its dual multiplier merely changes the arc costs used during pricing:

\[
\tilde c_{ij}
=
c_{ij}
+
\sum_h\mu_h\gamma^h_{ij}.
\]

No extra route history needs to be remembered. Such a cut is **robust** with respect to the chosen pricing representation.

In contrast, if \(\alpha_{hr}\) depends on a global property such as

\[
|r\cap S|,
\]

whether a particular combination of customers has already appeared, or another path-history property, pricing must carry additional state. That cut is **nonrobust** in the algorithmic sense.

The “robust” terminology in Fukasawa-style BCP emphasizes precisely the desire to retain controlled—typically pseudo-polynomial—pricing complexity while strengthening the master. citeturn15search0turn15search4

This produces a critical practical principle:

\[
\boxed{
\text{cut strength alone is not the objective;}
\quad
\text{cut strength per unit of pricing complexity is.}
}
\]

**Rounded capacity and related capacity inequalities.** Capacity inequalities are among the foundational polyhedral devices in CVRP algorithms. For a mandatory-service CVRP and customer subset \(S\), the classic rounded-capacity idea says that demand

\[
d(S)=\sum_{i\in S}d_i
\]

requires at least

\[
\left\lceil\frac{d(S)}{Q}\right\rceil
\]

vehicles to cross the boundary of \(S\), producing corresponding edge-cut inequalities. Modern work still describes rounded capacity inequalities as fundamental VRP cuts. citeturn15search7turn3search5

For the PTP, however, **you cannot simply copy this inequality unchanged**, because some customers in \(S\) may legitimately be unselected. Capacity cuts must be conditioned or lifted using \(y_i\) variables or route coverage. This is one of the main mathematical differences between a VRPTW BPC and a CPTP-TW BPC.

**Multistar and partial-multistar inequalities.** Multistar inequalities strengthen CVRP capacity/connectivity reasoning by considering more detailed interaction between a customer subset and edges linking it to the remainder of the graph rather than only a simple boundary count. Letchford, Eglese, and Lysgaard developed multistar and partial-multistar inequalities specifically for the CVRP, and they became part of sophisticated CVRP branch-and-cut algorithms. citeturn15search3turn15search7

For a multi-route CPTP they are potentially useful through projected edge variables

\[
x_{ij}
=
\sum_r b_{ijr}\lambda_r,
\]

but again the inequalities need **selection-aware lifting** because the classical versions assume mandatory customer service.

**Subset-row inequalities.** Subset-row inequalities (SRIs) became especially important because they attack fractional structure directly in the route master. Jepsen et al. derived them as Chvátal–Gomory rank-one cuts from subsets of set-partitioning constraints. citeturn15search2turn19search15

The easiest case to understand uses three customers

\[
S=\{i,j,k\}.
\]

Any route serving at least two members of \(S\) gets coefficient one. Since two selected routes each containing at least two of three customers would necessarily share at least one customer, the master obeys

\[
\sum_{r:\,|r\cap S|\ge2}\lambda_r\le1.
\]

More generally, rank-one cuts arise by aggregating master rows with fractional multipliers and rounding.

SRIs can strengthen route formulations substantially, but their route coefficients depend on path history; hence they are nonrobust and can inflate the pricing state.

This motivated **limited-memory subset-row/rank-one cuts**. Pecin et al. developed sharper limited-memory SRIs for VRPTW, with the cut memory represented through selected arcs rather than simply selected nodes. citeturn22search23 More recent route formulations continue to rely on limited-memory variants precisely because they trade some cutting complexity for manageable pricing. citeturn22search10

For your PTP this family is unusually appealing because optional-customer set-packing already naturally creates the combinatorial logic behind three-row SRIs. A research contribution could be a **profit-aware limited-memory rank-one family** in which memory is selected according to customer selection likelihood or dual-profit interaction.

**Path and \(k\)-path cuts.** Kohl et al.'s 2-path cuts strengthen VRPTW by identifying customer subsets that cannot be covered by one feasible route and enforcing the necessary routing structure; they integrated those cuts into a Dantzig–Wolfe master. citeturn19search2 Desaulniers et al. subsequently investigated generalized \(k\)-path inequalities alongside partial-elementarity pricing. citeturn21search24

For CPTP-TW this idea is even richer. A subset may be infeasible on a single route because of:

\[
\text{capacity},
\quad
\text{time windows},
\quad
\text{duration},
\quad
\text{precedence},
\quad
\text{time-dependent travel}.
\]

The complication is that the subset need not all be selected. A new inequality should therefore reason simultaneously about

\[
\text{how many of these customers are selected}
\]

and

\[
\text{how many routes are required to accommodate them}.
\]

That is a promising source of genuinely PTP-specific cuts rather than borrowed CVRP cuts.

**LCI / lifted cover inequalities.** I interpret your acronym **LCI** as **Lifted Cover Inequality**, which is the standard interpretation in this area, although the acronym is context-dependent.

For a binary knapsack row

\[
\sum_j a_jz_j\le b,
\]

a set \(C\) satisfying

\[
\sum_{j\in C}a_j>b
\]

is a cover, implying

\[
\sum_{j\in C}z_j\le |C|-1.
\]

Variables outside \(C\) can subsequently be lifted into the inequality with carefully derived coefficients.

This becomes relevant when a rich routing master contains knapsack-type coupling constraints—shift capacities, depot throughput, limited resources, route budgets, driver capacities, charging slots, or selection budgets. BPC work on routing with shifts has specifically exploited cover inequalities, while recent work has investigated strong nonrobust knapsack cuts for capacitated location-routing and related VRPs. citeturn15search28turn22search1

For CPTP-TW-TC, LCIs are potentially more interesting than in plain CVRP because customer selection naturally creates knapsack structures. For example, a group of attractive customers may collectively require more route time or capacity than an available resource can provide; a lifted cover can restrict their simultaneous selection.

**Branching.** The branching rule must accomplish two objectives simultaneously:

\[
\text{remove the current fractional solution}
\]

and

\[
\text{preserve tractable pricing}.
\]

Common routing alternatives include arc/edge branching, cutset branching, Ryan–Foster together/apart branching, and more recent cluster branching. Cluster branching groups multiple edge decisions into customer-level structures and was proposed specifically to overcome limitations of conventional VRP branch types. citeturn17search1

For profitable routing there is an additional, highly natural option:

\[
\boxed{y_i=0\quad\text{versus}\quad y_i=1.}
\]

This directly settles the central PTP decision—skip customer \(i\) or serve it.

The \(y_i=0\) child is pricing-friendly: remove \(i\) from the pricing graph.

The \(y_i=1\) child requires at least one generated route to serve \(i\). If none exists in the current RMP, Farkas pricing can generate one.

My recommended branching hierarchy for your problem is therefore:

\[
\boxed{
\text{customer-selection branching}
\rightarrow
\text{cluster/edge branching}
\rightarrow
\text{specialized temporal branching only if needed}.
}
\]

That is materially different from a standard VRPTW BPC and could itself become a publishable methodological contribution.

Recent research emphasizes that branching remains a major bottleneck: You et al.'s 2026 Operations Research paper notes that learning to branch in BPC is particularly difficult because route columns change continuously and branching can add dense constraints to the RMP. citeturn22search7


## What the literature says about successful BPC design

No finite review can literally include every paper using BPC, but the following set captures the main intellectual lineage and the papers most directly relevant to constructing your algorithm.

| Work | Main contribution | Relevance to your CPTP-TW BPC |
|---|---|---|
| Dantzig & Wolfe, 1960 | Decomposition of large structured LPs | Mathematical origin of route masters and pricing citeturn20search6 |
| Gilmore & Gomory, 1961/1963 | Practical dynamic generation of cutting patterns | Canonical origin of column generation citeturn20search3turn20search11 |
| Little et al., 1963 | Branch-and-bound for TSP | Historical tree-search foundation citeturn20search8 |
| Kolen et al., 1987 | B&B for VRPTW | Early exact treatment of capacity plus TWs citeturn19search21 |
| Desrosiers, Sauvé & Soumis, 1988 | Lagrangian methods for time-window routing | Shows early decomposition/resource pricing connection citeturn19search13 |
| Dell'Amico, Maffioli & Värbrand, 1995 | Profitable Tour Problem | Direct origin of your problem family citeturn16search1 |
| Kohl & Madsen, 1997 | VRPTW Lagrangian relaxation | Useful for bounds, multiplier strategies, pricing structure citeturn19search3 |
| Barnhart et al., 1998 | Modern branch-and-price synthesis | Core theoretical basis of B&P citeturn13search13 |
| Kohl et al., 1999 | 2-path cuts in DW VRPTW | Template for integrating temporal infeasibility cuts with pricing citeturn19search2 |
| Vanderbeck, 2000 | DW reformulation and B&P branching | Key theory behind strong reformulations and pricing-compatible branching citeturn13search4 |
| Letchford, Eglese & Lysgaard, 2002 | Multistar/partial-multistar cuts | Candidate capacity/connectivity cuts for multi-route CPTP citeturn15search3 |
| Lysgaard, Letchford & Eglese, 2004 | Strong CVRP branch-and-cut | Major catalogue of useful VRP inequalities citeturn15search7 |
| Fukasawa et al., 2006 | Robust BCP for CVRP | Essential paper on combining cuts and pricing without destroying subproblem tractability citeturn15search0 |
| Jepsen et al., 2008 | Subset-row inequalities for VRPTW BPC | Foundation for route-master rank-one cuts citeturn15search2 |
| Desaulniers et al., 2008 | Partial elementarity, heuristic pricing, generalized \(k\)-path cuts | Model for staged pricing and relaxed elementarity citeturn21search24 |
| Archetti et al., 2009 | CPTP and capacitated team orienteering | Direct capacitated-profit routing literature citeturn16search2 |
| Baldacci, Mingozzi & Roberti, 2011 | \(ng\)-route relaxation and pricing strategies | Probably the single most relevant pricing-relaxation paper for your design citeturn21search0turn22search22 |
| Jepsen et al., 2014 | Branch-and-cut CPTP | Direct baseline against which a CPTP BPC should be compared citeturn16search3 |
| Pecin et al., 2017 | Enhanced VRPTW BPC and arc-memory SRIs | Blueprint for modern nonrobust cutting citeturn22search23 |
| Sun et al., 2018 | Time-dependent CPTP with TW and precedence | Closest published problem structure to your proposed temporal PTP citeturn22search0 |
| Pessoa et al., 2020 | Generic exact VRP solver | Shows how modern BPC components can be generalized across VRPs and even related non-routing problems citeturn19search0 |
| Sadykov et al., 2021 | Bucket-graph labeling | Important pricing-engine acceleration citeturn17search16 |
| Yang, 2022/2023 | Exact price-cut-and-enumerate | Shows value of switching from CG to enumeration near the optimum citeturn22search2 |
| Resource-window reduction, 2023 | Reduced-cost temporal window tightening | Particularly attractive for TW/duration pricing citeturn15search23 |
| VRPSolverEasy, 2023/2024 | Accessible rich-VRP BCP interface | Useful reference implementation/prototyping option citeturn22search4 |
| DeLuxing, 2025 | Lagrangian fixing for CG | Promising path toward aggressive variable/state elimination citeturn17search23 |
| Silva, Uchoa & Subramanian, 2025 | Cluster branching | Natural second-stage branching candidate after PTP customer-selection branching citeturn17search1 |
| You et al., 2026 | Two-stage learning to branch in BPC | Evidence that adaptive branching is an active frontier citeturn22search7 |
| Eveborn & Rönnberg, 2026 preprint | Pricing for integrality | Directly attacks the weak connection between LP-useful and integer-useful columns citeturn17search3 |
| Liu & Yang, Oct. 2026 preprint | Learned column prefilling and learned dual center | Suggests learning-based CG warm starts without sacrificing final exact certification citeturn22academia26 |

Several general lessons emerge from this literature.

**First, route formulation quality matters more than compactness.** Modern exact routing increasingly accepts an exponential variable set because Dantzig–Wolfe decomposition embeds difficult route-level constraints directly in columns. Generic VRPSolver results show that this paradigm applies not only across many VRP variants but even to related pattern-based problems such as bin packing, vector packing, and generalized assignment. citeturn19search0

**Second, BPC is not intrinsically a routing-only technique.** Pattern-selection structures appear in inventory-routing, maritime routing, kidney exchange, machine scheduling, packing, assignment, crew scheduling, cutting stock, and other problems. A BPC for inventory-routing, for example, outperformed a branch-and-cut approach on challenging multivehicle instances; branch-price-and-cut has also been used for maritime transportation and kidney exchange. citeturn17search11turn17search15turn17search8

The unifying structure is:

\[
\boxed{
\text{many feasible patterns}
+
\text{few global linking constraints}
+
\text{a solvable pricing oracle}.
}
\]

**Third, pricing eventually becomes the main computational battlefield.** The theoretical route space is exponential, the pricing problem is generally NP-hard for rich VRPs, and every nonrobust cut or branching constraint can enlarge its state. citeturn21search11 Consequently, state-of-the-art work increasingly improves labeling internals—bucket graphs, bidirectional labeling, resource windows, partial dominance, and related techniques—rather than merely inventing stronger master formulations. Bucket-graph storage was shown to reduce dominance checks substantially in resource-constrained shortest-path labeling. citeturn17search16

**Fourth, the best relaxation can change during the solve.** Early iterations primarily need cheap columns; exact certification is needed only when heuristic/relaxed pricing can no longer improve the RMP. This motivates a cascade such as

\[
\text{heuristic pricing}
\rightarrow
\text{weak relaxed pricing}
\rightarrow
\text{\(ng\)-pricing}
\rightarrow
\text{exact elementary pricing}.
\]

DSSR and \(ng\)-route research embody this progressive-strengthening philosophy. citeturn21search1turn22search22

**Fifth, cutting strength must be priced in terms of pricing burden.** Robust CVRP BCP, SRIs, and limited-memory rank-one cuts form a progression: strong cuts help only to the extent that they do not overwhelm route pricing. citeturn15search0turn15search2turn22search23

**Sixth, LP quality and integer-solution quality are not identical objectives.** Traditional pricing optimizes reduced cost relative to the LP dual. A spectacular reduced-cost column need not fit well into an integer route set. The 2026 pricing-for-integrality preprint explicitly targets this mismatch and reports substantially improved root primal gaps on its electric-VRP test problem. citeturn17search3 This is particularly significant for profitable routing, where the solver must settle a combinatorial customer-selection pattern in addition to routing.

**Finally, modern exact algorithms increasingly become hybrid.** Column generation, separation, enumeration, restricted-master MIPs, local search, route pools, Lagrangian bounds, and machine learning can coexist while exactness is preserved by a final certified pricing/search layer. Yang's exact price-cut-and-enumerate approach is an instructive example: after obtaining strong bounds through alternating column and row generation, it transitions toward enumeration and exact completion rather than insisting that one computational mechanism be optimal throughout the solve. citeturn22search2


## A practical BPC architecture for a CPTP with time windows and time constraints

For a concrete algorithm, assume customers \(i\in V\) have

\[
p_i=\text{profit},\qquad
d_i=\text{demand},\qquad
[e_i,l_i]=\text{time window},\qquad
s_i=\text{service duration}.
\]

Vehicles have capacity \(Q\), possibly a maximum number \(K\), and maximum route duration \(H\). Travel cost is \(c_{ij}\), while travel time may be either static \(\tau_{ij}\) or time-dependent \(\tau_{ij}(t)\).

Let \(\mathcal R\) be all routes satisfying route-local restrictions.

For route \(r\),

\[
C_r=\text{routing/operating cost},
\qquad
P_r=\sum_i p_i a_{ir}-C_r.
\]

The master should be

\[
\begin{aligned}
\max\quad
&\sum_i p_i y_i-\sum_r C_r\lambda_r\\
\text{s.t.}\quad
&\sum_r a_{ir}\lambda_r=y_i &&\forall i,\\
&\sum_r\lambda_r\le K,\\
&\text{global resource/selection constraints},\\
&y_i\in\{0,1\},\quad \lambda_r\in\{0,1\}.
\end{aligned}
\tag{CPTP-MP}
\]

For the LP-RMP:

\[
0\le y_i\le1,\qquad \lambda_r\ge0.
\]

This formulation has several advantages. Optional selection is explicit; customer selection can be branched on; customer-side valid inequalities can be inserted; business constraints such as service quotas, required customer groups, total selected demand, minimum collected profit, or limited slots can act directly on \(y\); and route generation remains responsible only for constructing internally feasible tours.

For a pure single-tour CPTP, add

\[
\sum_r\lambda_r=1.
\]

But that exposes why an outer route-based BPC is less compelling in the strict single-tour case: the master is largely selecting one complete route from \(\mathcal R\). Unless substantial global constraints remain outside the route, solving the pricing problem is almost solving the original problem. This is why the literature on the CPTP has naturally included direct branch-and-cut approaches. citeturn16search3

For a **team/multivehicle CPTP**, however,

\[
\sum_r\lambda_r\le K
\]

creates genuine route-selection structure, and BPC becomes much more attractive. Archetti et al.'s joint treatment of the capacitated team-orienteering and profitable-tour families reinforces this connection. citeturn16search2

The pricing problem should be a profitable ESPPRC:

\[
\max_{r\in\mathcal R}
\left\{
\sum_{i\in r}\tilde p_i
-
\sum_{(i,j)\in r}\tilde c_{ij}
-
\text{cut penalties}
\right\},
\]

subject to

\[
\sum_{i\in r}d_i\le Q,
\]

\[
e_i\le t_i\le l_i,
\]

\[
t_j\ge t_i+s_i+\tau_{ij}(t_i+s_i),
\]

\[
t_{\text{return}}-t_{\text{departure}}\le H,
\]

plus elementarity, precedence, charging/break/work constraints if relevant.

Here \(\tilde p_i\) and \(\tilde c_{ij}\) incorporate master duals.

I would implement **four pricing levels**.

| Pricing level | Purpose |
|---|---|
| Fast heuristic route generator | Produce many useful columns quickly, without proof obligations |
| Relaxed/DSSR or small-\(ng\) labeling | Rapidly expose strong reduced-cost directions |
| Adaptive larger-\(ng\) exact-ish pricing | Strengthen the route relaxation as convergence approaches |
| Full elementary pricing | Provide the final column-generation certificate whenever needed |

This staged design is consistent with the direction of the pricing literature: partial elementarity, \(ng\)-route relaxation, DSSR, and modern labeling techniques all aim to spend full ESPPRC effort only when necessary. citeturn21search24turn21search1turn22search22

A high-level node algorithm should look like this:

```text
solve_node(node):

    inherit useful parent columns, cuts, and pricing state

    apply node branching restrictions
    preprocess route graph and resource windows

    repeat:

        solve restricted master LP

        if RMP is infeasible:
            run Farkas pricing

            if no Farkas column exists:
                fathom node as infeasible
                return

            add feasibility columns
            continue

        run fast heuristic pricing
        add improving columns

        if heuristic pricing finds nothing:
            run progressively stronger pricing

        if exact pricing proves no improving column:
            separate robust cuts

            if useful robust cuts were added:
                continue

            separate selected nonrobust cuts

            if useful nonrobust cuts were added:
                continue

            exact pricing again
            if still no improving column:
                break

    node_bound = certified full-master LP bound

    run primal / restricted-master / pricing-for-integrality heuristics

    if node_bound cannot beat incumbent:
        fathom node
        return

    if solution is feasible and integral:
        update incumbent
        return

    choose branching disjunction
    create children
```

The second exact-pricing call after cuts is not cosmetic. New rows produce new dual multipliers, and therefore new reduced costs:

\[
\bar P_r^{\text{before cut}}
\neq
\bar P_r^{\text{after cut}}.
\]

A route that was unattractive before cutting may become improving afterward.

The **cut hierarchy** I recommend is:

\[
\boxed{
\text{cheap robust cuts}
\rightarrow
\text{selection-aware capacity/path cuts}
\rightarrow
\text{limited-memory rank-one cuts}
\rightarrow
\text{expensive specialized cuts}.
}
\]

For the initial solver, I would prioritize the following.

**Selection and packing cuts.** Three-row subset inequalities are a natural starting point because they operate directly on route overlap and remain meaningful when customers are optional. Limited-memory versions can then be added once baseline pricing works. Jepsen et al. and Pecin et al. provide the central precedents. citeturn15search2turn22search23

**Capacity/route-count cuts with optional-service lifting.** Do not transplant classical RCCs blindly. Derive variants that explicitly incorporate \(y_i\).

**Time-window and duration \(k\)-path cuts.** Precompute or dynamically detect subsets that cannot fit in a single feasible route and create conditional cuts based on their selected customers. Kohl et al. and Desaulniers et al. give the classical required-service starting points. citeturn19search2turn21search24

**Lifted covers.** Apply them to true knapsack-like master constraints, especially global duration/shift/depot/vehicle-resource constraints. Strong knapsack cuts are an active component of modern BPC research. citeturn22search1

**Multistar/partial multistar.** Evaluate these primarily for a multi-route version with meaningful projected edge flows. Their benefit in a single-tour profitable setting is likely smaller than in CVRP, whereas selection-aware path and cover cuts are more problem-specific. citeturn15search3

The **branching hierarchy** should exploit the problem's optional customers.

At a fractional solution calculate customer selections

\[
\bar y_i=\sum_r a_{ir}\bar\lambda_r.
\]

When some

\[
0<\bar y_i<1,
\]

branch

\[
y_i=0
\qquad\text{versus}\qquad
y_i=1.
\]

A good candidate is not automatically the customer closest to \(0.5\). Candidate scoring should consider the expected bound effect, customer profit, number of fractional routes containing the customer, temporal compatibility, and perhaps strong-branching estimates.

Once customer selection becomes largely integral but the route combination remains fractional, switch to **cluster or arc branching**. Cluster branching is particularly interesting because it aggregates multiple edge decisions into customer clusters and was introduced as an alternative to traditional edge, cutset, and Ryan–Foster branching. citeturn17search1

This gives a natural problem-specific hierarchy:

\[
\boxed{
\text{What customers?}
\rightarrow
\text{Which customers together?}
\rightarrow
\text{Which arcs/order?}
}
\]

rather than immediately branching on routing edges.

**Preprocessing** should be extensive because pricing benefits disproportionately from graph reduction. Remove arc \((i,j)\) whenever even the earliest possible arrival violates \(j\)'s time window. Propagate TW bounds forward and backward. Use capacity compatibility, precedence, duration, and incumbent-based reduced-cost bounds. For time-dependent travel, feasibility propagation should respect the assumed travel-time model. The close CPTP-TW work of Sun et al. demonstrates the importance of specialized temporal labeling in the time-dependent setting. citeturn22search0

Reduced-cost resource-window techniques provide a natural second layer after ordinary feasibility preprocessing. citeturn15search23

**Primal heuristics** deserve much more attention than they receive in many first BPC implementations. Periodically solve a restricted integer master over the best current columns:

\[
\lambda_r\in\{0,1\},
\quad r\in\mathcal R'.
\]

Then augment it with local route improvement or large-neighborhood search. For PTP, additionally perturb customer selection: remove low-marginal-profit customers and insert high-adjusted-profit omitted customers.

The emerging pricing-for-integrality idea is even more relevant. Instead of generating a route solely because its LP reduced cost is attractive, generate routes that complement a partial integer solution. The 2026 EVRPTW work explicitly reports that this can produce significantly better root-node primal progress than a conventional restricted-master heuristic. citeturn17search3

**Route enumeration** should become an optional late-stage mode. Once the global bound/incumbent gap is small enough, reduced-cost arguments may prove that only a finite manageable subset of routes could participate in an improving integer solution. At that point it may be faster to enumerate them and hand the resulting restricted model to a conventional MIP search. Exact price-cut-and-enumerate algorithms demonstrate the effectiveness of such phase changes. citeturn22search2

The most important implementation parameters are not isolated solver settings but interacting controls:

| Component | Main parameter | Increasing it generally does | Recommended policy for CPTP-TW |
|---|---|---|---|
| \(ng\)-relaxation | \(|NG_i|\) | Stronger bound, slower pricing | Start small; expand selectively where repeated-node artifacts occur |
| DSSR | elementarity set size | Stronger pricing, larger labels | Add memory only when cycles demonstrate it is needed |
| Pricing | number of columns returned | Fewer CG iterations but larger RMP | Return a diverse batch, not only one best route |
| Heuristic/exact pricing | switching threshold | Controls certification cost | Heuristic first; exact only after heuristic exhaustion |
| Dual stabilization | smoothing/trust parameter | Reduces oscillation but can slow response | Adaptive rather than fixed |
| Column pool | retention/aging | More warm-start information but larger LP | Keep frequently useful or near-negative-reduced-cost routes |
| Robust cuts | batch/efficacy threshold | Stronger master with little pricing-state cost | Use aggressively near root |
| Nonrobust cuts | family/memory limit | Stronger master but harder pricing | Add selectively based on measured gain |
| Rank-one memory | number of remembered arcs/nodes | Stronger SRI representation but more states | Grow only for high-value cuts |
| Branching | candidate budget | Stronger tree decisions but more probe solves | Selection branching first; strong/ML scoring on small candidate set |
| Resource windows | reduction effort | Smaller label space | Recompute when incumbent or dual bound improves materially |
| Restricted-master MIP | frequency/time budget | Better incumbents but overhead | More frequent near root and after large column influx |
| Enumeration | gap threshold | Replaces pricing by finite route pool | Trigger only when reduced-cost filtering predicts manageable pool |
| Numerical tolerances | RC/Farkas feasibility tolerance | Changes proof reliability | Tie to LP solver tolerances; reverify borderline columns |

Those policies are more important than prescribing universal numerical constants. Optimal neighborhood sizes, cut limits, and pricing batches differ substantially with time-window width, capacity tightness, profit distribution, route duration, and graph density.

A serious computational study should therefore classify instances by at least:

\[
\frac{\sum_i d_i}{KQ},
\qquad
\text{TW tightness},
\qquad
\frac{p_i}{\text{local routing cost}},
\qquad
H,
\qquad
n,
\qquad
\text{time-dependence intensity}.
\]

The final exactness logs should distinguish

\[
\text{heuristic pricing failure}
\]

from

\[
\text{exact pricing certificate}.
\]

SCIP is one possible framework for implementing this architecture: its documentation explicitly provides pricers, Farkas-pricing callbacks, and branch-price infrastructure. citeturn18search6turn18search2 For a routing-focused prototype, VRPSolver/VRPSolverEasy is also highly relevant because the underlying solver implements generic BCP technology for a broad family of rich VRPs. citeturn19search0turn22search4 A highly customized time-dependent profitable model may nevertheless require working below the easiest interface so that selection branching, specialized labeling, and new cut families remain under your control.


## A research-grade BPC design specifically for the CPTP-TW-TC

The strongest potential contribution is not “apply BPC to another VRP.” It is to exploit the mathematical interaction between **profits and optionality** on one side and **time/resource feasibility** on the other.

I would organize the proposed algorithm around the following conceptual decomposition.

\[
\boxed{
\begin{array}{c}
\textbf{Master}\\
\text{select customers + combine routes}\\
\Updownarrow\\
\textbf{Pricing}\\
\text{construct profitable resource-feasible routes}\\
\Updownarrow\\
\textbf{Cuts}\\
\text{eliminate impossible/weak fractional combinations}\\
\Updownarrow\\
\textbf{Branching}\\
\text{fix customer and route structure}
\end{array}}
\]

The primary master variables should be

\[
y_i,\lambda_r,
\]

with optional projected edge variables

\[
x_{ij}
=
\sum_r b_{ijr}\lambda_r
\]

computed as needed for separation and branching rather than necessarily maintained as independent variables.

The pricing state should initially include

\[
(i,q,t,d,NG,\rho)
\]

and gain additional components only when a cut or side constraint makes them necessary.

For time-dependent travel with precedence, it may become

\[
(i,q,t,d,NG,P,\mathbf z,\rho),
\]

where \(P\) represents precedence state and \(\mathbf z\) limited-memory cut state.

A major design rule should be:

\[
\boxed{\text{Every proposed new master inequality must be evaluated for its pricing-state cost.}}
\]

A cut should not enter merely because it closes a large instantaneous LP gap.

For each family \(f\), estimate

\[
\eta_f
=
\frac{\text{observed bound improvement}}
{\text{separation time}
+
\text{induced additional pricing time}}.
\]

Cuts with persistently low \(\eta_f\) should be delayed or disabled. This transforms cut management from a static hierarchy into a feedback-control problem.

A **selection-aware capacity cut** deserves particular attention. In an ordinary CVRP, all demands in \(S\) must be served. In your model, the realized demand in \(S\) is

\[
D_y(S)
=
\sum_{i\in S}d_i y_i.
\]

The number of routes required to service selected customers in \(S\) depends on

\[
\left\lceil\frac{D_y(S)}{Q}\right\rceil,
\]

but the ceiling of a linear expression in binary variables cannot simply be substituted into a classical RCC as though it were a constant. The research problem is to derive linear lifted inequalities or families indexed by covers of selected demand.

One route is to identify capacity covers

\[
C\subseteq S
\]

such that

\[
\sum_{i\in C}d_i>Q.
\]

If all members of \(C\) are selected, they cannot belong to one route. Similar covers can be generated using temporal feasibility rather than only load.

That suggests a broader family:

\[
\boxed{\textbf{Resource-Incompatibility Covers}}
\]

where a customer subset \(C\) is certified impossible on one route because of any combination of

\[
Q,\quad H,\quad [e_i,l_i],\quad\text{precedence},\quad\text{time-dependent travel}.
\]

A dynamic program can certify the minimum number of feasible routes required by a selected subset, leading to **selection-activated \(k\)-path inequalities**. This would generalize the classical VRPTW path-cut idea to optional-profitable customers. Classical 2-path and generalized \(k\)-path work provides a direct theoretical starting point. citeturn19search2turn21search24

Another promising direction is a **profit-aware rank-one cut**. Standard SRI separation sees customers symmetrically except through the current fractional solution. But in PTP, customer profits and dual-adjusted profits contain valuable information.

For example, for a candidate triple

\[
S=\{i,j,k\},
\]

prioritize separation when

\[
\bar y_i,\bar y_j,\bar y_k
\]

are fractional **and**

\[
p_i-u_i,\quad p_j-u_j,\quad p_k-u_k
\]

are simultaneously high. The cut itself must remain purely valid combinatorially; profit is used only to prioritize which candidate subsets are worth separating. That preserves exactness while making separation PTP-aware.

The same concept applies to limited-memory design: allocate more cut memory to customer interactions that repeatedly appear in highly profitable fractional routes.

The **pricing relaxation** can similarly become profit-aware. Standard \(ng\)-neighborhoods commonly emphasize spatial or routing proximity. Instead define a compatibility score

\[
s_{ij}
=
w_1\,\text{distance compatibility}
+
w_2\,\text{TW overlap}
+
w_3\,\text{capacity interaction}
+
w_4\,\text{dual co-occurrence}
+
w_5\,\text{profit complementarity}.
\]

Then choose

\[
NG_i
=
\text{highest-scoring neighbors under }s_{ij}.
\]

The idea is not to let machine learning or profits alter validity. They alter **which nodes are remembered**, and exact elementary certification remains available.

This can be made adaptive:

\[
NG_i^{(k+1)}
=
NG_i^{(k)}
\cup
\{\text{nodes responsible for offending cycles or weak bounds}\}.
\]

Thus the solver learns where elementarity actually matters.

The **branching strategy** should exploit the hierarchy of decision uncertainty.

Define customer-selection fractionality

\[
F_y
=
\sum_i\min\{\bar y_i,1-\bar y_i\}.
\]

Define route/edge fractionality

\[
F_x
=
\sum_{(i,j)}
\min\{\bar x_{ij},1-\bar x_{ij}\}.
\]

When \(F_y\) is large, prioritize \(y\)-branching. When \(F_y\) becomes small but \(F_x\) remains large, move to cluster/edge branching.

This produces an adaptive rule of the form

\[
\text{branch type}=
\begin{cases}
\text{selection}, &F_y/\lvert V\rvert>\theta,\\
\text{cluster}, &\text{otherwise if strong fractional clusters exist},\\
\text{edge}, &\text{otherwise}.
\end{cases}
\]

Cluster branching's recent success and the 2026 learning-to-branch work make this a particularly timely direction. citeturn17search1turn22search7

Strong branching could evaluate candidate \(y_i\) decisions by solving approximate child RMPs using only heuristic pricing. The chosen branch would then be processed exactly. This preserves exactness because approximate probing affects only **which valid branch is chosen**, not any bound used for final pruning.

A second major PTP-specific idea is **pricing for integrality**.

Standard pricing asks

\[
\max_r\bar P_r.
\]

Instead, suppose a primal heuristic has temporarily fixed or strongly favored customer selections

\[
\hat y_i.
\]

Define a heuristic pricing score

\[
\bar P^{\,I}_r
=
\bar P_r
+
\alpha\,\text{compatibility}(r,\hat y)
-
\beta\,\text{redundancy}(r,\mathcal P),
\]

where \(\mathcal P\) is the existing column pool.

These routes need not replace ordinary pricing. They are inserted as **additional heuristic columns**, after which exact reduced-cost pricing continues unchanged. Therefore the method cannot invalidate the proof.

Recent pricing-for-integrality research explicitly argues that LP-driven columns are not guaranteed to be useful for constructing integer solutions and demonstrates the value of generating complementary routes for a partial integer solution. citeturn17search3 For PTP the opportunity should be even larger because fractional LP solutions may distribute coverage over many moderately profitable customers, whereas the integer optimum often needs a coherent subset of customers whose geography and time windows fit together.

A third new component should be **profit-resource completion bounds inside the label algorithm**.

For a label \(L\), let

\[
\rho(L)
\]

be its current reduced profit and let

\[
UB_{\text{future}}(L)
\]

be an upper bound on additional reduced profit attainable from unvisited feasible customers.

If

\[
\rho(L)+UB_{\text{future}}(L)
\le 0,
\]

then that label cannot yield an improving pricing column and can be deleted.

\(UB_{\text{future}}\) could use:

- remaining vehicle capacity;
- remaining route duration;
- customer TW compatibility;
- a fractional knapsack bound on remaining positive adjusted profits;
- reduced-cost resource windows;
- shortest return-to-depot time;
- precedence feasibility.

This is especially promising for PTP because, unlike ordinary VRPTW pricing where every customer must ultimately be covered somewhere, the pricing route has an explicit reason to skip low-value vertices.

A fourth component should be **root-node dual and column warm-starting**. An October 14, 2026 preprint by Liu and Yang reports substantial root-pricing acceleration from learned column prefilling and prediction of a smoothing center, while still using certified exact pricing afterward. Because that result is currently a preprint rather than an established peer-reviewed result, it should be treated as experimental evidence rather than settled best practice. citeturn22academia26

For your algorithm, a non-learning version could be developed first:

\[
\text{ALNS/greedy profitable routes}
\rightarrow
\text{initial column pool}
\rightarrow
\text{estimate initial customer duals}
\rightarrow
\text{stabilized CG}.
\]

A learned model could later predict

\[
\Pr(i\text{ selected}),
\quad
\Pr((i,j)\text{ useful}),
\quad
\text{initial dual center},
\]

but never decide feasibility or final optimality.

A fifth component should be **decision-diagram or relaxed-DP bounding**. Decision-diagram work has recently developed route relaxations capable of producing strong dual bounds, and the framework is explicitly connected to dynamic programming models used in BPC. citeturn21search3 A relaxed decision diagram could provide:

\[
UB_{\text{future}}(L)
\]

for pricing labels, an initial root bound, or a mechanism to identify highly consequential state variables before full elementarity is introduced.

Finally, **time-dependent travel should be treated as a first-class modeling choice**, not a small extension. Sun et al.'s time-dependent CPTP already shows that the combination of time-dependent travel, TWs, precedence, capacity, and profit changes the labeling problem materially. citeturn22search0 A publishable BPC contribution would be particularly compelling if its pricing and cuts are designed to exploit a FIFO time-dependent network rather than merely replacing every static \(\tau_{ij}\) by a lookup function.


## Promising future research agenda and an experimental roadmap

The most promising contribution would be a **selection-aware, adaptive BPC framework for profitable routing**, rather than a collection of existing VRPTW components assembled unchanged.

I would prioritize the research ideas below in roughly this order.

**Profit-aware adaptive \(ng\)-route relaxation.** The \(ng\)-route method is already one of the most important route relaxations in exact vehicle routing. citeturn22search22 Its natural extension for PTP is to let neighborhood memory depend not just on geography but on temporal compatibility, adjusted profit, fractional route co-occurrence, and repeated-cycle evidence. The hypothesis is that a smaller but better chosen \(NG_i\) can approximate elementary routes more effectively than a larger nearest-neighbor neighborhood.

An empirical test should compare:

\[
NG^{\text{distance}},
\quad
NG^{\text{time}},
\quad
NG^{\text{profit}},
\quad
NG^{\text{hybrid}},
\quad
NG^{\text{adaptive}}.
\]

Measure not only total time but

\[
\text{root bound},
\quad
\text{labels},
\quad
\text{dominance ratio},
\quad
\text{pricing time},
\quad
\text{number of offending cycles}.
\]

**Selection-aware \(k\)-path/resource-cover inequalities.** Classical VRPTW path cuts assume all customers are required. citeturn19search2turn21search24 A PTP analogue should derive lower bounds on the number of routes required by an **arbitrary selected subset**. Capacity, TW, route-duration, and precedence information can jointly establish that certain combinations cannot coexist on one route.

One theoretical target is a function

\[
\kappa(S,Y)
=
\text{minimum number of routes required for the selected customers in }S,
\]

followed by linear valid inequalities approximating the nonlinear dependence on \(Y=(y_i)\).

Even a restricted family based on covers of size three to six could be computationally useful.

**Hybrid robust/nonrobust cut scheduling.** Fukasawa-style robust cuts avoid increasing pricing complexity, whereas SRIs and similar cuts can be stronger but nonrobust. citeturn15search0turn15search2 Instead of a fixed schedule, maintain statistics for each family:

\[
\text{LP improvement},
\quad
\text{pricing slowdown},
\quad
\text{cut activity},
\quad
\text{age},
\quad
\text{node reduction}.
\]

Then dynamically activate a family when its estimated benefit exceeds its pricing cost. This is a form of online algorithm configuration and does not jeopardize exactness because it changes only the choice among valid cuts.

**Selection-first cluster branching.** Cluster branching is one of the most interesting recent exact-routing developments. citeturn17search1 CPTP offers an extra layer of branching unavailable in ordinary mandatory-service VRP:

\[
\text{customer selection}
\rightarrow
\text{customer clustering}
\rightarrow
\text{routing edges}.
\]

A paper could compare:

\[
\text{edge only},
\quad
y\text{-then-edge},
\quad
y\text{-then-cluster},
\quad
\text{adaptive learned choice}.
\]

The 2026 two-stage learning-to-branch paper provides contemporary evidence that BPC-specific branching selection is sufficiently difficult and important to merit dedicated learning methods. citeturn22search7

**Pricing for integrality adapted to profitable routing.** The 2026 work by Eveborn and Rönnberg is highly suggestive: reduced-cost pricing optimizes the continuous relaxation, not integer complementarity. citeturn17search3 For CPTP, design a heuristic pricer explicitly around a partial chosen-customer set.

Its objective could balance

\[
\text{LP reduced profit}
+
\text{integer completion value}
+
\text{route diversity}
-
\text{overlap with existing columns}.
\]

The crucial methodological point is that this remains a **heuristic secondary pricer**. Exact reduced-cost pricing still supplies the certificate.

**Reduced-cost resource windows plus profit bounds.** The recent success of directional arc-specific resource-window reduction suggests combining reduced-cost temporal elimination with a PTP-specific remaining-profit bound. citeturn15search23 A label or arc could be removed when even its best possible continuation cannot produce a positive-reduced-profit route or cannot improve the incumbent enough to matter.

Conceptually,

\[
\text{time feasibility}
+
\text{capacity feasibility}
+
\text{profit potential}
+
\text{reduced-cost gap}
\]

should be propagated jointly rather than separately.

**Farkas-informed branching and infeasibility learning.** Farkas pricing is usually treated as an implementation necessity. It can instead become an information source. When a \(y_i=1\) branch repeatedly creates difficult feasibility pricing, the Farkas ray reveals which master requirements are in tension. Because SCIP exposes these multipliers explicitly, the information is available algorithmically. citeturn18search2turn18search14

Research questions include whether Farkas rays can guide:

\[
\text{which customer-selection branches to avoid},
\]

\[
\text{which initial columns to pre-generate},
\]

or

\[
\text{which incompatible customer groups deserve cuts}.
\]

**Lagrangian-guided fixing and decomposition.** Lagrangian routing methods are historically well established, and modern DeLuxing shows renewed potential for Lagrangian fixing within column-generation algorithms. citeturn19search3turn17search23 For PTP this could be especially powerful because customer selection itself can often be fixed:

\[
y_i=0
\]

when a rigorous upper bound on every solution containing \(i\) falls below the incumbent, or

\[
y_i=1
\]

under an appropriate complementary argument.

A dual-ascent/Lagrangian phase before full BPC could also provide better dual starting points and useful columns.

**Decision-diagram-assisted pricing.** Decision-diagram route relaxations can produce strong bounds using dynamic-programming structure. citeturn21search3 Their most interesting integration into BPC would not necessarily replace column generation. Instead, a relaxed DD could:

\[
\text{bound future profit in a label},
\]

\[
\text{identify dominated resource states},
\]

\[
\text{generate cuts},
\]

or

\[
\text{decide where DSSR/\(ng\) memory should be refined}.
\]

This is a natural bridge between two modern exact-optimization paradigms.

**GPU and learned pricing acceleration with exact certification.** Bucket-graph labeling already attacks the large number of dominance comparisons in RCSP. citeturn17search16 The October 2026 column-prefilling preprint additionally reports GPU pricing work combined with CPU certification. citeturn22academia26 A safe research architecture is therefore

\[
\text{GPU/ML candidate generation}
\rightarrow
\text{CPU exact verification}.
\]

Learning should control search order, prefilling, neighborhood choice, label extension priority, and branch/cut selection—not replace feasibility or optimality certificates.

**Price-cut-and-enumerate termination.** If pricing remains difficult deep in the tree, the algorithm should consider changing paradigms. Exact price-cut-and-enumerate research demonstrates that after strong column and row generation, a finite candidate set can sometimes be enumerated and solved exactly. citeturn22search2 CPTP may be particularly amenable because an incumbent gives an immediate threshold on the profitability required from any omitted route.

A useful route-enumeration test could derive a reduced-profit threshold

\[
\bar P_r>\Delta,
\]

where \(\Delta\) is computed from the current global gap. Any route unable to reach that threshold can never participate in an improving solution.

**Dynamic and stochastic profitable routing** is a longer-term direction. Profitable routing naturally models order acceptance, where requests need not be served. A static CPTP-TW BPC could become the exact offline oracle inside a rolling-horizon or stochastic framework. The research value would be to reuse duals, columns, \(ng\)-neighborhoods, and route pools as the instance evolves rather than solving each snapshot from scratch.

For the actual research program, I would develop the solver in controlled stages rather than introducing every advanced component simultaneously.

The first baseline should contain:

\[
\text{route master}
+
\text{exact ESPPRC}
+
\text{Farkas pricing}
+
y\text{-branching}
+
\text{simple arc branching}.
\]

This proves that the formulation and pricing are correct.

The second baseline should add:

\[
\text{heuristic pricing}
+
\text{DSSR}
+
ng\text{-route relaxation}
+
\text{bidirectional/bucket labeling}.
\]

The third should add:

\[
\text{robust cuts}
+
\text{3-row/lm-SRI}
+
\text{time/resource \(k\)-path cuts}.
\]

The fourth should add:

\[
\text{restricted-master primal heuristic}
+
\text{pricing for integrality}
+
\text{resource-window reduction}
+
\text{enumeration}.
\]

Only afterward should machine learning or highly adaptive control be introduced. Otherwise it becomes difficult to determine which mechanism actually caused a performance change.

A proper experimental matrix should report at least:

| Metric | Why it matters |
|---|---|
| Root LP upper bound for maximization | Direct measure of formulation/relaxation strength |
| Root gap to best feasible solution | Combined dual/primal quality |
| CG iterations | Dual convergence behavior |
| Exact vs heuristic pricing calls | Whether expensive certification dominates |
| Generated labels | True difficulty of pricing |
| Dominance checks | Pricing-engine efficiency |
| Peak active labels | Memory scalability |
| Columns generated/retained | RMP growth |
| Cuts by family | Separation behavior |
| Bound gain per cut family | Direct cutting value |
| Pricing slowdown after each cut family | Cost of nonrobustness |
| Farkas pricing calls | Difficulty introduced by branching |
| B&B nodes | Overall branching/bound strength |
| Time to first/best incumbent | Primal performance |
| Time to certified optimum | Main exact-solver metric |
| Peak memory | Critical for route-based methods |

Ablation studies should remove one component at a time:

\[
\text{no }ng,
\quad
\text{no SRI},
\quad
\text{no stabilization},
\quad
\text{no profit-aware pricing},
\quad
\text{no selection branching},
\quad
\text{no resource windows},
\]

rather than comparing only total runtime of the final solver.

The benchmark design should vary instance structure systematically. In a profitable problem the hardest cases are unlikely to be characterized solely by customer count. A useful “profit ambiguity” measure is the fraction of customers for which

\[
p_i \approx \text{marginal routing cost of serving }i.
\]

When almost every customer is extremely profitable, the problem tends toward a mandatory-service VRP. When almost none is profitable, selection is easy. The genuinely PTP-specific difficulty should occur when many customers sit near the inclusion/exclusion boundary.

Similarly, vary time-window tightness and route duration to create a spectrum:

\[
\text{loose TW}
\rightarrow
\text{capacity-dominated}
\]

through

\[
\text{mixed}
\]

to

\[
\text{tight TW}
\rightarrow
\text{temporal-feasibility-dominated}.
\]

That is essential for deciding whether a proposed cut or \(ng\)-memory strategy is genuinely robust rather than tuned to one instance type.

The most compelling end-state algorithm would therefore be something like

\[
\boxed{
\begin{aligned}
&\textbf{Selection-aware stabilized Branch-Price-and-Cut}\\
&+\text{ adaptive profit/time-aware }ng\text{-pricing}\\
&+\text{ DSSR/exact ESPPRC certification}\\
&+\text{ resource-window and profit-potential pruning}\\
&+\text{ robust capacity/temporal inequalities}\\
&+\text{ limited-memory rank-one cuts}\\
&+\text{ lifted resource-cover cuts}\\
&+\text{ customer-selection/cluster branching}\\
&+\text{ Farkas feasibility pricing}\\
&+\text{ pricing-for-integrality primal search}\\
&+\text{ late route enumeration}.
\end{aligned}}
\]

That design is theoretically grounded in the historical evolution of B&B, Dantzig–Wolfe decomposition, column generation, branch-and-price, branch-and-cut, robust BCP, resource-constrained shortest-path algorithms, \(ng\)-route relaxation, SRIs, and modern BPC accelerations. citeturn20search6turn13search13turn15search0turn15search2turn22search22turn22search23

More importantly, it is **not merely another VRPTW solver with profits added to the objective**. Its novel organizing principle is that profitable routing contains a second combinatorial layer—**whether a customer should be present at all**—and that selection decision should inform every major BPC component: pricing memory, cuts, branching, primal column generation, resource reduction, and eventually learned guidance. That is where the strongest theoretical and computational contribution for a new CPTP-with-Time-Windows-and-Time-Constraints BPC is likely to lie.