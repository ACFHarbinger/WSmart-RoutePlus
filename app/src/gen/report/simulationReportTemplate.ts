/**
 * Markdown section templates for simulation analysis report (§H.5).
 */
import { PLACEHOLDER } from "./markdown";
import type { HorizonRender } from "./simulationReportFigures";

export interface HorizonTemplateCtx {
  days: number;
  sectionNum: number;
  nLogs: number;
  constructors: string[];
  figuresRel: string;
  privateRel: string;
  interactive: boolean;
  paretoTable: string;
  overflowTable: string;
  efficiencyTable: string;
  kmTable: string;
  strategyBest: string;
  fullResultsTable: string;
  flags: HorizonRender["flags"];
}

export function horizonSection(h: HorizonTemplateCtx, improvers: string[], strategies: string[]): string {
  const n = h.sectionNum;
  const f = h.figuresRel;
  return `
## ${n}. ${h.days}-Day Horizon Results

> **Logs analysed:** ${h.nLogs}
> **Constructors available:** ${h.constructors.join(", ")}

### ${n}.1 Analytics Comparison — Pareto View

![Overflow vs Efficiency — Pareto Front](${f}/pareto_scatter.png)

*Scatter of all ${h.days}-day runs in the overflows–kg/km space, one panel per waste distribution. Colour encodes the packing variant, marker shape encodes the scenario (region/N), filled markers = ${improvers[0]}, open markers = ${improvers[improvers.length - 1]}. Dashed lines = Pareto fronts, one colour per scenario (region × N × distribution).*
${h.flags.paretoLog ? `
![Overflow vs Efficiency — Pareto Front (log scale)](${f}/pareto_scatter_log.png)

*Same chart with symlog X-axis — spreads the densely clustered low-overflow region.*
` : ""}${h.interactive ? `
**[Interactive version](${h.privateRel}/pareto_scatter_interactive.html)**
` : ""}
#### Pareto-Front Policy Catalogue (${h.days} days)

_TABCAP_: Pareto-optimal policy configurations over the ${h.days}-day horizon — each unique (selection variant, constructor, improver) that appeared on the Pareto front of at least one scenario, sorted by scenario count; metrics averaged across those scenarios.

${h.paretoTable}

${PLACEHOLDER}

### ${n}.2 Summary KPI Analysis

#### Overflow Performance

![Overflow Count by Configuration](${f}/overflow_by_config.png)

*Mean overflow count per scenario and packing algorithm (mean ± min/max range across routing algorithms); route improvers shown as paired bars within each configuration.*
${h.flags.overflowLog ? `
![Overflow Count by Configuration (log scale)](${f}/overflow_by_config_log.png)

*Same chart with symlog Y axis — reveals structure compressed in the linear scale.*
` : ""}
_TABCAP_: Overflow counts by configuration over ${h.days} days — min/max/mean across routing algorithms, per route improver.

${h.overflowTable}

${PLACEHOLDER}

#### Route Efficiency (kg/km)

![kg/km Efficiency by Configuration](${f}/kgkm_by_config.png)

*Mean kg/km efficiency per scenario and packing algorithm, with min–max whiskers across constructors; improvers as paired bars.*

_TABCAP_: Route efficiency (kg/km) by configuration over ${h.days} days — min/max/mean across routing algorithms, per route improver.

${h.efficiencyTable}

${PLACEHOLDER}

#### Distance Driven (km)

![Vehicle Distance by Strategy](${f}/km_violin.png)

*Distribution of total vehicle distance (km) per packing algorithm and scenario (all constructors and improvers pooled), one panel per waste distribution.*

_TABCAP_: Vehicle distance driven (km) by configuration over ${h.days} days — min/max/mean across routing algorithms, per route improver.

${h.kmTable}

${PLACEHOLDER}

### ${n}.3 Policy × Scenario Heatmaps

![Policy × Scenario Heatmap — Overflows](${f}/policy_scenario_heatmap_overflows.png)

*Overflow count heatmap: each row is a full policy configuration (selection variant + constructor + improver), each column a simulation scenario (region × N × distribution).*

![Policy × Scenario Heatmap — Efficiency](${f}/policy_scenario_heatmap_kgkm.png)

*kg/km efficiency heatmap with the same layout (rows = policy configurations, columns = scenarios).*

![Per-Scenario Constructor Heatmaps](${f}/scenario_constructor_heatmap.png)

*One panel per scenario: routing algorithms on the rows, packing algorithm × route improver combinations on the columns.*
${h.interactive ? `
**[Interactive heatmap](${h.privateRel}/policy_heatmap_interactive.html)**
` : ""}
${PLACEHOLDER}

### ${n}.4 Packing Algorithm Comparison (${strategies.join(" vs ")})

![Strategy Trade-off Bubble Chart](${f}/strategy_bubble.png)

*One panel per waste distribution. Each bubble = one (strategy, scenario) combination, averaged over constructors and improvers; bubble size ∝ N.*
${h.flags.bubbleLog ? `
![Strategy Trade-off Bubble Chart (log X scale)](${f}/strategy_bubble_log.png)

*Same chart with symlog X axis.*
` : ""}${h.interactive ? `
**[Interactive bubble chart](${h.privateRel}/strategy_bubble_interactive.html)**
` : ""}
${h.strategyBest}

${PLACEHOLDER}

### ${n}.5 Route Improver Comparison (${improvers.join(" vs ")})

![Improver Trade-off Bubble Chart](${f}/improver_bubble.png)

*Each bubble = one (improver, scenario) combination averaged over strategies and constructors — contrasts the route improvers directly.*
${h.flags.improverDelta ? `
![Improver Delta Heatmap](${f}/improver_delta.png)

*Delta heatmap (${improvers[improvers.length - 1]} − ${improvers[0]}) per constructor × configuration.*
` : ""}
${PLACEHOLDER}

### ${n}.6 Key Findings

![Policy Performance Radar](${f}/policy_radar.png)

*Overlaid radar chart for key constructors. Outer = better on all axes.*

![Policy Performance Radar — All Constructors](${f}/policy_radar_combined.png)

*Same normalised radar, overlaying every routing algorithm instead of the curated subset above.*

![Routing Algorithm Average Rank](${f}/constructor_ranking.png)

*Average rank of each routing algorithm across all scenarios and strategies (improvers pooled). Bars grow upward — shorter = better.*

${PLACEHOLDER}

### ${n}.7 Full Results Table

_TABCAP_: Full results over the ${h.days}-day horizon — rows: graph size × region × data distribution; columns: packing algorithm × routing algorithm × route improver. Each cell reports mean±std overflows and mean±std kg/km.

${h.fullResultsTable}

${PLACEHOLDER}

---
`;
}

export function comparisonSection(cmp: {
  sectionNum: number;
  label: string;
  figuresRel: string;
  firstDays: number;
  lastDays: number;
  fullResultsTable: string;
}): string {
  const f = cmp.figuresRel;
  return `
## ${cmp.sectionNum}. Horizon Comparison (${cmp.label})

This section compares results across the simulation horizons to identify which patterns are
robust across time scales and which shift as the evaluation window extends.

### Overflow Across Horizons

![Overflow Horizon Comparison](${f}/horizon_overflow_comparison.png)

*Side-by-side overflow bars for every configuration, one bar colour per horizon.
Growth across horizons indicates that overflow pressure accumulates over time.*

![Overflow Relative Delta](${f}/horizon_overflow_delta.png)

*Relative change in mean overflows between the shortest and longest horizon: (${cmp.lastDays}d − ${cmp.firstDays}d) / ${cmp.firstDays}d × 100.
Red bars = more overflows on the longer horizon; green bars = fewer.*

${PLACEHOLDER}

### Efficiency Across Horizons

![kg/km Horizon Comparison](${f}/horizon_kgkm_comparison.png)

*Side-by-side kg/km efficiency comparison.
Consistent efficiency across horizons suggests the routing policy scales well.*

${PLACEHOLDER}

### Constructor Rankings Across Horizons

![Constructor Ranking Across Horizons](${f}/horizon_constructor_ranking.png)

*Average constructor rank (lower = better) compared across horizons.
Constructors with stable ranks are robust; those that improve or regress warrant deeper investigation.*

${PLACEHOLDER}

### Key Observations

${PLACEHOLDER}

### Full Results Table — All Horizons

_TABCAP_: Full results across all simulation horizons — rows: graph size × region × data distribution; columns: horizon × packing algorithm × routing algorithm × route improver. Each cell reports mean±std overflows and mean±std kg/km.

${cmp.fullResultsTable}

${PLACEHOLDER}

---
`;
}
