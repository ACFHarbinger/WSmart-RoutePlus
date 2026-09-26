export interface PipelineEntry {
  key: string;
  name: string;
}

export interface ConstructionFamily {
  family: string;
  key: string;
  constructors: PipelineEntry[];
}

export interface Pipeline {
  selection: PipelineEntry[];
  construction: ConstructionFamily[];
  improvement: PipelineEntry[];
  counts: {
    selection: number;
    constructors: number;
    families: number;
    improvers: number;
  };
  benchmark: {
    selection: string[];
    constructors: string[];
    improvers: string[];
  };
}

export interface ConstructorRow {
  name: string;
  color?: string | null;
  n: number;
  kgkm: number;
  kgkm_med: number;
  overflows: number;
  overflows_med: number;
  km: number;
  time: number;
}

export interface StrategyRow {
  key: string;
  name: string;
  n: number;
  kgkm: number;
  overflows: number;
  km: number;
  time: number;
}

export interface ParetoPoint {
  name: string;
  kgkm: number;
  overflows: number;
  front: boolean;
}

export interface ScenarioDist {
  dist: string;
  kgkm: number;
  overflows: number;
  kg_lost: number;
}

export interface ScenarioN {
  N: number;
  kgkm: number;
  km: number;
  time: number;
  overflows: number;
}

export interface HorizonCtor {
  constructor: string;
  pairs: number;
  kgkm_30: number;
  kgkm_90: number;
  overflows_30: number;
  overflows_90: number;
}

export interface ExcludedRun {
  constructor: string;
  horizon: number;
  city: string;
  N: number;
  policy: string;
  kg: number;
  shortfall: number;
  overflows: number;
}

export interface Results {
  meta: {
    primary_horizon: number;
    retained_runs: number;
    constructor_n: number;
  };
  constructors: ConstructorRow[];
  strategies: StrategyRow[];
  improvers: {
    pairs: number;
    wins: number;
    losses: number;
    ties: number;
    mean_delta_kgkm: number;
    mean_km_cls: number;
    mean_km_ftsp: number;
    loss_breakdown: { constructor: string; n: number; mean: number }[];
  };
  pareto: ParetoPoint[];
  scenario: { by_dist: ScenarioDist[]; by_N: ScenarioN[] };
  heatmap: {
    columns: string[];
    rows: { constructor: string; values: number[] }[];
  };
  horizon: {
    pairs: number;
    positive_pairs: number;
    mean_delta_kgkm: number | null;
    by_constructor: HorizonCtor[];
  };
  excluded: ExcludedRun[];
  colors: {
    constructors: Record<string, string>;
    variants: Record<string, string>;
    improvers: Record<string, string>;
  };
}

export interface BinFillScenario {
  city: string;
  N: number;
  capacity: number;
  days: number;
  fills: number[][]; // fills[bin][day], percent 0-100
  increments: number[][]; // recovered, policy-independent daily waste
}

export interface BinFills {
  capacity: number;
  scenarios: BinFillScenario[];
}

export interface RouteNode {
  id: number;
  x: number;
  y: number;
  depot: boolean;
}

export interface RouteDay {
  day: number;
  tour: number[]; // indices into nodes
  mandatory: number[];
  km: number;
  kg: number;
  overflows: number;
  kgkm: number;
  ncol: number;
}

export interface RouteScenario {
  city: string;
  N: number;
  constructor: string;
  strategy: string;
  improver: string;
  layout: string;
  nodes: RouteNode[];
  days: RouteDay[];
}

export interface Routes {
  scenarios: RouteScenario[];
}
