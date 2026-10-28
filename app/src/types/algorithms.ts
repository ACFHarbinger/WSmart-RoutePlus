export interface CatalogEntry {
  id: string;
  family: string;
  status: string;
  target?: string | null;
  stages?: string[];
  role?: string;
  interior?: string;
  error?: string | null;
  alias_of?: string | null;
}

export interface PolicyCatalog {
  packing: CatalogEntry[];
  routing: CatalogEntry[];
  route_improvement: CatalogEntry[];
  acceptance: CatalogEntry[];
  operators: CatalogEntry[];
  separation: CatalogEntry[];
  models: CatalogEntry[];
}

export interface SolutionRecord {
  routes: number[][];
  objective?: number | null;
  unchanged_hash?: string;
}

export interface TraceEvent {
  id: string;
  parent_id: string | null;
  name: string;
  kind: "input" | "stage" | "operator" | "incumbent" | "output";
  changed_solution: boolean;
  arguments: Record<string, unknown>;
  before?: SolutionRecord | null;
  after?: SolutionRecord | null;
  interior?: string;
}

export interface ExecutionTraceDocument {
  schema_version: number;
  problem: string;
  policies: Record<string, string | null>;
  seed: number | null;
  instance: Record<string, unknown>;
  events: TraceEvent[];
  final: SolutionRecord | null;
}

export interface TraceRequest {
  problem: string;
  graphSize: number;
  seed: number;
  acceptance: string;
  packing?: string | null;
  routeImprovement?: string | null;
  maxIterations: number;
}
