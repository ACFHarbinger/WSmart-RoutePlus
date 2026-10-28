import { useAlgorithmDraftStore } from "../../../store/algorithmDraft";
import type { BuilderEdge, BuilderNode } from "./graphTypes";
import { normFamily } from "./linkRules";

export interface DraftScalars {
  problem: string;
  graphSize: number;
  seed: number;
  packing: string | null;
  routing: string | null;
  model: string | null;
  routeImprovement: string | null;
  acceptance: string | null;
  operators: string[];
}

type DraftState = DraftScalars & {
  nodes?: BuilderNode[];
  edges?: BuilderEdge[];
  setDraft: (patch: Record<string, unknown>) => void;
  clear: () => void;
};

export function readDraft(): DraftState {
  return useAlgorithmDraftStore.getState() as DraftState;
}

export function operatorIds(value: unknown): string[] {
  if (Array.isArray(value)) {
    return value.filter((item): item is string => typeof item === "string" && item.length > 0);
  }
  if (typeof value === "string" && value.trim()) {
    return value.split(/[,\s]+/).filter(Boolean);
  }
  return [];
}

export function scalarsFromNodes(nodes: BuilderNode[]): Pick<
  DraftScalars,
  "packing" | "routing" | "model" | "routeImprovement" | "acceptance" | "operators"
> {
  const latest = (family: string): string | null => {
    const matches = nodes.filter((node) => normFamily(node.family) === family);
    return matches.length > 0 ? matches[matches.length - 1].entryId : null;
  };
  return {
    packing: latest("packing"),
    routing: latest("routing"),
    model: latest("model"),
    routeImprovement: latest("route_improvement"),
    acceptance: latest("acceptance"),
    operators: nodes.filter((node) => normFamily(node.family) === "operator").map((node) => node.entryId),
  };
}

export function restoreCanvas(draft: DraftState): { nodes: BuilderNode[]; edges: BuilderEdge[] } {
  if (Array.isArray(draft.nodes) && draft.nodes.length > 0) {
    return {
      nodes: draft.nodes,
      edges: Array.isArray(draft.edges) ? draft.edges : [],
    };
  }
  const specs: { id: string; family: string }[] = [];
  if (draft.packing) specs.push({ id: draft.packing, family: "packing" });
  if (draft.routing) specs.push({ id: draft.routing, family: "routing" });
  if (draft.routeImprovement) specs.push({ id: draft.routeImprovement, family: "route_improvement" });
  if (draft.acceptance) specs.push({ id: draft.acceptance, family: "acceptance" });
  if (draft.model) specs.push({ id: draft.model, family: "model" });
  for (const id of operatorIds(draft.operators)) specs.push({ id, family: "operator" });
  return {
    nodes: specs.map((spec, index) => ({
      uid: `restored-${spec.family}-${spec.id}-${index}`,
      entryId: spec.id,
      family: spec.family,
      displayName: spec.id,
      status: "registered",
      parameters: {},
    })),
    edges: [],
  };
}

/** Mirror scalars. Persist nodes and edges only when the store already has those keys. */
export function writeDraft(nodes: BuilderNode[], edges: BuilderEdge[]): void {
  const draft = readDraft();
  const patch: Record<string, unknown> = { ...scalarsFromNodes(nodes) };
  if (Object.prototype.hasOwnProperty.call(draft, "nodes")) patch.nodes = nodes;
  if (Object.prototype.hasOwnProperty.call(draft, "edges")) patch.edges = edges;
  draft.setDraft(patch);
}

interface PersistApi {
  hasHydrated: () => boolean;
  onFinishHydration: (cb: () => void) => () => void;
}

export function draftPersist(): PersistApi | undefined {
  const store = useAlgorithmDraftStore as unknown as { persist?: PersistApi };
  return store.persist;
}
