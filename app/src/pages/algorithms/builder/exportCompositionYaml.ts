import { normFamily } from "./linkRules";

export interface CompositionNode {
  uid: string;
  entryId: string;
  family: string;
  displayName: string;
  parameters?: Record<string, string>;
}

export interface CompositionEdge {
  source: string;
  target: string;
}

export interface CompositionGraph {
  problem: string;
  graphSize: number;
  seed: number;
  nodes: CompositionNode[];
  edges: CompositionEdge[];
}

/** Neural configs are not simulation runners. `na` / `hna` are routing policies, not this list. */
const ROUTE_MODEL_IDS = new Set(["na", "hna"]);

export function modulesWithoutRunner(nodes: CompositionNode[]): string[] {
  const names: string[] = [];
  for (const node of nodes) {
    const family = normFamily(node.family);
    const label = node.displayName || node.entryId;
    if (family === "separation") names.push(label);
    else if (family === "model" && !ROUTE_MODEL_IDS.has(node.entryId)) names.push(label);
  }
  return names;
}

function yamlScalar(value: string | number | boolean): string {
  if (typeof value === "number") return Number.isFinite(value) ? String(value) : "0";
  if (typeof value === "boolean") return value ? "true" : "false";
  if (/^[A-Za-z0-9_./-]+$/.test(value)) return value;
  return JSON.stringify(value);
}

function latest(nodes: CompositionNode[], family: string): CompositionNode | undefined {
  const matches = nodes.filter((node) => normFamily(node.family) === family);
  return matches[matches.length - 1];
}

/** Routing (and a route-model stand-in) roots the fragment. Other modules count once linked. */
function exportedNodes(graph: CompositionGraph): CompositionNode[] {
  const byUid = new Map(graph.nodes.map((node) => [node.uid, node]));
  const keep = new Set<string>();
  for (const node of graph.nodes) {
    const family = normFamily(node.family);
    if (family === "routing" || family === "model") keep.add(node.uid);
  }
  let grew = true;
  while (grew) {
    grew = false;
    for (const edge of graph.edges) {
      if (keep.has(edge.source) === keep.has(edge.target)) continue;
      if (!byUid.has(edge.source) || !byUid.has(edge.target)) continue;
      keep.add(edge.source);
      keep.add(edge.target);
      grew = true;
    }
  }
  return graph.nodes.filter((node) => keep.has(node.uid));
}

/**
 * Hydra/YAML fragment the simulation config already reads:
 * `sim.policies`, `packing`, `route_improvement`, `acceptance_criteria`.
 * Operator families are listed beside those keys. The function does not write a file.
 */
export function exportCompositionYaml(graph: CompositionGraph): string {
  const nodes = exportedNodes(graph);
  const lines: string[] = [
    `problem: ${yamlScalar(graph.problem || "ptp")}`,
    `seed: ${yamlScalar(graph.seed)}`,
    "sim:",
    "  policies: [" +
      nodes
        .filter((node) => normFamily(node.family) === "routing")
        .map((node) => yamlScalar(node.entryId))
        .join(", ") +
      "]",
    "  graph:",
    `    num_loc: ${yamlScalar(graph.graphSize)}`,
  ];

  const packing = latest(nodes, "packing");
  if (packing) lines.push(`packing: ${yamlScalar(packing.entryId)}`);

  const improver = latest(nodes, "route_improvement");
  if (improver) lines.push(`route_improvement: ${yamlScalar(improver.entryId)}`);

  const acceptance = latest(nodes, "acceptance");
  if (acceptance) {
    lines.push("acceptance_criteria:");
    lines.push(`  - ${yamlScalar(acceptance.entryId)}`);
    lines.push("acceptance_criterion:");
    lines.push(`  method: ${yamlScalar(acceptance.entryId)}`);
  }

  const operators = nodes.filter((node) => normFamily(node.family) === "operator");
  if (operators.length > 0) {
    lines.push("operators:");
    for (const node of operators) lines.push(`  - ${yamlScalar(node.entryId)}`);
  }

  const model = latest(nodes, "model");
  if (model) lines.push(`model: ${yamlScalar(model.entryId)}`);

  const params = nodes.filter((node) => node.parameters && Object.keys(node.parameters).length > 0);
  if (params.length > 0) {
    lines.push("parameters:");
    for (const node of params) {
      lines.push(`  ${yamlScalar(node.entryId)}:`);
      for (const [key, value] of Object.entries(node.parameters ?? {})) {
        lines.push(`    ${yamlScalar(key)}: ${yamlScalar(value)}`);
      }
    }
  }

  if (graph.edges.length > 0) {
    lines.push("links:");
    for (const edge of graph.edges) {
      const source = graph.nodes.find((node) => node.uid === edge.source);
      const target = graph.nodes.find((node) => node.uid === edge.target);
      if (!source || !target) continue;
      lines.push(`  - ${yamlScalar(source.entryId)} -> ${yamlScalar(target.entryId)}`);
    }
  }

  return `${lines.join("\n")}\n`;
}
