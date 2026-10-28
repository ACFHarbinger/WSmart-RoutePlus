import { invoke } from "@tauri-apps/api/core";
import type { ExecutionTraceDocument, PolicyCatalog, TraceRequest } from "../../types/algorithms";

/** Commands are registered `rename_all = "snake_case"`, so IPC keys match the Rust arguments. */
export async function listPolicyModules(projectRoot: string): Promise<PolicyCatalog> {
  return invoke<PolicyCatalog>("list_policy_modules", { project_root: projectRoot });
}

export async function tracePolicy(
  projectRoot: string,
  request: TraceRequest
): Promise<ExecutionTraceDocument> {
  return invoke<ExecutionTraceDocument>("trace_policy", {
    project_root: projectRoot,
    problem: request.problem,
    graph_size: request.graphSize,
    seed: request.seed,
    acceptance: request.acceptance,
    packing: request.packing ?? null,
    route_improvement: request.routeImprovement ?? null,
    max_iterations: request.maxIterations,
  });
}
