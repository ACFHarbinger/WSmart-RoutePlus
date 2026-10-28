import { useEffect, useMemo, useState } from "react";
import { PolicyDraftForm, type DraftPatch } from "../../../components/algorithms/visualizer/PolicyDraftForm";
import { TraceFlow } from "../../../components/algorithms/visualizer/TraceFlow";
import { TraceInspector } from "../../../components/algorithms/visualizer/TraceInspector";
import { isCatalogEmpty } from "../../../components/algorithms/visualizer/catalogEntries";
import { useAppStore } from "../../../store/app";
import { useAlgorithmDraftStore } from "../../../store/algorithmDraft";
import type { ExecutionTraceDocument, PolicyCatalog, TraceRequest } from "../../../types/algorithms";
import { listPolicyModules, tracePolicy } from "../../../utils/algorithms/inspectClient";
import {
  buildFlow,
  coordinatesFromInstance,
  eventFields,
  formatRoutes,
  solutionRoutes,
  stageIds,
  visibleFlowRows,
} from "./flowModel";

export { buildFlow } from "./flowModel";

interface AlgorithmVisualizerProps {
  /** Fixture or saved document. Skips the live trace until the user presses Run. */
  initialDocument?: ExecutionTraceDocument;
}

type CatalogPhase = "idle" | "loading" | "ready" | "empty" | "error";

const SETTERS: Record<keyof DraftPatch, string> = {
  problem: "setProblem",
  graphSize: "setGraphSize",
  seed: "setSeed",
  packing: "setPacking",
  routing: "setRouting",
  model: "setModel",
  routeImprovement: "setRouteImprovement",
  acceptance: "setAcceptance",
  operators: "setOperators",
};

function text(value: unknown): string {
  return typeof value === "string" ? value : "";
}

function whole(value: unknown, fallback: number): number {
  const parsed = typeof value === "number" ? value : Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function applyDraftPatch(patch: DraftPatch): void {
  const state = useAlgorithmDraftStore.getState() as unknown as Record<string, unknown>;
  const setDraft = state.setDraft;
  if (typeof setDraft === "function") {
    setDraft(patch);
    return;
  }
  for (const [key, value] of Object.entries(patch)) {
    const setter = state[SETTERS[key as keyof DraftPatch]];
    if (typeof setter === "function") setter(value);
  }
}

function blockedTraceId(routing: string, model: string): string | null {
  if (model.trim()) return model.trim();
  const routeId = routing.trim();
  if (routeId && routeId !== "alns") return routeId;
  return null;
}

export function AlgorithmVisualizer({ initialDocument }: AlgorithmVisualizerProps) {
  const projectRoot = useAppStore((state) => state.projectRoot);
  const draft = useAlgorithmDraftStore();
  const problem = text(draft.problem) || "ptp";
  const graphSize = whole(draft.graphSize, 5);
  const seed = whole(draft.seed, 1);
  const packing = text(draft.packing);
  const routing = text(draft.routing);
  const model = text(draft.model);
  const routeImprovement = text(draft.routeImprovement);
  const acceptance = text(draft.acceptance);
  const operators = Array.isArray(draft.operators) ? draft.operators.filter((item) => typeof item === "string") : [];

  const [catalog, setCatalog] = useState<PolicyCatalog | null>(null);
  const [catalogPhase, setCatalogPhase] = useState<CatalogPhase>(projectRoot ? "loading" : "idle");
  const [catalogError, setCatalogError] = useState<string | null>(null);
  const [traceDocument, setTraceDocument] = useState<ExecutionTraceDocument | null>(initialDocument ?? null);
  const [expandedStageIds, setExpandedStageIds] = useState<Set<string>>(() => stageIds(initialDocument));
  const [expandOperators, setExpandOperators] = useState(false);
  const [solutionChangesOnly, setSolutionChangesOnly] = useState(false);
  const [selectedId, setSelectedId] = useState<string | null>(initialDocument?.events?.[0]?.id ?? null);
  const [running, setRunning] = useState(false);
  const [runNotice, setRunNotice] = useState<string | null>(null);
  const [runError, setRunError] = useState<string | null>(null);

  useEffect(() => {
    if (!projectRoot) {
      setCatalog(null);
      setCatalogPhase("idle");
      setCatalogError(null);
      return;
    }
    let cancel = false;
    setCatalogPhase("loading");
    setCatalogError(null);
    Promise.resolve()
      .then(() => listPolicyModules(projectRoot))
      .then((next) => {
        if (cancel) return;
        if (!next || typeof next !== "object") {
          throw new Error("Catalog command returned no document.");
        }
        setCatalog(next);
        setCatalogPhase(isCatalogEmpty(next) ? "empty" : "ready");
      })
      .catch((error: unknown) => {
        if (cancel) return;
        setCatalog(null);
        setCatalogError(error instanceof Error ? error.message : String(error));
        setCatalogPhase("error");
      });
    return () => {
      cancel = true;
    };
  }, [projectRoot]);

  useEffect(() => {
    setExpandedStageIds(stageIds(traceDocument));
    setExpandOperators(false);
    setSelectedId(traceDocument?.events?.[0]?.id ?? null);
  }, [traceDocument]);

  const rows = useMemo(
    () =>
      visibleFlowRows(buildFlow(traceDocument?.events), {
        expandedStageIds,
        solutionChangesOnly,
        expandOperators,
      }),
    [traceDocument, expandedStageIds, solutionChangesOnly, expandOperators]
  );

  useEffect(() => {
    const visible = rows.some((row) => row.kind === "event" && eventFields(row.event).id === selectedId);
    if (visible) return;
    const first = rows.find((row) => row.kind === "event");
    setSelectedId(first && first.kind === "event" ? eventFields(first.event).id : null);
  }, [rows, selectedId]);

  const selected = useMemo(() => {
    for (const event of traceDocument?.events ?? []) {
      if (eventFields(event).id === selectedId) return event;
    }
    return null;
  }, [traceDocument, selectedId]);

  const toggleStage = (id: string) => {
    setExpandedStageIds((current) => {
      const next = new Set(current);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const run = () => {
    const blocked = blockedTraceId(routing, model);
    if (blocked) {
      setRunError(null);
      setRunNotice(`${blocked} is not runnable from the trace command yet.`);
      return;
    }
    if (!projectRoot) {
      setRunNotice(null);
      setRunError("Project root is not set.");
      return;
    }
    if (routing !== "alns") {
      setRunError(null);
      setRunNotice("Select the alns routing policy to run a trace.");
      return;
    }
    const request: TraceRequest = {
      problem,
      graphSize,
      seed,
      acceptance: acceptance || "oi",
      packing: packing || null,
      routeImprovement: routeImprovement || null,
      maxIterations: 20,
    };
    setRunning(true);
    setRunNotice(null);
    setRunError(null);
    tracePolicy(projectRoot, request)
      .then((document) => {
        setTraceDocument(document);
      })
      .catch((error: unknown) => {
        setRunError(error instanceof Error ? error.message : String(error));
      })
      .finally(() => setRunning(false));
  };

  const finalRoutes = solutionRoutes(traceDocument?.final ?? null);

  return (
    <div className="h-full overflow-auto bg-noc-deep text-noc-textPrimary">
      <header className="px-6 py-4 bg-noc-panel border-b border-noc-border">
        <h1 className="text-lg font-semibold text-noc-textPrimary">Algorithm Visualizer</h1>
        <p className="text-xs text-noc-textMuted mt-1">
          Input to the final route. Algorithm Registry stays the metric comparison page.
        </p>
      </header>
      {traceDocument?.final ? (
        <p className="px-6 pt-4 text-sm text-noc-textPrimary" data-testid="final-route">
          Final route {formatRoutes(finalRoutes)}
        </p>
      ) : null}
      <div className="p-4 grid grid-cols-1 xl:grid-cols-[300px_minmax(0,1fr)_300px] gap-4 items-start">
        <PolicyDraftForm
          problem={problem}
          graphSize={graphSize}
          seed={seed}
          packing={packing}
          routing={routing}
          model={model}
          routeImprovement={routeImprovement}
          acceptance={acceptance}
          operators={operators}
          catalog={catalog}
          catalogPhase={catalogPhase}
          catalogError={catalogError}
          runNotice={runNotice}
          runError={runError}
          running={running}
          onPatch={applyDraftPatch}
          onRun={run}
        />
        <TraceFlow
          rows={rows}
          selectedId={selectedId}
          solutionChangesOnly={solutionChangesOnly}
          onSelect={setSelectedId}
          onToggleStage={toggleStage}
          expandedStageIds={expandedStageIds}
          onSolutionChangesOnly={setSolutionChangesOnly}
          onExpandOperators={() => setExpandOperators(true)}
        />
        <TraceInspector event={selected} coordinates={coordinatesFromInstance(traceDocument?.instance)} />
      </div>
    </div>
  );
}
