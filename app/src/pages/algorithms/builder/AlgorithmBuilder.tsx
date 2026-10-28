import { useEffect, useState, type ReactNode } from "react";
import { ModuleInspector } from "../../../components/algorithms/builder/ModuleInspector";
import { BuilderCanvas } from "../../../components/algorithms/builder/BuilderCanvas";
import { ModulePalette } from "../../../components/algorithms/builder/ModulePalette";
import { useAppStore } from "../../../store/app";
import { useAlgorithmDraftStore } from "../../../store/algorithmDraft";
import type { PolicyCatalog } from "../../../types/algorithms";
import { listPolicyModules } from "../../../utils/algorithms/inspectClient";
import { readDraft, writeDraft } from "./draftBridge";
import { normFamily } from "./linkRules";
import { exportCompositionYaml, modulesWithoutRunner } from "./exportCompositionYaml";
import { useBuilderGraph } from "./useBuilderGraph";

const PROBLEMS = ["ptp", "cptp", "cptptc", "cptptwtc", "cvrp"] as const;

interface AlgorithmBuilderProps {
  initialCatalog?: PolicyCatalog;
}

export function AlgorithmBuilder({ initialCatalog }: AlgorithmBuilderProps = {}) {
  const projectRoot = useAppStore((s) => s.projectRoot);
  const problem = useAlgorithmDraftStore((s) => s.problem);
  const graphSize = useAlgorithmDraftStore((s) => s.graphSize);
  const seed = useAlgorithmDraftStore((s) => s.seed);
  const [catalog, setCatalog] = useState<PolicyCatalog | null>(initialCatalog ?? null);
  const [catalogError, setCatalogError] = useState<string | null>(null);
  const graph = useBuilderGraph();

  useEffect(() => {
    if (initialCatalog) {
      setCatalog(initialCatalog);
      return;
    }
    if (!projectRoot) return;
    let cancelled = false;
    listPolicyModules(projectRoot)
      .then((document) => {
        if (!cancelled) setCatalog(document);
      })
      .catch((err: unknown) => {
        if (!cancelled) setCatalogError(err instanceof Error ? err.message : String(err));
      });
    return () => {
      cancelled = true;
    };
  }, [initialCatalog, projectRoot]);

  const yaml = exportCompositionYaml({
    problem: problem || "ptp",
    graphSize: typeof graphSize === "number" ? graphSize : 20,
    seed: typeof seed === "number" ? seed : 1,
    nodes: graph.nodes,
    edges: graph.edges,
  });
  const missingRunner = modulesWithoutRunner(graph.nodes);
  const hasRouting = graph.nodes.some((node) => normFamily(node.family) === "routing");
  const exportDisabled = missingRunner.length > 0 || !hasRouting;
  const runnerText =
    missingRunner.length > 1
      ? `${missingRunner.join(", ")} have no runner`
      : missingRunner.length === 1
        ? `${missingRunner[0]} has no runner`
        : null;

  const patchDraft = (patch: Record<string, unknown>) => {
    readDraft().setDraft(patch);
  };

  const openVisualizer = () => {
    writeDraft(graph.nodes, graph.edges);
    useAppStore.getState().setMode("algorithm_visualizer" as never);
  };

  const copyYaml = async () => {
    try {
      await navigator.clipboard.writeText(yaml);
    } catch {
      /* The textarea stays selectable when the clipboard is blocked. */
    }
  };

  return (
    <div className="flex flex-col gap-3 text-noc-textPrimary">
      <header className="flex flex-wrap items-end justify-between gap-3 rounded-xl border border-noc-border bg-noc-panel px-4 py-3">
        <div>
          <h1 className="text-lg font-semibold">Algorithm Builder</h1>
          <p className="text-xs text-noc-textMuted">Compose registered modules. The builder does not create new solvers.</p>
        </div>
        <div className="flex flex-wrap items-end gap-2">
          <Field label="Problem">
            <select
              className="rounded border border-noc-border bg-noc-deep px-2 py-1 text-sm"
              value={problem || "ptp"}
              onChange={(event) => patchDraft({ problem: event.target.value })}
            >
              {PROBLEMS.map((id) => (
                <option key={id} value={id}>
                  {id}
                </option>
              ))}
            </select>
          </Field>
          <Field label="Graph size">
            <input
              type="number"
              min={1}
              className="w-24 rounded border border-noc-border bg-noc-deep px-2 py-1 text-sm"
              value={typeof graphSize === "number" ? graphSize : 20}
              onChange={(event) => patchDraft({ graphSize: Number(event.target.value) })}
            />
          </Field>
          <Field label="Seed">
            <input
              type="number"
              className="w-24 rounded border border-noc-border bg-noc-deep px-2 py-1 text-sm"
              value={typeof seed === "number" ? seed : 1}
              onChange={(event) => patchDraft({ seed: Number(event.target.value) })}
            />
          </Field>
          <button
            type="button"
            className="rounded border border-noc-border px-3 py-1.5 text-xs"
            onClick={() => {
              readDraft().clear();
              graph.clearCanvas();
            }}
          >
            Clear
          </button>
          <button
            type="button"
            className="rounded border border-noc-cyan px-3 py-1.5 text-xs text-noc-cyan"
            onClick={openVisualizer}
          >
            Open in visualizer
          </button>
        </div>
      </header>

      {catalogError && <p className="text-xs text-noc-crit">{catalogError}</p>}
      {graph.alert && (
        <p role="alert" className="rounded-lg border border-noc-crit/40 bg-noc-card px-3 py-2 text-xs text-noc-crit">
          {graph.alert}
        </p>
      )}

      <div className="flex min-h-[560px] overflow-hidden rounded-xl border border-noc-border">
        <ModulePalette catalog={catalog} onAdd={graph.addEntry} />
        <BuilderCanvas
          nodes={graph.nodes}
          edges={graph.edges}
          pending={graph.pending}
          selectedUid={graph.selectedUid}
          onSelect={graph.setSelectedUid}
          onPort={graph.onPort}
          onConnect={graph.onConnectButton}
          onRemove={graph.removeNode}
        />
        <ModuleInspector
          catalog={catalog}
          node={graph.nodes.find((node) => node.uid === graph.selectedUid) ?? null}
          onParameter={graph.setParameter}
          onRemove={graph.removeNode}
        />
      </div>

      <section className="rounded-xl border border-noc-border bg-noc-panel p-3 space-y-2">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <h2 className="text-xs font-semibold uppercase tracking-wide text-noc-textMuted">Export</h2>
          <div className="flex gap-2">
            <button
              type="button"
              className="rounded border border-noc-border px-3 py-1 text-xs disabled:cursor-not-allowed disabled:opacity-40"
              disabled={exportDisabled}
              onClick={() => void copyYaml()}
            >
              Export
            </button>
            <button
              type="button"
              className="rounded border border-noc-border px-3 py-1 text-xs"
              onClick={() => void copyYaml()}
            >
              Copy
            </button>
          </div>
        </div>
        {runnerText && <p className="text-xs text-noc-crit">{runnerText}</p>}
        {!hasRouting && missingRunner.length === 0 && (
          <p className="text-xs text-noc-textMuted">Add a routing module before export.</p>
        )}
        <textarea
          data-testid="composition-yaml"
          readOnly
          className="h-40 w-full rounded border border-noc-border bg-noc-deep p-2 font-mono text-xs text-noc-textPrimary"
          value={yaml}
          aria-label="Composition YAML"
        />
      </section>
    </div>
  );
}

function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label className="flex flex-col gap-1 text-[11px] text-noc-textMuted">
      {label}
      {children}
    </label>
  );
}
