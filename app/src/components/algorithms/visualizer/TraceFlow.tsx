import type { KeyboardEvent } from "react";
import {
  eventFields,
  eventInterior,
  eventKind,
  eventName,
  type FlowRow,
} from "../../../pages/algorithms/visualizer/flowModel";

interface TraceFlowProps {
  rows: FlowRow[];
  selectedId: string | null;
  solutionChangesOnly: boolean;
  onSelect: (id: string) => void;
  onToggleStage: (id: string) => void;
  expandedStageIds: ReadonlySet<string>;
  onSolutionChangesOnly: (value: boolean) => void;
  onExpandOperators: () => void;
}

function rowIds(rows: FlowRow[]): string[] {
  const ids: string[] = [];
  for (const row of rows) {
    if (row.kind === "event") ids.push(eventFields(row.event).id);
  }
  return ids;
}

export function TraceFlow({
  rows,
  selectedId,
  solutionChangesOnly,
  onSelect,
  onToggleStage,
  expandedStageIds,
  onSolutionChangesOnly,
  onExpandOperators,
}: TraceFlowProps) {
  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key !== "ArrowDown" && event.key !== "ArrowUp") return;
    const ids = rowIds(rows);
    if (!ids.length) return;
    event.preventDefault();
    const current = selectedId ? ids.indexOf(selectedId) : -1;
    const next =
      event.key === "ArrowDown"
        ? Math.min(ids.length - 1, current + 1)
        : Math.max(0, current < 0 ? 0 : current - 1);
    const id = ids[next];
    if (id) onSelect(id);
  };

  return (
    <section className="bg-noc-card border border-noc-border rounded-xl min-h-0 flex flex-col">
      <div className="px-3 py-2 border-b border-noc-border flex items-center justify-between gap-2">
        <h2 className="text-sm font-medium text-noc-textPrimary">Execution flow</h2>
        <label className="flex items-center gap-2 text-xs text-noc-textMuted">
          <input
            type="checkbox"
            checked={solutionChangesOnly}
            onChange={(event) => onSolutionChangesOnly(event.target.checked)}
          />
          Solution changes only
        </label>
      </div>
      <div
        role="listbox"
        aria-label="Execution flow"
        tabIndex={0}
        onKeyDown={onKeyDown}
        className="p-2 overflow-auto max-h-[640px] outline-none focus:ring-1 focus:ring-noc-cyan"
      >
        {rows.length === 0 ? (
          <p className="text-xs text-noc-textMuted px-1 py-2">No trace loaded.</p>
        ) : (
          rows.map((row, index) => {
            if (row.kind === "gap") {
              return (
                <div
                  key={`gap-${index}`}
                  className="my-1 px-2 py-2 rounded-lg border border-dashed border-noc-border text-xs text-noc-textMuted"
                  data-testid="operator-gap"
                >
                  {row.hiddenCount} operator events hidden
                  <button
                    type="button"
                    className="ml-2 text-noc-cyan hover:underline"
                    onClick={onExpandOperators}
                  >
                    Expand all
                  </button>
                </div>
              );
            }
            const id = eventFields(row.event).id;
            const kind = eventKind(row.event);
            const interior = eventInterior(row.event);
            const selected = id === selectedId;
            const isStage = kind === "stage";
            const expanded = expandedStageIds.has(id);
            return (
              <div key={id} style={{ marginLeft: row.depth * 12 }} className="mb-1">
                <div className="flex items-stretch gap-1">
                  {isStage ? (
                    <button
                      type="button"
                      aria-label={expanded ? `Collapse ${eventName(row.event)}` : `Expand ${eventName(row.event)}`}
                      className="px-1 text-xs text-noc-textMuted border border-noc-border rounded bg-noc-deep"
                      onClick={() => onToggleStage(id)}
                    >
                      {expanded ? "−" : "+"}
                    </button>
                  ) : null}
                  <button
                    type="button"
                    role="option"
                    aria-selected={selected}
                    tabIndex={-1}
                    onClick={() => onSelect(id)}
                    className={`flex-1 text-left rounded-lg border px-2 py-1.5 ${
                      selected
                        ? "bg-noc-deep border-noc-cyan text-noc-cyan"
                        : "bg-noc-deep border-noc-border text-noc-textPrimary"
                    }`}
                  >
                    <span>{eventName(row.event)}</span>
                    <span className="ml-2 text-[10px] uppercase tracking-wide text-noc-textMuted">{kind}</span>
                    {interior === "not_instrumented" ? (
                      <span className="ml-2 text-[10px] uppercase text-noc-amber">not_instrumented</span>
                    ) : null}
                  </button>
                </div>
                {interior === "not_instrumented" ? (
                  <pre
                    className="mt-1 text-[11px] leading-4 text-noc-textMuted bg-noc-deep border border-noc-border rounded-lg p-2 overflow-auto"
                    data-testid={`raw-stage-${id}`}
                  >
                    {JSON.stringify(row.event, null, 2)}
                  </pre>
                ) : null}
              </div>
            );
          })
        )}
      </div>
    </section>
  );
}
