import type { CorpusRun } from "../../../../types/corpus";

/**
 * `buildCorpusView` treats an empty selection as every visible run.
 * A non-matching id is the unchecked-all state.
 */
export const CORPUS_NONE_SELECTED = "__corpus_none__";

interface ScenarioGroup {
  scenarioId: string;
  scenarioLabel: string;
  runs: CorpusRun[];
}

export function runIsChecked(
  selectedRunIds: readonly string[],
  visibleIds: readonly string[],
  id: string
): boolean {
  if (selectedRunIds.length === 0) return visibleIds.includes(id);
  return selectedRunIds.includes(id);
}

export function nextSelectedRunIds(
  selectedRunIds: readonly string[],
  visibleIds: readonly string[],
  id: string
): string[] {
  const current =
    selectedRunIds.length === 0
      ? [...visibleIds]
      : selectedRunIds.filter((runId) => visibleIds.includes(runId));
  const next = current.includes(id) ? current.filter((runId) => runId !== id) : [...current, id];
  return next.length === 0 ? [CORPUS_NONE_SELECTED] : next;
}

function groupByScenario(runs: readonly CorpusRun[]): ScenarioGroup[] {
  const groups = new Map<string, ScenarioGroup>();
  for (const run of runs) {
    const existing = groups.get(run.scenarioId);
    if (existing) {
      existing.runs.push(run);
      continue;
    }
    groups.set(run.scenarioId, {
      scenarioId: run.scenarioId,
      scenarioLabel: run.scenarioLabel,
      runs: [run],
    });
  }
  return [...groups.values()].sort((a, b) => a.scenarioLabel.localeCompare(b.scenarioLabel));
}

interface CorpusRunListProps {
  runs: readonly CorpusRun[];
  selectedRunIds: readonly string[];
  highlightRunId?: string;
  onToggleRun: (id: string) => void;
  onHideRun: (id: string) => void;
  onHighlightRun: (id: string) => void;
}

/** Runs under one scenario label. Several checked runs are one merged comparison. */
export function CorpusRunList({
  runs,
  selectedRunIds,
  highlightRunId,
  onToggleRun,
  onHideRun,
  onHighlightRun,
}: CorpusRunListProps) {
  const visibleIds = runs.map((run) => run.id);
  const groups = groupByScenario(runs);

  return (
    <div className="space-y-3">
      {groups.map((group) => (
        <fieldset key={group.scenarioId} className="space-y-1">
          <legend className="text-xs font-semibold uppercase tracking-wider text-canvas-muted">
            {group.scenarioLabel}
          </legend>
          {group.runs.map((run) => {
            const highlighted = highlightRunId === run.id;
            return (
              <div key={run.id} className="flex flex-wrap items-center gap-2">
                <input
                  type="checkbox"
                  aria-label={run.label}
                  checked={runIsChecked(selectedRunIds, visibleIds, run.id)}
                  onChange={() => onToggleRun(run.id)}
                />
                <button
                  type="button"
                  className="btn-ghost px-2 py-1 text-xs"
                  aria-pressed={highlighted}
                  onClick={() => onHighlightRun(run.id)}
                >
                  {run.label}
                </button>
                {run.partial ? <span className="text-xs text-canvas-muted">partial</span> : null}
                <button
                  type="button"
                  className="btn-ghost px-2 py-1 text-xs"
                  aria-label={`Hide ${run.label}`}
                  onClick={() => onHideRun(run.id)}
                >
                  Hide
                </button>
              </div>
            );
          })}
        </fieldset>
      ))}
    </div>
  );
}
