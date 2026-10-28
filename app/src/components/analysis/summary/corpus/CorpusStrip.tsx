import { useMemo } from "react";
import { open } from "@tauri-apps/plugin-dialog";

import { useCorpusSceneState, useSimulationCorpusStore } from "../../../../store/simulationCorpus";
import {
  CORPUS_METRIC_LABEL,
  CORPUS_METRICS,
  type CorpusMetric,
  type CorpusScanResult,
} from "../../../../types/corpus";
import { groupCorpusRuns } from "../../../../utils/corpus/select";
import { CorpusRunList, nextSelectedRunIds } from "./CorpusRunList";

/** Corpus controls for the summary scenes. The day range does not move the monitor map. */
export function CorpusStrip() {
  const {
    roots,
    scan,
    loading,
    error,
    selectedRunIds,
    hiddenRunIds,
    metric,
    dayMin,
    dayMax,
    policies,
    cities,
    baselineMode,
    pinnedRunId,
    typedBaseline,
    addRoots,
    reload,
    hideRun,
    setMetric,
    setDayRange,
    setPolicies,
    setCities,
    setBaselineMode,
    setPinnedRunId,
    setTypedBaseline,
    setSelectedRunIds,
  } = useSimulationCorpusStore();
  const { view, highlight, setHighlight } = useCorpusSceneState();

  const runs = useMemo(() => {
    if (!scan) return [];
    return groupCorpusRuns(scan.files).filter((run) => !hiddenRunIds.includes(run.id));
  }, [scan, hiddenRunIds]);

  const bounds = useMemo(() => dayBounds(scan), [scan]);
  const policyOptions = useMemo(() => facetValues(scan, "policy"), [scan]);
  const cityOptions = useMemo(() => facetValues(scan, "city"), [scan]);

  async function addFolders(): Promise<void> {
    try {
      const picked = await open({ directory: true, multiple: true });
      const paths = pickedPaths(picked);
      if (paths.length === 0) return;
      await addRoots(paths);
    } catch (err) {
      reportDialogError(err);
    }
  }

  async function addFiles(): Promise<void> {
    try {
      const picked = await open({
        multiple: true,
        filters: [{ name: "Simulation JSON", extensions: ["json", "jsonl"] }],
      });
      const paths = pickedPaths(picked);
      if (paths.length === 0) return;
      await addRoots(paths);
    } catch (err) {
      reportDialogError(err);
    }
  }

  function toggleRun(id: string): void {
    setSelectedRunIds(
      nextSelectedRunIds(
        selectedRunIds,
        runs.map((run) => run.id),
        id
      )
    );
  }

  function toggleHighlight(id: string): void {
    if (highlight?.runId === id) setHighlight(null);
    else setHighlight({ runId: id });
  }

  return (
    <section className="card space-y-3" aria-label="Simulation corpus" aria-busy={loading}>
      <div className="flex flex-wrap items-center gap-2">
        <button type="button" className="btn-primary text-xs" onClick={() => void addFolders()}>
          Add result folders
        </button>
        <button type="button" className="btn-ghost text-xs" onClick={() => void addFiles()}>
          Add JSON file
        </button>
        <button
          type="button"
          className="btn-ghost text-xs"
          onClick={() => void reload()}
          disabled={loading}
        >
          Reload
        </button>
      </div>

      <p className="text-xs text-canvas-muted" data-testid="corpus-counts">
        {scan
          ? `Kept ${scan.kept}, skipped ${scan.skipped}, failed ${scan.failed.length}.`
          : "No corpus loaded."}
      </p>
      {roots.length > 0 ? (
        <p className="break-all text-xs text-canvas-muted">Roots: {roots.join(", ")}</p>
      ) : null}
      {loading ? <p className="text-xs text-canvas-muted">Scanning corpus…</p> : null}
      {error ? (
        <p className="text-xs text-red-300" role="alert">
          {error}
        </p>
      ) : null}

      {scan && scan.failed.length > 0 ? (
        <ul className="space-y-1 text-xs text-canvas-muted" aria-label="Failed files">
          {scan.failed.map((failure, index) => (
            <li key={`${failure.path}:${index}`}>
              {failure.path}: {failure.reason}
            </li>
          ))}
        </ul>
      ) : null}

      {scan && runs.length === 0 ? (
        <p className="text-sm text-canvas-muted">The corpus is empty.</p>
      ) : null}

      {runs.length > 0 ? (
        <CorpusRunList
          runs={runs}
          selectedRunIds={selectedRunIds}
          highlightRunId={highlight?.runId}
          onToggleRun={toggleRun}
          onHideRun={hideRun}
          onHighlightRun={toggleHighlight}
        />
      ) : null}

      <div className="flex flex-wrap items-end gap-3">
        <label className="flex flex-col gap-1 text-xs text-canvas-muted">
          Metric
          <select
            className="select-base w-36"
            value={metric}
            onChange={(event) => setMetric(event.target.value as CorpusMetric)}
          >
            {CORPUS_METRICS.map((id) => (
              <option key={id} value={id}>
                {CORPUS_METRIC_LABEL[id]}
              </option>
            ))}
          </select>
        </label>
        <label className="flex flex-col gap-1 text-xs text-canvas-muted">
          Day from
          <input
            className="input-base w-24"
            type="number"
            min={bounds.min ?? undefined}
            max={bounds.max ?? undefined}
            value={dayMin ?? ""}
            onChange={(event) => setDayRange(clampOptionalDay(event.target.value, bounds), dayMax)}
          />
        </label>
        <label className="flex flex-col gap-1 text-xs text-canvas-muted">
          Day to
          <input
            className="input-base w-24"
            type="number"
            min={bounds.min ?? undefined}
            max={bounds.max ?? undefined}
            value={dayMax ?? ""}
            onChange={(event) => setDayRange(dayMin, clampOptionalDay(event.target.value, bounds))}
          />
        </label>
        <label className="flex flex-col gap-1 text-xs text-canvas-muted">
          Policies
          <select
            className="select-base w-36"
            multiple
            value={policies}
            onChange={(event) => setPolicies(selectedValues(event.target))}
          >
            {policyOptions.map((policy) => (
              <option key={policy} value={policy}>
                {policy}
              </option>
            ))}
          </select>
        </label>
        <label className="flex flex-col gap-1 text-xs text-canvas-muted">
          Cities
          <select
            className="select-base w-36"
            multiple
            value={cities}
            onChange={(event) => setCities(selectedValues(event.target))}
          >
            {cityOptions.map((city) => (
              <option key={city} value={city}>
                {city}
              </option>
            ))}
          </select>
        </label>
      </div>
      <p className="text-xs text-canvas-muted">
        An empty policy or city selection means all. An empty day field means no bound.
      </p>

      <div className="flex flex-wrap items-end gap-3">
        <label className="flex items-center gap-2 text-xs">
          <input
            type="radio"
            name="corpus-baseline-mode"
            checked={baselineMode === "run"}
            onChange={() => setBaselineMode("run")}
          />
          Pin a run
        </label>
        <label className="flex flex-col gap-1 text-xs text-canvas-muted">
          Pinned run
          <select
            className="select-base w-44"
            value={pinnedRunId ?? ""}
            onChange={(event) => setPinnedRunId(event.target.value || null)}
          >
            <option value="">Select a run</option>
            {runs.map((run) => (
              <option key={run.id} value={run.id}>
                {run.label}
              </option>
            ))}
          </select>
        </label>
        <label className="flex items-center gap-2 text-xs">
          <input
            type="radio"
            name="corpus-baseline-mode"
            checked={baselineMode === "value"}
            onChange={() => setBaselineMode("value")}
          />
          Type a value
        </label>
        <label className="flex flex-col gap-1 text-xs text-canvas-muted">
          Typed baseline
          <input
            className="input-base w-28"
            type="number"
            step="any"
            value={typedBaseline ?? ""}
            onChange={(event) => setTypedBaseline(parseOptionalNumber(event.target.value))}
          />
        </label>
      </div>
      <p className="text-xs text-canvas-muted" data-testid="corpus-baseline">
        {baselineLine(view?.baseline ?? null, view?.baselineLabel ?? "")}
      </p>
    </section>
  );
}

function baselineLine(baseline: number | null, baselineLabel: string): string {
  if (baseline == null) return "Baseline: No baseline";
  return `Baseline: ${baselineLabel} ${formatBaseline(baseline)}`;
}

function formatBaseline(value: number): string {
  const rounded = Math.round(value * 1000) / 1000;
  return String(rounded);
}

/** An empty field stays null. It is not coerced to zero. */
function parseOptionalNumber(raw: string): number | null {
  if (raw.trim() === "") return null;
  const value = Number(raw);
  return Number.isFinite(value) ? value : null;
}

function clampOptionalDay(
  raw: string,
  bounds: { min: number | null; max: number | null }
): number | null {
  const parsed = parseOptionalNumber(raw);
  if (parsed == null) return null;
  if (bounds.min != null && parsed < bounds.min) return bounds.min;
  if (bounds.max != null && parsed > bounds.max) return bounds.max;
  return parsed;
}

function pickedPaths(picked: string | readonly string[] | null): string[] {
  if (typeof picked === "string") return picked ? [picked] : [];
  if (Array.isArray(picked)) return picked.filter((path) => path.length > 0);
  return [];
}

function selectedValues(select: HTMLSelectElement): string[] {
  return [...select.selectedOptions].map((option) => option.value);
}

function dayBounds(scan: CorpusScanResult | null): { min: number | null; max: number | null } {
  if (!scan) return { min: null, max: null };
  let min: number | null = null;
  let max: number | null = null;
  for (const file of scan.files) {
    for (const day of file.days) {
      if (day.day == null || !Number.isFinite(day.day)) continue;
      if (min == null || day.day < min) min = day.day;
      if (max == null || day.day > max) max = day.day;
    }
  }
  return { min, max };
}

function facetValues(scan: CorpusScanResult | null, key: "policy" | "city"): string[] {
  if (!scan) return [];
  const values = new Set<string>();
  for (const file of scan.files) {
    for (const day of file.days) {
      const value = day[key];
      if (value == null || value === "") continue;
      values.add(value);
    }
  }
  return [...values].sort((a, b) => a.localeCompare(b));
}

function reportDialogError(err: unknown): void {
  const message = err instanceof Error && err.message ? err.message : "Could not open the file dialog";
  useSimulationCorpusStore.setState({ error: message, loading: false });
}
