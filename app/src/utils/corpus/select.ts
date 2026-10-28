import type {
  CorpusDay,
  CorpusFileRecord,
  CorpusFilters,
  CorpusMetric,
  CorpusRun,
  CorpusScanResult,
  CorpusSeriesPoint,
  CorpusView,
  GeneratedLayoutNode,
} from "../../types/corpus";

function normalizePath(path: string): string {
  return path.replace(/\\/g, "/").replace(/\/+$/, "");
}

function parentDir(path: string): string {
  const norm = normalizePath(path);
  const index = norm.lastIndexOf("/");
  return index <= 0 ? "" : norm.slice(0, index);
}

function baseName(path: string): string {
  const norm = normalizePath(path);
  const index = norm.lastIndexOf("/");
  const name = index < 0 ? norm : norm.slice(index + 1);
  return name.replace(/\.(jsonl|json)$/i, "");
}

function finite(value: number | null | undefined): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function dayHasCoordinates(day: CorpusDay): boolean {
  return finite(day.lat) && finite(day.lon);
}

/** Group files into runs. One directory of JSON files is one run. One chosen file is a partial run. */
export function groupCorpusRuns(files: readonly CorpusFileRecord[]): CorpusRun[] {
  const grouped = new Map<string, CorpusRun>();

  for (const file of files) {
    const directory = file.partial ? "" : parentDir(file.path);
    const id = file.partial ? `file:${normalizePath(file.path)}` : `dir:${directory}`;
    const scenarioDirectory = parentDir(file.partial ? file.path : directory);
    const existing = grouped.get(id);
    const coordinates = file.days.some(dayHasCoordinates);
    if (existing) {
      existing.files.push(file.path);
      existing.hasCoordinates = existing.hasCoordinates || coordinates;
      continue;
    }
    grouped.set(id, {
      id,
      label: file.partial ? baseName(file.path) : baseName(directory),
      scenarioId: scenarioDirectory || "scenario",
      scenarioLabel: scenarioDirectory ? baseName(scenarioDirectory) : "Scenario",
      directory,
      partial: file.partial,
      files: [file.path],
      hasCoordinates: coordinates,
    });
  }

  return [...grouped.values()].sort((a, b) => a.label.localeCompare(b.label));
}

function runForFile(runs: readonly CorpusRun[], file: CorpusFileRecord): CorpusRun | undefined {
  const id = file.partial ? `file:${normalizePath(file.path)}` : `dir:${parentDir(file.path)}`;
  return runs.find((run) => run.id === id);
}

function mean(values: readonly number[]): number | null {
  if (values.length === 0) return null;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function passesDay(day: CorpusDay, filters: CorpusFilters): boolean {
  if (day.day == null) return filters.dayMin == null && filters.dayMax == null && filters.playbackDay == null;
  if (filters.dayMin != null && day.day < filters.dayMin) return false;
  if (filters.dayMax != null && day.day > filters.dayMax) return false;
  if (filters.playbackDay != null && day.day !== filters.playbackDay) return false;
  return true;
}

function passesFacet(day: CorpusDay, filters: CorpusFilters): boolean {
  if (filters.policies.length > 0 && !filters.policies.includes(day.policy ?? "")) return false;
  if (filters.cities.length > 0 && !filters.cities.includes(day.city ?? "")) return false;
  return true;
}

function pointFromDay(run: CorpusRun, day: CorpusDay, metric: CorpusMetric): CorpusSeriesPoint {
  const value = day.metrics[metric];
  return {
    runId: run.id,
    runLabel: run.label,
    scenarioId: run.scenarioId,
    scenarioLabel: run.scenarioLabel,
    partial: run.partial,
    day: day.day,
    policy: day.policy,
    city: day.city,
    value: finite(value) ? value : null,
    metrics: day.metrics,
    kgLost: finite(day.kgLost) ? day.kgLost : null,
    lat: finite(day.lat) ? day.lat : null,
    lon: finite(day.lon) ? day.lon : null,
    bins: day.bins,
    tour: day.tour,
    tourIndices: day.tourIndices,
    routeBins: day.routeBins,
    hasCoordinates: dayHasCoordinates(day),
  };
}

function baselineValue(
  filters: CorpusFilters,
  pointsByRun: ReadonlyMap<string, CorpusSeriesPoint[]>
): { baseline: number | null; baselineLabel: string } {
  if (filters.baselineMode === "value") {
    return {
      baseline: finite(filters.typedBaseline) ? filters.typedBaseline : null,
      baselineLabel: "Typed baseline",
    };
  }
  const pinned = filters.pinnedRunId;
  if (!pinned) return { baseline: null, baselineLabel: "Pinned run" };
  const points = pointsByRun.get(pinned) ?? [];
  const values = points.map((point) => point.value).filter(finite);
  const runLabel = points[0]?.runLabel ?? "Pinned run";
  return { baseline: mean(values), baselineLabel: runLabel };
}

export function buildCorpusView(scan: CorpusScanResult, filters: CorpusFilters): CorpusView {
  const runs = groupCorpusRuns(scan.files).filter((run) => !filters.hiddenRunIds.includes(run.id));
  const visible = new Set(runs.map((run) => run.id));
  const selected =
    filters.selectedRunIds.length === 0
      ? visible
      : new Set(filters.selectedRunIds.filter((id) => visible.has(id)));

  const pointsByRun = new Map<string, CorpusSeriesPoint[]>();
  const points: CorpusSeriesPoint[] = [];

  for (const file of scan.files) {
    const run = runForFile(runs, file);
    if (!run) continue;
    for (const day of file.days) {
      if (!passesDay(day, filters) || !passesFacet(day, filters)) continue;
      const point = pointFromDay(run, day, filters.metric);
      if (filters.baselineMode === "run" && run.id === filters.pinnedRunId) {
        const bucket = pointsByRun.get(run.id);
        if (bucket) bucket.push(point);
        else pointsByRun.set(run.id, [point]);
      }
      if (!selected.has(run.id)) continue;
      points.push(point);
    }
  }

  const shownRuns = runs.filter((run) => selected.has(run.id));
  const { baseline, baselineLabel } = baselineValue(filters, pointsByRun);

  return {
    runs: shownRuns,
    points,
    baseline,
    baselineLabel,
    kept: scan.kept,
    skipped: scan.skipped,
    failed: scan.failed,
    dayLevelOnly: points.every((point) => point.bins.length === 0),
  };
}

/** Plane positions for runs that have no geographic coordinates. Not longitude or latitude. */
export function generatedLayout(ids: readonly { id: string; label: string }[]): GeneratedLayoutNode[] {
  const columns = Math.max(1, Math.ceil(Math.sqrt(ids.length)));
  const origin = (columns - 1) / 2;
  return ids.map((item, index) => ({
    id: item.id,
    label: item.label,
    x: (index % columns) - origin,
    y: Math.floor(index / columns),
  }));
}

/** Tab-separated table for the copy button. Blank cells are gaps. */
export function corpusTableTsv(view: CorpusView, metric: CorpusMetric): string {
  const header = ["run", "scenario", "day", "policy", "city", metric, "baseline"].join("\t");
  const rows = view.points.map((point) =>
    [
      point.runLabel,
      point.scenarioLabel,
      point.day ?? "",
      point.policy ?? "",
      point.city ?? "",
      point.value ?? "",
      view.baseline ?? "",
    ].join("\t")
  );
  return [header, ...rows].join("\n");
}
