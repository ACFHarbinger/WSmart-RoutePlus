import type { CorpusHighlight, CorpusSeriesPoint } from "../../../../../types/corpus";
import type { CorpusSceneProps } from "../sceneProps";
import {
  CorpusSceneFrame,
  formatCorpusNumber,
  seriesDimmed,
  sharedIdentity,
  toggleHighlight,
} from "./CorpusSceneFrame";

interface DotMark {
  key: string;
  name: string;
  color: string;
  identity: CorpusHighlight;
  dim: boolean;
}

function fillColor(fill: number | null): string {
  if (fill == null) return "#64748b";
  const clamped = Math.min(1, Math.max(0, fill));
  const hue = Math.round(200 - clamped * 200);
  return `hsl(${hue} 75% 52%)`;
}

function binMarks(points: readonly CorpusSeriesPoint[], highlight: CorpusSceneProps["highlight"]): DotMark[] {
  const marks: DotMark[] = [];
  points.forEach((point, pointIndex) => {
    point.bins.forEach((bin, binIndex) => {
      const fill = typeof bin.fill === "number" && Number.isFinite(bin.fill) ? bin.fill : null;
      marks.push({
        key: `${point.runId}-${pointIndex}-${bin.binId}-${binIndex}`,
        name: fill == null ? `${point.runLabel} ${bin.binId} gap` : `${point.runLabel} ${bin.binId} fill ${formatCorpusNumber(fill)}`,
        color: fillColor(fill),
        identity: sharedIdentity([point], point.runId),
        dim: seriesDimmed(highlight, [point]),
      });
    });
  });
  return marks;
}

function dayMarks(points: readonly CorpusSeriesPoint[], highlight: CorpusSceneProps["highlight"]): DotMark[] {
  return points.map((point, index) => ({
    key: `${point.runId}-${index}`,
    name: point.value == null ? `${point.runLabel} gap` : `${point.runLabel} ${formatCorpusNumber(point.value)}`,
    color: point.value == null ? "#64748b" : "#38bdf8",
    identity: sharedIdentity([point], point.runId),
    dim: seriesDimmed(highlight, [point]),
  }));
}

function DotField({
  marks,
  highlight,
  onHighlight,
}: {
  marks: readonly DotMark[];
  highlight: CorpusSceneProps["highlight"];
  onHighlight: CorpusSceneProps["onHighlight"];
}) {
  const columns = Math.max(1, Math.ceil(Math.sqrt(marks.length)));
  return (
    <div
      className="grid gap-2"
      style={{ gridTemplateColumns: `repeat(${columns}, minmax(0, 1fr))` }}
    >
      {marks.map((mark) => (
        <button
          key={mark.key}
          type="button"
          className="flex items-center gap-1 text-left text-[10px] text-canvas-muted"
          style={{ opacity: mark.dim ? 0.35 : 1 }}
          aria-label={mark.name}
          aria-pressed={highlight != null && mark.dim === false && (highlight.runId != null || highlight.policy != null || highlight.city != null)}
          onClick={() => onHighlight(toggleHighlight(highlight, mark.identity))}
        >
          <span className="inline-block h-3 w-3 shrink-0 rounded-full" style={{ background: mark.color }} />
          <span className="truncate">{mark.name}</span>
        </button>
      ))}
    </div>
  );
}

export function DayBinFieldScene({ view, metric, highlight, onHighlight }: CorpusSceneProps) {
  const hasBins = view.points.some((point) => point.bins.length > 0);
  const dayLevel = view.dayLevelOnly || !hasBins;

  return (
    <CorpusSceneFrame title="Day by bin" view={view} metric={metric}>
      {hasBins && (
        <>
          <p className="text-xs text-canvas-muted">One dot per bin. Color follows fill. A missing fill is a gap.</p>
          <DotField marks={binMarks(view.points, highlight)} highlight={highlight} onHighlight={onHighlight} />
        </>
      )}
      {dayLevel && (
        <>
          <p className="text-xs text-canvas-muted">Day-level log: one dot per day.</p>
          <DotField marks={dayMarks(view.points, highlight)} highlight={highlight} onHighlight={onHighlight} />
        </>
      )}
    </CorpusSceneFrame>
  );
}
