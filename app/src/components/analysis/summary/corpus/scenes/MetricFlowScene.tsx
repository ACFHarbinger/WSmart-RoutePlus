import { CORPUS_METRIC_LABEL } from "../../../../../types/corpus";
import type { CorpusHighlight, CorpusSeriesPoint } from "../../../../../types/corpus";
import type { CorpusSceneProps } from "../sceneProps";
import {
  CorpusSceneFrame,
  formatCorpusNumber,
  seriesDimmed,
  sharedIdentity,
  toggleHighlight,
} from "./CorpusSceneFrame";

interface FlowStrand {
  id: string;
  label: string;
  value: number;
  points: CorpusSeriesPoint[];
  identity: CorpusHighlight;
}

const STRAND_COLOR = ["#38bdf8", "#f59e0b", "#ef4444", "#818cf8", "#22c55e", "#e879f9"];

function finitePick(
  points: readonly CorpusSeriesPoint[],
  pick: (point: CorpusSeriesPoint) => number | null | undefined
): CorpusSeriesPoint[] {
  return points.filter((point) => {
    const value = pick(point);
    return value != null && Number.isFinite(value);
  });
}

function strandFrom(
  id: string,
  label: string,
  points: readonly CorpusSeriesPoint[],
  pick: (point: CorpusSeriesPoint) => number | null | undefined
): FlowStrand | null {
  const contributors = finitePick(points, pick);
  if (contributors.length === 0) return null;
  const value = contributors.reduce((sum, point) => sum + (pick(point) as number), 0);
  return {
    id,
    label,
    value,
    points: contributors,
    identity: sharedIdentity(contributors),
  };
}

function wasteStrands(points: readonly CorpusSeriesPoint[]): FlowStrand[] | null {
  const waste = points.some(
    (point) => point.metrics.kg != null || point.kgLost != null || point.metrics.overflows != null
  );
  if (!waste) return null;
  return [
    strandFrom("served", "served kg", points, (point) => point.metrics.kg),
    strandFrom("lost", "lost kg", points, (point) => point.kgLost),
    strandFrom("overflows", "overflows", points, (point) => point.metrics.overflows),
  ].filter((strand): strand is FlowStrand => strand != null);
}

function policyStrands(points: readonly CorpusSeriesPoint[]): FlowStrand[] {
  const groups = new Map<string, CorpusSeriesPoint[]>();
  for (const point of points) {
    const key = point.policy ?? "";
    const group = groups.get(key);
    if (group) group.push(point);
    else groups.set(key, [point]);
  }
  const strands: FlowStrand[] = [];
  for (const [key, group] of groups) {
    const strand = strandFrom(key || "unspecified", key || "unspecified", group, (point) => point.value);
    if (strand) strands.push(strand);
  }
  return strands;
}

function ribbonPath(x0: number, y0: number, h0: number, x1: number, y1: number, h1: number): string {
  const bend = (x1 - x0) / 2;
  return [
    `M ${x0} ${y0}`,
    `C ${x0 + bend} ${y0}, ${x1 - bend} ${y1}, ${x1} ${y1}`,
    `L ${x1} ${y1 + h1}`,
    `C ${x1 - bend} ${y1 + h1}, ${x0 + bend} ${y0 + h0}, ${x0} ${y0 + h0}`,
    "Z",
  ].join(" ");
}

export function MetricFlowScene({ view, metric, highlight, onHighlight }: CorpusSceneProps) {
  const strands = wasteStrands(view.points) ?? policyStrands(view.points);
  const magnitude = strands.reduce((sum, strand) => sum + Math.abs(strand.value), 0);
  const top = 28;
  const usable = 220;
  const gap = strands.length > 1 ? 8 : 0;
  const ribbonBudget = Math.max(0, usable - gap * Math.max(0, strands.length - 1));
  let cursor = top;

  return (
    <CorpusSceneFrame title="Metric flow" view={view} metric={metric}>
      <p className="text-xs text-canvas-muted">
        Ribbon width follows the present total. A missing strand is omitted.
      </p>
      {strands.length === 0 ? (
        <p className="text-xs text-canvas-muted">No present values.</p>
      ) : (
        <svg viewBox="0 0 640 280" className="w-full" role="group" aria-label="Metric flow">
          <text x={16} y={18} fill="currentColor" fontSize="11" className="text-canvas-muted">
            {wasteStrands(view.points) ? "kg" : CORPUS_METRIC_LABEL[metric]}
          </text>
          {strands.map((strand, index) => {
            const height =
              magnitude === 0 ? 0 : (Math.abs(strand.value) / magnitude) * ribbonBudget;
            const y = cursor;
            cursor += height + gap;
            const dim = seriesDimmed(highlight, strand.points);
            const color = STRAND_COLOR[index % STRAND_COLOR.length];
            const label = `${strand.label} ${formatCorpusNumber(strand.value)}`;
            return (
              <g
                key={strand.id}
                role="button"
                tabIndex={0}
                aria-label={label}
                aria-pressed={highlight != null && seriesDimmed(highlight, strand.points) === false}
                style={{ opacity: dim ? 0.35 : 1 }}
                onClick={() => onHighlight(toggleHighlight(highlight, strand.identity))}
                onKeyDown={(event) => {
                  if (event.key !== "Enter" && event.key !== " ") return;
                  event.preventDefault();
                  onHighlight(toggleHighlight(highlight, strand.identity));
                }}
              >
                <title>{strand.points.map((point) => point.runLabel).join(", ")}</title>
                {height > 0 && (
                  <path
                    d={ribbonPath(110, y, height, 430, y, height)}
                    fill={color}
                    opacity={0.85}
                  />
                )}
                <text x={444} y={y + Math.max(height, 12) / 2} fill="currentColor" fontSize="12" dominantBaseline="middle">
                  {label}
                </text>
              </g>
            );
          })}
        </svg>
      )}
    </CorpusSceneFrame>
  );
}
