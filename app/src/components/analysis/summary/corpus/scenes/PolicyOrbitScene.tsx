import type { OrbitalBubble } from "../../../../ui/orion/OrionOrbitalBubbleChart";
import { OrionOrbitalBubbleChart } from "../../../../ui/orion/OrionOrbitalBubbleChart";
import { CORPUS_METRIC_LABEL } from "../../../../../types/corpus";
import type { CorpusHighlight } from "../../../../../types/corpus";
import type { CorpusSceneProps } from "../sceneProps";
import {
  CorpusSceneFrame,
  formatCorpusNumber,
  presentMean,
  runsInView,
  seriesDimmed,
  sharedIdentity,
  toggleHighlight,
} from "./CorpusSceneFrame";

/** Bubble area tracks |mean|. Radius is the square root of that area, clamped to the orbit chart. */
function bubbleRadius(mean: number, maxAbs: number): number {
  const span = maxAbs > 0 ? maxAbs : 1;
  return 10 + Math.sqrt(Math.abs(mean) / span) * 22;
}

export function PolicyOrbitScene({ view, metric, highlight, onHighlight }: CorpusSceneProps) {
  const metricLabel = CORPUS_METRIC_LABEL[metric];
  const gaps: { id: string; label: string }[] = [];
  const prepared: {
    id: string;
    label: string;
    mean: number;
    identity: CorpusHighlight;
    dim: boolean;
    days: number;
  }[] = [];

  for (const run of runsInView(view)) {
    const rows = view.points.filter((point) => point.runId === run.id);
    const mean = presentMean(rows.map((point) => point.value));
    if (mean == null) {
      gaps.push(run);
      continue;
    }
    prepared.push({
      id: run.id,
      label: run.label,
      mean,
      identity: sharedIdentity(rows, run.id),
      dim: seriesDimmed(highlight, rows),
      days: rows.length,
    });
  }

  const maxAbs = prepared.reduce((max, item) => Math.max(max, Math.abs(item.mean)), 0);
  const bubbles: OrbitalBubble[] = prepared.map((item, index) => ({
    id: item.id,
    label: item.label,
    orbitIdx: index % 3,
    angleDeg: (360 * index) / prepared.length,
    radius: bubbleRadius(item.mean, maxAbs),
    value: Number(formatCorpusNumber(item.mean)),
    unit: metricLabel,
    tone: item.dim ? "idle" : "cyan",
    metrics: { Days: item.days },
  }));
  const identities = new Map(prepared.map((item) => [item.id, item.identity]));

  return (
    <CorpusSceneFrame title="Policy orbit" view={view} metric={metric}>
      <p className="text-xs text-canvas-muted">
        One bubble per run. Area is the mean of present {metricLabel} values. A run with only gaps
        is listed and not drawn.
      </p>
      {bubbles.length > 0 && (
        <OrionOrbitalBubbleChart
          bubbles={bubbles}
          title="Run means"
          subtitle="Bubble area is the mean of present values"
          centerLabel={metricLabel}
          centerValue={String(bubbles.length)}
          selectedBubbleId={highlight?.runId ?? ""}
          onSelectBubble={(bubble) => {
            const identity = identities.get(bubble.id) ?? { runId: bubble.id };
            onHighlight(toggleHighlight(highlight, identity));
          }}
          height={320}
        />
      )}
      {gaps.length > 0 && (
        <ul className="flex flex-wrap gap-2">
          {gaps.map((gap) => (
            <li key={gap.id}>
              <button
                type="button"
                className="btn-ghost text-xs"
                aria-pressed={highlight?.runId === gap.id && highlight.policy == null && highlight.city == null}
                onClick={() => onHighlight(toggleHighlight(highlight, { runId: gap.id }))}
              >
                {gap.label} gap
              </button>
            </li>
          ))}
        </ul>
      )}
    </CorpusSceneFrame>
  );
}
