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

const HALF = 72;

interface BarMark {
  id: string;
  label: string;
  delta: number;
  text: string;
  identity: CorpusHighlight;
  dim: boolean;
}

export function DivergingBarsScene({ view, metric, highlight, onHighlight }: CorpusSceneProps) {
  const hasBaseline = view.baseline != null;
  const gaps: { id: string; label: string }[] = [];
  const bars: BarMark[] = [];

  for (const run of runsInView(view)) {
    const rows = view.points.filter((point) => point.runId === run.id);
    const mean = presentMean(rows.map((point) => point.value));
    if (mean == null) {
      gaps.push(run);
      continue;
    }
    const delta = hasBaseline ? mean - (view.baseline as number) : mean;
    bars.push({
      id: run.id,
      label: run.label,
      delta,
      text: formatCorpusNumber(delta),
      identity: sharedIdentity(rows, run.id),
      dim: seriesDimmed(highlight, rows),
    });
  }

  const domain = bars.reduce((max, bar) => Math.max(max, Math.abs(bar.delta)), 0);

  return (
    <CorpusSceneFrame title="Against baseline" view={view} metric={metric}>
      <p className="text-xs text-canvas-muted">
        {hasBaseline ? `Baseline ${view.baselineLabel} (${formatCorpusNumber(view.baseline as number)})` : "No baseline"}
      </p>
      {bars.length > 0 && (
        <div className="relative" style={{ height: HALF * 2 + 36 }}>
          <div className="absolute left-0 right-0 border-t border-slate-500" style={{ top: HALF }} />
          <div className="flex h-full items-stretch gap-3 px-2" style={{ height: HALF * 2 }}>
            {bars.map((bar) => {
              const px = domain === 0 ? 0 : (Math.abs(bar.delta) / domain) * HALF;
              const above = bar.delta >= 0;
              return (
                <button
                  key={bar.id}
                  type="button"
                  className="relative h-full flex-1"
                  style={{ opacity: bar.dim ? 0.35 : 1 }}
                  aria-label={hasBaseline ? `${bar.label} delta ${bar.text}` : `${bar.label} ${bar.text}`}
                  aria-pressed={highlight?.runId === bar.id}
                  onClick={() => onHighlight(toggleHighlight(highlight, bar.identity))}
                >
                  <span
                    className="absolute left-1 right-1 rounded-sm"
                    style={{
                      height: px,
                      background: above ? "#38bdf8" : "#f59e0b",
                      bottom: above ? "50%" : undefined,
                      top: above ? undefined : "50%",
                    }}
                  />
                  <span
                    className="absolute inset-x-0 text-center text-xs text-canvas-muted"
                    style={{ top: above ? Math.max(0, HALF - px - 16) : HALF + px + 2 }}
                  >
                    {bar.text}
                  </span>
                  <span className="absolute inset-x-0 bottom-0 truncate text-center text-[10px] text-canvas-muted">
                    {bar.label}
                  </span>
                </button>
              );
            })}
          </div>
          <span className="absolute left-0 text-[10px] text-canvas-muted" style={{ top: HALF + 2 }}>
            0
          </span>
        </div>
      )}
      {gaps.length > 0 && (
        <p className="text-xs text-canvas-muted">{gaps.map((gap) => `${gap.label} gap`).join("; ")}</p>
      )}
    </CorpusSceneFrame>
  );
}
