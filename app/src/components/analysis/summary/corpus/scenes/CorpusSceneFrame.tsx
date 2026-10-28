import { useRef, type ReactNode } from "react";
import { toPng } from "html-to-image";
import type { CorpusHighlight, CorpusMetric, CorpusView } from "../../../../../types/corpus";
import { corpusTableTsv } from "../../../../../utils/corpus/select";

export interface CorpusSceneFrameProps {
  title: string;
  view: CorpusView;
  metric: CorpusMetric;
  children: ReactNode;
}

/** Heading, copy-table, and PNG actions shared by the four 2D corpus scenes. */
export function CorpusSceneFrame({ title, view, metric, children }: CorpusSceneFrameProps) {
  const sceneRef = useRef<HTMLDivElement>(null);

  async function copyTable(): Promise<void> {
    const write = navigator.clipboard?.writeText;
    if (!write) return;
    try {
      await write.call(navigator.clipboard, corpusTableTsv(view, metric));
    } catch {
      return;
    }
  }

  async function savePng(): Promise<void> {
    const node = sceneRef.current;
    if (!node) return;
    try {
      const dataUrl = await toPng(node, {
        pixelRatio: 2,
        backgroundColor: "#1a1a2e",
      });
      const link = document.createElement("a");
      link.href = dataUrl;
      link.download = pngName(title);
      link.click();
    } catch {
      return;
    }
  }

  return (
    <section className="card space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h2 className="text-sm font-semibold">{title}</h2>
        <div className="flex gap-2">
          <button type="button" className="btn-ghost text-xs" onClick={() => void copyTable()}>
            Copy table
          </button>
          <button type="button" className="btn-ghost text-xs" onClick={() => void savePng()}>
            PNG
          </button>
        </div>
      </div>
      <div ref={sceneRef}>{children}</div>
    </section>
  );
}

function pngName(title: string): string {
  const slug = title
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "");
  return `${slug || "corpus-scene"}.png`;
}

export function presentMean(values: readonly (number | null | undefined)[]): number | null {
  let sum = 0;
  let count = 0;
  for (const value of values) {
    if (value == null || !Number.isFinite(value)) continue;
    sum += value;
    count += 1;
  }
  if (count === 0) return null;
  return sum / count;
}

/** Finite numbers keep their sign. A rounded non-zero value is never shown as 0. */
export function formatCorpusNumber(value: number): string {
  if (!Number.isFinite(value)) return "gap";
  if (Object.is(value, -0) || value === 0) return "0";
  const rounded = Math.round(value * 1000) / 1000;
  if (rounded === 0) return value.toExponential(2);
  return String(rounded);
}

export function runsInView(view: CorpusView): { id: string; label: string }[] {
  const seen = new Set<string>();
  const runs: { id: string; label: string }[] = [];
  for (const run of view.runs) {
    if (seen.has(run.id)) continue;
    seen.add(run.id);
    runs.push({ id: run.id, label: run.label });
  }
  for (const point of view.points) {
    if (seen.has(point.runId)) continue;
    seen.add(point.runId);
    runs.push({ id: point.runId, label: point.runLabel });
  }
  return runs;
}

export function highlightMatches(
  highlight: CorpusHighlight | null,
  row: { runId: string; policy: string | null; city: string | null }
): boolean {
  if (!highlight) return true;
  if (highlight.runId != null && highlight.runId !== row.runId) return false;
  if (highlight.policy != null && highlight.policy !== row.policy) return false;
  if (highlight.city != null && highlight.city !== row.city) return false;
  return true;
}

export function seriesDimmed(
  highlight: CorpusHighlight | null,
  rows: readonly { runId: string; policy: string | null; city: string | null }[]
): boolean {
  if (!highlight) return false;
  if (highlight.runId == null && highlight.policy == null && highlight.city == null) return false;
  return !rows.some((row) => highlightMatches(highlight, row));
}

export function sharedIdentity(
  rows: readonly { runId: string; policy: string | null; city: string | null }[],
  runId?: string
): CorpusHighlight {
  const identity: CorpusHighlight = {};
  if (runId != null) identity.runId = runId;
  else if (rows.length > 0 && rows.every((row) => row.runId === rows[0]?.runId)) {
    identity.runId = rows[0]?.runId;
  }
  const policy = commonText(rows.map((row) => row.policy));
  if (policy != null) identity.policy = policy;
  const city = commonText(rows.map((row) => row.city));
  if (city != null) identity.city = city;
  return identity;
}

function commonText(values: readonly (string | null)[]): string | null {
  if (values.length === 0) return null;
  const first = values[0];
  if (first == null || first === "") return null;
  for (const value of values) {
    if (value !== first) return null;
  }
  return first;
}

export function sameHighlight(current: CorpusHighlight | null, next: CorpusHighlight): boolean {
  if (!current) return false;
  return (
    (current.runId ?? undefined) === next.runId &&
    (current.policy ?? undefined) === next.policy &&
    (current.city ?? undefined) === next.city
  );
}

export function toggleHighlight(
  current: CorpusHighlight | null,
  next: CorpusHighlight
): CorpusHighlight | null {
  return sameHighlight(current, next) ? null : next;
}
