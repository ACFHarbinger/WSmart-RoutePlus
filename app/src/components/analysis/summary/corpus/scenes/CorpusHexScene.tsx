import { useEffect, useMemo, useState } from "react";
import {
  CORPUS_METRIC_LABEL,
  type CorpusHighlight,
  type CorpusSeriesPoint,
} from "../../../../../types/corpus";
import { generatedLayout } from "../../../../../utils/corpus/select";
import type { CorpusSceneProps } from "../sceneProps";

/** Outline used when a run has no present metric. Independent of the value scale. */
const GAP_OUTLINE = 16;
const MIN_RADIUS = 12;
const MAX_RADIUS = 28;

interface PlacedNode {
  id: string;
  label: string;
  cx: number;
  cy: number;
  mean: number | null;
  radius: number | null;
  lit: boolean;
}

function useReducedMotion(): boolean {
  const [reduced, setReduced] = useState(false);
  useEffect(() => {
    if (typeof window.matchMedia !== "function") return;
    const query = window.matchMedia("(prefers-reduced-motion: reduce)");
    const apply = () => setReduced(query.matches);
    apply();
    query.addEventListener("change", apply);
    return () => query.removeEventListener("change", apply);
  }, []);
  return reduced;
}

function finite(value: number | null | undefined): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function formatValue(value: number): string {
  const rounded = Math.round(value * 100) / 100;
  return String(rounded);
}

function meanMetric(points: readonly CorpusSeriesPoint[], runId: string): number | null {
  let sum = 0;
  let count = 0;
  for (const point of points) {
    if (point.runId !== runId || !finite(point.value)) continue;
    sum += point.value;
    count += 1;
  }
  if (count === 0) return null;
  return sum / count;
}

function runLit(
  runId: string,
  points: readonly CorpusSeriesPoint[],
  highlight: CorpusHighlight | null
): boolean {
  if (!highlight) return true;
  if (highlight.runId && highlight.runId !== runId) return false;
  if (!highlight.policy && !highlight.city) return true;
  return points.some((point) => {
    if (point.runId !== runId) return false;
    if (highlight.policy && point.policy !== highlight.policy) return false;
    if (highlight.city && point.city !== highlight.city) return false;
    return true;
  });
}

function radiusFor(value: number, min: number, max: number): number {
  if (max === min) return (MIN_RADIUS + MAX_RADIUS) / 2;
  const t = (value - min) / (max - min);
  return MIN_RADIUS + t * (MAX_RADIUS - MIN_RADIUS);
}

function hexPoints(cx: number, cy: number, radius: number): string {
  const points: string[] = [];
  for (let vertex = 0; vertex < 6; vertex += 1) {
    const angle = Math.PI / 6 + (Math.PI / 3) * vertex;
    const px = cx + radius * Math.cos(angle);
    const py = cy + radius * Math.sin(angle);
    points.push(`${px.toFixed(1)},${py.toFixed(1)}`);
  }
  return points.join(" ");
}

function layoutIds(view: CorpusSceneProps["view"]): { id: string; label: string }[] {
  if (view.runs.length > 0) {
    return view.runs.map((run) => ({ id: run.id, label: run.label }));
  }
  const seen = new Map<string, string>();
  for (const point of view.points) {
    if (!seen.has(point.runId)) seen.set(point.runId, point.runLabel);
  }
  return [...seen.entries()].map(([id, label]) => ({ id, label }));
}

function placeNodes(
  view: CorpusSceneProps["view"],
  highlight: CorpusHighlight | null
): PlacedNode[] {
  const nodes = generatedLayout(layoutIds(view));
  if (nodes.length === 0) return [];
  const means = nodes.map((node) => meanMetric(view.points, node.id));
  const present = means.filter(finite);
  const min = present.length > 0 ? Math.min(...present) : 0;
  const max = present.length > 0 ? Math.max(...present) : 0;
  const minX = Math.min(...nodes.map((node) => node.x));
  const minY = Math.min(...nodes.map((node) => node.y));
  const stepX = 78;
  const stepY = 84;
  const pad = 48;
  return nodes.map((node, index) => {
    const mean = means[index];
    return {
      id: node.id,
      label: node.label,
      cx: (node.x - minX) * stepX + pad,
      cy: (node.y - minY) * stepY + pad,
      mean,
      radius: finite(mean) ? radiusFor(mean, min, max) : null,
      lit: runLit(node.id, view.points, highlight),
    };
  });
}

export function CorpusHexScene({ view, metric, highlight, onHighlight }: CorpusSceneProps) {
  const reducedMotion = useReducedMotion();
  const nodes = useMemo(() => placeNodes(view, highlight), [highlight, view]);
  const width =
    nodes.length === 0 ? 240 : Math.max(...nodes.map((node) => node.cx)) + 72;
  const height =
    nodes.length === 0 ? 120 : Math.max(...nodes.map((node) => node.cy)) + 64;

  return (
    <section
      aria-label="Hex map"
      data-motion={reducedMotion ? "reduced" : "static"}
      className="space-y-2 rounded-lg border border-canvas-border p-3"
    >
      <header>
        <h3 className="text-sm font-semibold text-white">Hex map</h3>
        <p className="text-xs text-canvas-muted">
          {CORPUS_METRIC_LABEL[metric]}. <span>Generated layout</span>. Positions are a plane, not
          longitude or latitude, even when a run also has real coordinates.
        </p>
      </header>
      <svg
        viewBox={`0 0 ${width} ${height}`}
        className="w-full select-none"
        role="img"
        aria-label="Generated layout"
      >
        {nodes.map((node) => {
          const gap = node.radius == null;
          const drawRadius = node.radius ?? GAP_OUTLINE;
          const valueText = gap ? "gap" : formatValue(node.mean as number);
          return (
            <g
              key={node.id}
              role="button"
              tabIndex={0}
              data-dimmed={node.lit ? "false" : "true"}
              data-value={valueText}
              opacity={node.lit ? 1 : 0.28}
              aria-label={`${node.label} ${valueText}`}
              className="cursor-pointer outline-none"
              onClick={() =>
                onHighlight(
                  highlight?.runId === node.id && !highlight.policy && !highlight.city
                    ? null
                    : { runId: node.id }
                )
              }
              onKeyDown={(event) => {
                if (event.key !== "Enter" && event.key !== " ") return;
                event.preventDefault();
                onHighlight(
                  highlight?.runId === node.id && !highlight.policy && !highlight.city
                    ? null
                    : { runId: node.id }
                );
              }}
            >
              <polygon
                points={hexPoints(node.cx, node.cy, drawRadius)}
                fill={gap ? "none" : "#38bdf8"}
                stroke={gap ? "#94a3b8" : "#e2e8f0"}
                strokeWidth={gap ? 1.5 : 1}
                strokeDasharray={gap ? "3 2" : undefined}
              />
              <text
                x={node.cx}
                y={node.cy + 4}
                textAnchor="middle"
                fill="#f8fafc"
                fontSize={11}
              >
                {valueText}
              </text>
              <text
                x={node.cx}
                y={node.cy + drawRadius + 14}
                textAnchor="middle"
                fill="#cbd5e1"
                fontSize={10}
              >
                {node.label}
              </text>
            </g>
          );
        })}
      </svg>
    </section>
  );
}
