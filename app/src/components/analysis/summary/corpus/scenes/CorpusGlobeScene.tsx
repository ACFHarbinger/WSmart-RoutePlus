import { useEffect, useMemo, useState } from "react";
import DeckGL from "@deck.gl/react";
import { ArcLayer, ScatterplotLayer } from "@deck.gl/layers";
import MapGL from "react-map-gl/maplibre";
import "maplibre-gl/dist/maplibre-gl.css";
import {
  CORPUS_METRIC_LABEL,
  type CorpusHighlight,
  type CorpusSeriesPoint,
  type CorpusView,
} from "../../../../../types/corpus";
import type { CorpusSceneProps } from "../sceneProps";

const MAP_STYLE = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json";

const PALETTE: readonly [number, number, number][] = [
  [56, 189, 248],
  [251, 191, 36],
  [52, 211, 153],
  [244, 114, 182],
  [167, 139, 250],
  [251, 146, 60],
];

/** Marker size for a missing metric. Not the radius a zero value would get. */
const GAP_RADIUS = 5;
const MIN_RADIUS = 7;
const MAX_RADIUS = 18;

interface LonLat {
  lon: number;
  lat: number;
}

interface GlobeArc {
  id: string;
  source: [number, number];
  target: [number, number];
  color: [number, number, number, number];
}

interface GlobeDot {
  key: string;
  position: [number, number];
  runId: string;
  policy: string | null;
  city: string | null;
  gap: boolean;
  radius: number;
  color: [number, number, number, number];
}

interface MapCamera {
  longitude: number;
  latitude: number;
  zoom: number;
  pitch: number;
  bearing: number;
}

function finite(value: number | null | undefined): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function onGlobe(point: CorpusSeriesPoint): boolean {
  return point.hasCoordinates === true && finite(point.lat) && finite(point.lon);
}

function meanLonLat(points: readonly LonLat[]): LonLat | null {
  if (points.length === 0) return null;
  const lon = points.reduce((sum, point) => sum + point.lon, 0) / points.length;
  const lat = points.reduce((sum, point) => sum + point.lat, 0) / points.length;
  return { lon, lat };
}

function colorFor(key: string): [number, number, number] {
  let hash = 0;
  for (let index = 0; index < key.length; index += 1) {
    hash = (hash * 31 + key.charCodeAt(index)) >>> 0;
  }
  return PALETTE[hash % PALETTE.length];
}

function pointLit(point: CorpusSeriesPoint, highlight: CorpusHighlight | null): boolean {
  if (!highlight) return true;
  if (highlight.runId && highlight.runId !== point.runId) return false;
  if (highlight.policy && highlight.policy !== point.policy) return false;
  if (highlight.city && highlight.city !== point.city) return false;
  return true;
}

function samePlace(a: LonLat, b: LonLat): boolean {
  return Math.abs(a.lon - b.lon) < 1e-6 && Math.abs(a.lat - b.lat) < 1e-6;
}

function baselineCentroid(view: CorpusView, geo: readonly CorpusSeriesPoint[]): LonLat | null {
  const owned = geo.filter((point) => point.runLabel === view.baselineLabel);
  return meanLonLat(owned.map((point) => ({ lon: point.lon as number, lat: point.lat as number })));
}

function policyCityArcs(geo: readonly CorpusSeriesPoint[]): GlobeArc[] {
  const byPolicy = new Map<string, Map<string, CorpusSeriesPoint[]>>();
  for (const point of geo) {
    if (!point.policy || !point.city) continue;
    const cities = byPolicy.get(point.policy) ?? new Map<string, CorpusSeriesPoint[]>();
    const bucket = cities.get(point.city) ?? [];
    bucket.push(point);
    cities.set(point.city, bucket);
    byPolicy.set(point.policy, cities);
  }

  const arcs: GlobeArc[] = [];
  for (const [policy, cities] of byPolicy) {
    const anchors = [...cities.entries()]
      .flatMap(([city, points]) => {
        const mean = meanLonLat(
          points.map((point) => ({ lon: point.lon as number, lat: point.lat as number }))
        );
        return mean ? [{ city, ...mean }] : [];
      })
      .sort((a, b) => a.city.localeCompare(b.city));
    const [red, green, blue] = colorFor(policy);
    for (let left = 0; left < anchors.length; left += 1) {
      for (let right = left + 1; right < anchors.length; right += 1) {
        const source = anchors[left];
        const target = anchors[right];
        if (samePlace(source, target)) continue;
        arcs.push({
          id: `${policy}:${source.city}:${target.city}`,
          source: [source.lon, source.lat],
          target: [target.lon, target.lat],
          color: [red, green, blue, 180],
        });
      }
    }
  }
  return arcs;
}

function arcsToBaseline(
  view: CorpusView,
  geo: readonly CorpusSeriesPoint[],
  anchor: LonLat
): GlobeArc[] {
  const byRun = new Map<string, CorpusSeriesPoint[]>();
  for (const point of geo) {
    if (point.runLabel === view.baselineLabel) continue;
    const bucket = byRun.get(point.runId) ?? [];
    bucket.push(point);
    byRun.set(point.runId, bucket);
  }

  const arcs: GlobeArc[] = [];
  for (const [runId, points] of byRun) {
    const mean = meanLonLat(
      points.map((point) => ({ lon: point.lon as number, lat: point.lat as number }))
    );
    if (!mean || samePlace(mean, anchor)) continue;
    const [red, green, blue] = colorFor(points[0]?.policy ?? runId);
    arcs.push({
      id: `baseline:${runId}`,
      source: [mean.lon, mean.lat],
      target: [anchor.lon, anchor.lat],
      color: [red, green, blue, 180],
    });
  }
  return arcs;
}

/** Policy arcs, or arcs into the pinned baseline when that run is on the globe. */
function buildArcs(view: CorpusView, geo: readonly CorpusSeriesPoint[]): GlobeArc[] {
  const anchor = baselineCentroid(view, geo);
  if (anchor) return arcsToBaseline(view, geo, anchor);
  return policyCityArcs(geo);
}

function radiusFor(value: number, min: number, max: number): number {
  if (max === min) return (MIN_RADIUS + MAX_RADIUS) / 2;
  const t = (value - min) / (max - min);
  return MIN_RADIUS + t * (MAX_RADIUS - MIN_RADIUS);
}

function buildDots(geo: readonly CorpusSeriesPoint[], highlight: CorpusHighlight | null): GlobeDot[] {
  const present = geo.map((point) => point.value).filter(finite);
  const min = present.length > 0 ? Math.min(...present) : 0;
  const max = present.length > 0 ? Math.max(...present) : 0;
  return geo.map((point, index) => {
    const gap = !finite(point.value);
    const lit = pointLit(point, highlight);
    const [red, green, blue] = gap ? [148, 163, 184] : colorFor(point.policy ?? point.runId);
    return {
      key: `${point.runId}:${point.day ?? "day"}:${index}`,
      position: [point.lon as number, point.lat as number],
      runId: point.runId,
      policy: point.policy,
      city: point.city,
      gap,
      radius: gap ? GAP_RADIUS : radiusFor(point.value as number, min, max),
      color: [red, green, blue, lit ? 220 : 48],
    };
  });
}

function cameraFor(geo: readonly CorpusSeriesPoint[]): MapCamera {
  if (geo.length === 0) {
    return { longitude: 10, latitude: 20, zoom: 1.5, pitch: 0, bearing: 0 };
  }
  const lats = geo.map((point) => point.lat as number);
  const lons = geo.map((point) => point.lon as number);
  const latSpan = Math.max(...lats) - Math.min(...lats);
  const lonSpan = Math.max(...lons) - Math.min(...lons);
  const span = Math.max(latSpan, lonSpan);
  const zoom = span < 0.5 ? 8 : span < 2 ? 6 : span < 8 ? 4.5 : span < 20 ? 3.2 : 1.8;
  return {
    longitude: (Math.min(...lons) + Math.max(...lons)) / 2,
    latitude: (Math.min(...lats) + Math.max(...lats)) / 2,
    zoom,
    pitch: 0,
    bearing: 0,
  };
}

function listedLit(
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

function missingRuns(points: readonly CorpusSeriesPoint[]): [string, string][] {
  const missing = new Map<string, string>();
  for (const point of points) {
    if (point.hasCoordinates === false) missing.set(point.runId, point.runLabel);
  }
  return [...missing.entries()].sort((a, b) => a[1].localeCompare(b[1]));
}

function pickHighlight(dot: GlobeDot): CorpusHighlight {
  return {
    runId: dot.runId,
    ...(dot.policy ? { policy: dot.policy } : {}),
    ...(dot.city ? { city: dot.city } : {}),
  };
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

function sameHighlight(highlight: CorpusHighlight | null, next: CorpusHighlight): boolean {
  if (!highlight) return false;
  return (
    highlight.runId === next.runId &&
    (highlight.policy ?? undefined) === next.policy &&
    (highlight.city ?? undefined) === next.city
  );
}

export function CorpusGlobeScene({ view, metric, highlight, onHighlight }: CorpusSceneProps) {
  const reducedMotion = useReducedMotion();
  const geo = useMemo(() => view.points.filter(onGlobe), [view.points]);
  const arcs = useMemo(() => (geo.length === 0 ? [] : buildArcs(view, geo)), [geo, view]);
  const dots = useMemo(
    () => (geo.length === 0 ? [] : buildDots(geo, highlight)),
    [geo, highlight]
  );
  const camera = useMemo(() => cameraFor(geo), [geo]);
  const missing = useMemo(() => missingRuns(view.points), [view.points]);
  const layers = useMemo(() => {
    if (geo.length === 0) return [];
    return [
      new ArcLayer<GlobeArc>({
        id: "corpus-globe-arcs",
        data: arcs,
        greatCircle: true,
        getSourcePosition: (arc) => arc.source,
        getTargetPosition: (arc) => arc.target,
        getSourceColor: (arc) => arc.color,
        getTargetColor: (arc) => arc.color,
        getWidth: 2,
        widthMinPixels: 2,
        pickable: false,
      }),
      new ScatterplotLayer<GlobeDot>({
        id: "corpus-globe-points",
        data: dots,
        getPosition: (dot) => dot.position,
        getFillColor: (dot) => (dot.gap ? [15, 23, 42, dot.color[3]] : dot.color),
        getLineColor: (dot) => dot.color,
        stroked: true,
        filled: true,
        lineWidthMinPixels: 1,
        radiusUnits: "pixels",
        getRadius: (dot) => dot.radius,
        pickable: true,
        autoHighlight: false,
        onClick: (info) => {
          const dot = info.object as GlobeDot | undefined;
          if (!dot) return;
          const next = pickHighlight(dot);
          onHighlight(sameHighlight(highlight, next) ? null : next);
        },
      }),
    ];
  }, [arcs, dots, geo.length, highlight, onHighlight]);

  const arcCaption = baselineCentroid(view, geo)
    ? "Arcs run from each coordinated run to the pinned baseline."
    : "Arcs join cities that share a policy.";

  return (
    <section
      aria-label="City globe"
      data-motion={reducedMotion ? "reduced" : "static"}
      className="space-y-2 rounded-lg border border-canvas-border p-3"
    >
      <header>
        <h3 className="text-sm font-semibold text-white">City globe</h3>
        <p className="text-xs text-canvas-muted">{CORPUS_METRIC_LABEL[metric]}</p>
      </header>
      {geo.length === 0 ? (
        <p className="text-xs text-canvas-muted">No coordinates in this corpus.</p>
      ) : (
        <>
          <p className="text-[11px] text-canvas-muted">{arcCaption}</p>
          <div className="relative h-[320px] overflow-hidden rounded-lg border border-canvas-border">
            <DeckGL
              key={`${camera.longitude.toFixed(4)}:${camera.latitude.toFixed(4)}:${geo.length}`}
              initialViewState={camera}
              controller={{ inertia: false }}
              layers={layers}
            >
              <MapGL mapStyle={MAP_STYLE} />
            </DeckGL>
          </div>
        </>
      )}
      {geo.length > 0 && missing.length > 0 && (
        <ul className="space-y-1 text-xs text-canvas-muted">
          {missing.map(([runId, label]) => {
            const dimmed = !listedLit(runId, view.points, highlight);
            return (
              <li key={runId}>
                <button
                  type="button"
                  className={dimmed ? "opacity-40" : undefined}
                  data-dimmed={dimmed ? "true" : "false"}
                  onClick={() =>
                    onHighlight(
                      highlight?.runId === runId && !highlight.policy && !highlight.city
                        ? null
                        : { runId }
                    )
                  }
                >
                  Not on the globe: {label}
                </button>
              </li>
            );
          })}
        </ul>
      )}
    </section>
  );
}
