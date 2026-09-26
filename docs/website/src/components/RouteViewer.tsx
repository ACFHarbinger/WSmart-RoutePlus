import { Suspense, lazy, useState } from "react";
import type { Routes } from "../data/types";
import { useJson } from "../hooks/useJson";
import RouteMap2D from "./RouteMap2D";

const RouteMap3D = lazy(() => import("./RouteMap3D"));

function webglAvailable(): boolean {
  try {
    const canvas = document.createElement("canvas");
    return !!(canvas.getContext("webgl2") || canvas.getContext("webgl"));
  } catch {
    return false;
  }
}

/**
 * V3 — the multi-period route view. 2-D projection by default (no WebGL
 * required); the 3-D stacked view is lazy-loaded behind a button so three.js
 * never lands in the initial bundle.
 */
export default function RouteViewer() {
  const { data, status, error } = useJson<Routes>("/data/routes.json");
  const [scenarioIdx, setScenarioIdx] = useState(0);
  const [threeD, setThreeD] = useState(false);

  if (status === "loading") return <div className="viz-loading">Loading routes…</div>;
  if (status === "error" || !data)
    return <div className="viz-error">Could not load routes: {error ?? "unknown error"}</div>;

  const scenario = data.scenarios[scenarioIdx];

  return (
    <div className="viz-panel">
      <div className="viz-panel-head">
        <span>Multi-period routing</span>
        <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
          <select
            className="viz-select"
            value={scenarioIdx}
            onChange={(e) => setScenarioIdx(Number(e.target.value))}
            style={{ width: "auto", minWidth: 200 }}
            aria-label="Choose a network"
          >
            {data.scenarios.map((s, i) => (
              <option key={i} value={i}>
                {s.city} · N={s.N}
              </option>
            ))}
          </select>
          {webglAvailable() && (
            <button
              className="viz-preset"
              aria-pressed={threeD}
              onClick={() => setThreeD((v) => !v)}
            >
              {threeD ? "2-D view" : "Open 3-D view"}
            </button>
          )}
        </div>
      </div>

      <p className="viz-lede">
        {scenario.constructor} · {scenario.strategy} · {scenario.improver}, 30 days over{" "}
        {scenario.city} (N={scenario.N}). Scrub the horizon and watch which bins are
        revisited each day and which are left to fill for a week.
      </p>

      {threeD ? (
        <Suspense fallback={<div className="viz-loading">Loading 3-D view…</div>}>
          <RouteMap3D scenario={scenario} />
        </Suspense>
      ) : (
        <RouteMap2D scenario={scenario} />
      )}

      <p className="viz-note">
        Layout is a classical-MDS embedding of the repository's road-distance matrix —
        raw bin coordinates are not checked in, so pairwise road distances are preserved
        up to a 2-D projection. This is a layout, not a map.
      </p>
    </div>
  );
}
