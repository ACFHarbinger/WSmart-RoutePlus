import { lazy, Suspense, useRef } from "react";
import ReactECharts from "echarts-for-react";
import { CanvasExportButton } from "../../common/CanvasExportButton";
import { ChartExportButtons } from "../../common/ChartExportButtons";
import { GRAPH_PRESETS } from "../../../utils/graph/graphCoords";
import {
  buildMatrixHeatmapOption,
  type AttentionRole,
} from "../../../utils/charts/tensorHeatmap";
import { transformMatrixLogScale } from "../../../utils/charts/chartLogScale";
import type { LandscapeMarker } from "../../../utils/graph/lossLandscape";
import type { TensorSlicePreview } from "../../../types";
import type { LossLandscapeView } from "./LossLandscape3D";
import type { AttentionView, CompareMode } from "./MLAttentionControls";

const LossLandscape3D = lazy(() =>
  import("./LossLandscape3D").then((m) => ({ default: m.LossLandscape3D }))
);
const AttentionSigmaView = lazy(() =>
  import("../topology/AttentionSigmaView").then((m) => ({
    default: m.AttentionSigmaView,
  }))
);

export function buildLogAwareMatrixHeatmap(
  rawValues: number[][],
  opts: Parameters<typeof buildMatrixHeatmapOption>[1] & { title?: string },
  useLogScale: boolean,
  metricKey = "attention"
): Record<string, unknown> {
  const displayValues = useLogScale
    ? transformMatrixLogScale(rawValues, metricKey, true)
    : rawValues;
  const title = opts.title
    ? `${opts.title}${useLogScale ? " · log colour" : ""}`
    : opts.title;
  const base = buildMatrixHeatmapOption(displayValues, { ...opts, title });
  if (!useLogScale) return base;
  const xLabel = opts.xLabel ?? "Col";
  const yLabel = opts.yLabel ?? "Row";
  return {
    ...base,
    tooltip: {
      position: "top",
      formatter: (p: { value?: [number, number, number] }) => {
        const v = p.value;
        if (!v) return "";
        const raw = rawValues[v[1]]?.[v[0]];
        const rawText = Number.isFinite(raw) ? Number(raw).toFixed(4) : "—";
        return `${yLabel} ${v[1]} · ${xLabel} ${v[0]}<br/>${rawText}`;
      },
    },
  };
}

export function MLAttentionVisualSection({
  attentionView,
  attentionGraphOption,
  selectedKey,
  decodeStep,
  graphPreset,
  logScale,
  processedValues,
  graphCoords,
  sparseK,
  theme,
  heatmapOption,
  compareMode,
  compareHeatmapOption,
  primaryHeatmapOption,
  distCompareLabel,
  compareStep,
  activeAttentionRole,
}: {
  attentionView: AttentionView;
  attentionGraphOption: Record<string, unknown> | null;
  selectedKey: string;
  decodeStep: number;
  graphPreset: string;
  logScale: boolean;
  processedValues: number[][] | null;
  graphCoords: Array<{ lat: number; lng: number }> | null;
  sparseK: number;
  theme: "dark" | "light";
  heatmapOption: Record<string, unknown> | null;
  compareMode: CompareMode;
  compareHeatmapOption: Record<string, unknown> | null;
  primaryHeatmapOption: Record<string, unknown> | null;
  distCompareLabel: string;
  compareStep: number;
  activeAttentionRole: AttentionRole;
}) {
  const chartRef = useRef<ReactECharts>(null);
  const compareChartRef = useRef<ReactECharts>(null);
  const graphChartRef = useRef<ReactECharts>(null);
  const attentionSigmaRef = useRef<HTMLDivElement>(null);

  return (
    <>
      {attentionView === "graph" && attentionGraphOption && (
        <div className="space-y-2">
          <div className="flex justify-end gap-2">
            <ChartExportButtons
              chartRef={graphChartRef}
              filenameStem={`attention-graph-${selectedKey}`}
            />
          </div>
          <ReactECharts
            ref={graphChartRef}
            option={attentionGraphOption}
            style={{ height: 400 }}
            notMerge
          />
          <p className="text-[10px] text-canvas-muted">
            Bipartite graph overlay: query node (amber) at decode step{" "}
            {decodeStep}; edge opacity ∝ attention weight to key nodes on{" "}
            {GRAPH_PRESETS.find((p) => p.id === graphPreset)?.label}.
            {logScale
              ? " Log-scale edge mapping — tooltips show raw weights."
              : ""}
          </p>
        </div>
      )}

      {attentionView === "sigma" && processedValues && graphCoords?.length && (
        <div className="space-y-2">
          <div className="flex justify-end gap-2">
            <CanvasExportButton
              container={() => attentionSigmaRef.current}
              filename={`attention-sigma-${selectedKey}.png`}
            />
          </div>
          <div ref={attentionSigmaRef}>
            <Suspense
              fallback={
                <div className="h-[400px] flex items-center justify-center text-xs text-canvas-muted">
                  Loading Sigma.js canvas…
                </div>
              }
            >
              <AttentionSigmaView
                coords={graphCoords}
                values={processedValues}
                queryRow={decodeStep}
                topK={sparseK > 0 ? sparseK : 24}
                sparseValues={sparseK > 0 ? processedValues : undefined}
                theme={theme}
                logScale={logScale}
              />
            </Suspense>
          </div>
          <p className="text-[10px] text-canvas-muted">
            Sigma.js WebGL bipartite graph: ForceAtlas2 layout on{" "}
            {GRAPH_PRESETS.find((p) => p.id === graphPreset)?.label}; query node
            (amber) at decode step {decodeStep}; edge opacity ∝ attention
            weight.
            {logScale
              ? " Log-scale edge mapping — edge weight attribute stores raw values."
              : ""}
          </p>
        </div>
      )}

      {attentionView === "heatmap" && heatmapOption && (
        <div className="space-y-2">
          <div className="flex justify-end gap-2 flex-wrap">
            <ChartExportButtons
              chartRef={chartRef}
              filenameStem={`attention-${selectedKey}`}
            />
            {(compareMode === "side-by-side" ||
              compareMode === "distribution") &&
              compareHeatmapOption && (
                <ChartExportButtons
                  chartRef={compareChartRef}
                  filenameStem={
                    compareMode === "distribution"
                      ? `attention-compare-${distCompareLabel}`
                      : `attention-compare-step-${compareStep}`
                  }
                />
              )}
          </div>
          <div
            className={
              (compareMode === "side-by-side" ||
                compareMode === "distribution") &&
              compareHeatmapOption
                ? "grid grid-cols-1 lg:grid-cols-2 gap-3"
                : ""
            }
          >
            <ReactECharts
              ref={chartRef}
              option={
                compareMode === "distribution"
                  ? (primaryHeatmapOption ?? heatmapOption)
                  : heatmapOption
              }
              style={{ height: 360 }}
              notMerge
            />
            {(compareMode === "side-by-side" ||
              compareMode === "distribution") &&
              compareHeatmapOption && (
                <ReactECharts
                  ref={compareChartRef}
                  option={compareHeatmapOption}
                  style={{ height: 360 }}
                  notMerge
                />
              )}
          </div>
          <p className="text-[10px] text-canvas-muted">
            Q/K/V colour palettes ({activeAttentionRole}) · spherical k-means
            row bands · head selector · sparse top-k · decode-step compare ·
            Empirical vs Gamma-3 (side-by-side / overlay Δ).
            {logScale
              ? " Log-scale colour map on raw weights (overlay Δ stays linear)."
              : " Linear colour map."}{" "}
            Graph-on-coords via ECharts or Sigma.js WebGL (View toggle).
          </p>
        </div>
      )}
    </>
  );
}

export function MLLossLandscapeSection({
  lossPreview,
  lossView,
  setLossView,
  lossMarkers,
  lossOption,
  lossMinima,
  logScale,
}: {
  lossPreview: TensorSlicePreview | null;
  lossView: LossLandscapeView;
  setLossView: (v: LossLandscapeView) => void;
  lossMarkers: LandscapeMarker[];
  lossOption: Record<string, unknown> | null;
  lossMinima: {
    label: string;
    sharpness: number;
    generalizationNote: string;
  } | null;
  logScale: boolean;
}) {
  const lossChartRef = useRef<ReactECharts>(null);
  const loss3dRef = useRef<HTMLDivElement>(null);

  if (!lossPreview) {
    return (
      <p className="text-xs text-canvas-muted">
        Open a loss landscape `.npz` (e.g. from{" "}
        <code className="font-mono">logic/gen/export_loss_landscape.py</code>)
        containing a <code className="font-mono">loss_grid</code> 2D array.
      </p>
    );
  }

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2 text-xs">
        <span className="text-canvas-muted">3D view</span>
        <select
          className="select-base text-xs py-0.5"
          value={lossView}
          onChange={(e) => setLossView(e.target.value as LossLandscapeView)}
        >
          <option value="surface">Surface mesh</option>
          <option value="voxels">InstancedMesh voxels</option>
        </select>
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
        <div className="space-y-2">
          <div className="flex justify-end gap-2">
            <CanvasExportButton
              container={() => loss3dRef.current}
              filename={`loss-terrain-3d-${lossView}.png`}
              label="3D PNG"
            />
          </div>
          <Suspense
            fallback={
              <div className="h-[280px] flex items-center justify-center text-xs text-canvas-muted">
                Loading 3D canvas…
              </div>
            }
          >
            <LossLandscape3D
              ref={loss3dRef}
              values={lossPreview.values}
              markers={lossMarkers}
              view={lossView}
              logScale={logScale}
            />
          </Suspense>
        </div>
        <div className="space-y-2">
          <div className="flex justify-end gap-2">
            <ChartExportButtons
              chartRef={lossChartRef}
              filenameStem="loss-landscape"
            />
          </div>
          {lossOption && (
            <ReactECharts
              ref={lossChartRef}
              option={lossOption}
              style={{ height: 248 }}
              notMerge
            />
          )}
          <p className="text-[10px] text-canvas-muted">
            {logScale
              ? "Log-scale 3D terrain + contour colour map — tooltips show raw loss values"
              : "Linear 3D terrain + contour colour map — loss grid from NPZ export"}
          </p>
        </div>
      </div>
      {lossMinima && (
        <p className="text-[10px] text-canvas-muted">
          Minima annotation: {lossMinima.label} basin (sharpness{" "}
          {lossMinima.sharpness.toFixed(3)}). {lossMinima.generalizationNote}
          {lossMarkers.length > 0
            ? " Amber diamond = BPC exact-solver projection on the landscape."
            : " Bundle bpc_theta1/bpc_theta2 in the NPZ (see export_loss_landscape.py) for the BPC marker."}
        </p>
      )}
    </div>
  );
}
