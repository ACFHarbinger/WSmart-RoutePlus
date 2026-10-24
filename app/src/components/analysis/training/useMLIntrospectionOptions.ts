import { useMemo } from "react";
import { buildAttentionGraphOption } from "../../../utils/graph/attentionGraph";
import { chartMetricDisplay } from "../../../utils/charts/chartLogScale";
import {
  analyzeLossMinima,
  type LandscapeMarker,
} from "../../../utils/graph/lossLandscape";
import {
  CLUSTER_PALETTE,
  reorderRowsByClusters,
  sphericalKMeans,
} from "../../../utils/graph/sphericalKMeans";
import {
  applySparseTopK,
  buildMatrixHeatmapOption,
  diffMatrices,
  type AttentionRole,
} from "../../../utils/charts/tensorHeatmap";
import type { TensorSlicePreview } from "../../../types";
import type { AttentionView, CompareMode, IntrospectionTab } from "./MLAttentionControls";
import { buildLogAwareMatrixHeatmap } from "./MLAttentionAndLossViews";

export function useMLIntrospectionOptions({
  preview,
  comparePreview,
  distPreview,
  lossPreview,
  lossMarkers,
  tab,
  clusterK,
  sparseK,
  compareMode,
  effectiveIndices,
  theme,
  primaryDistLabel,
  distCompareLabel,
  selectedKey,
  compareStep,
  activeAttentionRole,
  logScale,
  graphCoords,
  attentionView,
  decodeStep,
}: {
  preview: TensorSlicePreview | null;
  comparePreview: TensorSlicePreview | null;
  distPreview: TensorSlicePreview | null;
  lossPreview: TensorSlicePreview | null;
  lossMarkers: LandscapeMarker[];
  tab: IntrospectionTab;
  clusterK: number;
  sparseK: number;
  compareMode: CompareMode;
  effectiveIndices: number[];
  theme: "dark" | "light";
  primaryDistLabel: string;
  distCompareLabel: string;
  selectedKey: string;
  compareStep: number;
  activeAttentionRole: AttentionRole;
  logScale: boolean;
  graphCoords: Array<{ lat: number; lng: number }> | null;
  attentionView: AttentionView;
  decodeStep: number;
}) {
  const clusteredAttention = useMemo(() => {
    if (!preview || tab !== "attention" || clusterK <= 1) {
      return { values: preview?.values ?? null, bandSplits: [] as number[] };
    }
    let grid = preview.values;
    if (sparseK > 0) grid = applySparseTopK(grid, sparseK);
    const labels = sphericalKMeans(grid, clusterK);
    const reordered = reorderRowsByClusters(grid, labels);
    return { values: reordered.values, bandSplits: reordered.bandSplits };
  }, [preview, tab, clusterK, sparseK]);

  const processedValues = useMemo(() => {
    if (tab === "attention" && clusterK > 1) return clusteredAttention.values;
    if (!preview) return null;
    let grid = preview.values;
    if (sparseK > 0) grid = applySparseTopK(grid, sparseK);
    return grid;
  }, [preview, sparseK, tab, clusterK, clusteredAttention.values]);

  const compareValues = useMemo(() => {
    if (!comparePreview) return null;
    let grid = comparePreview.values;
    if (sparseK > 0) grid = applySparseTopK(grid, sparseK);
    return grid;
  }, [comparePreview, sparseK]);

  const distValues = useMemo(() => {
    if (!distPreview) return null;
    let grid = distPreview.values;
    if (sparseK > 0) grid = applySparseTopK(grid, sparseK);
    return grid;
  }, [distPreview, sparseK]);

  const heatmapOption = useMemo(() => {
    if (!processedValues) return null;
    const isDiffMatrix =
      (compareMode === "overlay" && !!compareValues) ||
      (compareMode === "distribution" && !!distValues);
    const values =
      compareMode === "overlay" && compareValues
        ? diffMatrices(processedValues, compareValues)
        : compareMode === "distribution" && distValues
        ? diffMatrices(processedValues, distValues)
        : processedValues;
    const label =
      tab === "attention"
        ? compareMode === "distribution"
          ? `${primaryDistLabel} · ${preview?.key} [${effectiveIndices.join(",")}]${
              distValues ? " Δ" : ""
            }`
          : `Attention · ${preview?.key} [${effectiveIndices.join(",")}]${
              compareMode === "overlay" && compareValues ? " Δ" : ""
            }`
        : `${preview?.key} [${preview?.full_shape.join("×")}]`;
    const heatmapOpts = {
      title: label,
      min:
        compareMode === "overlay" || compareMode === "distribution"
          ? undefined
          : preview?.min,
      max:
        compareMode === "overlay" || compareMode === "distribution"
          ? undefined
          : preview?.max,
      theme,
      xLabel: tab === "attention" ? "Key" : "X",
      yLabel: tab === "attention" ? "Query" : "Y",
      attentionRole: tab === "attention" ? activeAttentionRole : undefined,
      clusterBandSplits: tab === "attention" ? clusteredAttention.bandSplits : undefined,
      clusterPalette: CLUSTER_PALETTE,
    };
    const useLogScale = logScale && tab === "attention" && !isDiffMatrix;
    return buildLogAwareMatrixHeatmap(values, heatmapOpts, useLogScale);
  }, [
    processedValues,
    compareValues,
    distValues,
    compareMode,
    preview,
    tab,
    effectiveIndices,
    theme,
    primaryDistLabel,
    activeAttentionRole,
    clusteredAttention.bandSplits,
    logScale,
  ]);

  const compareHeatmapOption = useMemo(() => {
    if (compareMode === "side-by-side" && compareValues) {
      return buildLogAwareMatrixHeatmap(
        compareValues,
        {
          title: `Compare step ${compareStep}`,
          min: comparePreview?.min,
          max: comparePreview?.max,
          theme,
          xLabel: "Key",
          yLabel: "Query",
          attentionRole: activeAttentionRole,
        },
        logScale
      );
    }
    if (compareMode === "distribution" && distValues) {
      return buildLogAwareMatrixHeatmap(
        distValues,
        {
          title: `${distCompareLabel} · ${distPreview?.key ?? selectedKey}`,
          min: distPreview?.min,
          max: distPreview?.max,
          theme,
          xLabel: "Key",
          yLabel: "Query",
          attentionRole: activeAttentionRole,
        },
        logScale
      );
    }
    return null;
  }, [
    compareValues,
    distValues,
    compareMode,
    compareStep,
    comparePreview,
    distPreview,
    distCompareLabel,
    selectedKey,
    theme,
    activeAttentionRole,
    logScale,
  ]);

  const primaryHeatmapOption = useMemo(() => {
    if (!processedValues || compareMode !== "distribution") return heatmapOption;
    return buildLogAwareMatrixHeatmap(
      processedValues,
      {
        title: `${primaryDistLabel} · ${preview?.key ?? selectedKey} [${effectiveIndices.join(",")}]`,
        min: preview?.min,
        max: preview?.max,
        theme,
        xLabel: "Key",
        yLabel: "Query",
        attentionRole: activeAttentionRole,
      },
      logScale
    );
  }, [
    processedValues,
    compareMode,
    heatmapOption,
    primaryDistLabel,
    preview,
    selectedKey,
    effectiveIndices,
    theme,
    activeAttentionRole,
    logScale,
  ]);

  const attentionGraphOption = useMemo(() => {
    if (!processedValues || !graphCoords?.length || attentionView !== "graph") return null;
    return buildAttentionGraphOption(graphCoords, processedValues, {
      title: `Attention graph · query row ${decodeStep} · ${selectedKey}`,
      theme,
      queryRow: decodeStep,
      topK: sparseK > 0 ? sparseK : 24,
      sparseValues: sparseK > 0 ? processedValues : undefined,
      logScale,
    });
  }, [processedValues, graphCoords, attentionView, decodeStep, selectedKey, theme, sparseK, logScale]);

  const lossOption = useMemo(() => {
    if (!lossPreview) return null;
    const rawValues = lossPreview.values;
    const displayValues = logScale
      ? rawValues.map((row) =>
          row.map((v) =>
            Number.isFinite(v) ? (chartMetricDisplay(v, "loss", true) ?? v) : v
          )
        )
      : rawValues;
    const base = buildMatrixHeatmapOption(displayValues, {
      title: `Loss landscape · ${lossPreview.key}${logScale ? " · log colour" : ""}`,
      theme,
      xLabel: "θ₁",
      yLabel: "θ₂",
    });
    const withTooltip: Record<string, unknown> = {
      ...base,
      tooltip: {
        position: "top",
        formatter: (p: { value?: [number, number, number] }) => {
          const v = p.value;
          if (!v) return "";
          const raw = rawValues[v[1]]?.[v[0]];
          const rawText = Number.isFinite(raw) ? Number(raw).toFixed(4) : "—";
          return `θ₂ ${v[1]} · θ₁ ${v[0]}<br/>loss ${rawText}`;
        },
      },
    };
    if (!lossMarkers.length) return withTooltip;
    const markPoint = {
      data: lossMarkers.map((m) => ({
        name: m.label,
        coord: [m.col, m.row],
        value: m.loss != null ? m.loss.toFixed(4) : "",
        itemStyle: { color: m.color ?? "#f59e0b" },
      })),
      symbol: "diamond",
      symbolSize: 14,
    };
    const series = (withTooltip.series as Record<string, unknown>[])?.[0];
    if (!series) return withTooltip;
    return { ...withTooltip, series: [{ ...series, markPoint }] };
  }, [lossPreview, lossMarkers, theme, logScale]);

  const lossMinima = useMemo(
    () => (lossPreview ? analyzeLossMinima(lossPreview.values) : null),
    [lossPreview]
  );

  return {
    processedValues,
    heatmapOption,
    compareHeatmapOption,
    primaryHeatmapOption,
    attentionGraphOption,
    lossOption,
    lossMinima,
  };
}
