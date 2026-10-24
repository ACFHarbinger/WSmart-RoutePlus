import { useMemo, useRef } from "react";
import ReactECharts from "echarts-for-react";
import type EChartsReact from "echarts-for-react";
import { BarChart2, FolderOpen } from "lucide-react";
import { ChartExportButtons } from "../common/ChartExportButtons";
import { OpenPathToolbar } from "../common/OpenPathToolbar";
import {
  chartMetricYAxisType,
  displayBarValue,
} from "../../utils/charts/chartLogScale";
import type { DatasetPreviewStats } from "../../types";

export function formatBytes(b: number) {
  if (b < 1024) return `${b} B`;
  if (b < 1024 ** 2) return `${(b / 1024).toFixed(1)} KB`;
  return `${(b / 1024 ** 2).toFixed(1)} MB`;
}

export function DemandHistogram({
  option,
  logScale = false,
}: {
  option: object;
  logScale?: boolean;
}) {
  const chartRef = useRef<EChartsReact | null>(null);
  return (
    <div>
      <div className="flex items-center justify-between mb-1">
        <p className="text-xs text-canvas-muted">
          Demand distribution{logScale ? " · log-scale counts" : ""}
        </p>
        <ChartExportButtons
          chartRef={{ current: chartRef.current }}
          filenameStem="dataset-demand-hist"
        />
      </div>
      <ReactECharts ref={chartRef} option={option} style={{ height: 140 }} />
    </div>
  );
}

export function DataGenSourceAndPreview({
  dataSource,
  setDataSource,
  tsplibPath,
  setTsplibPath,
  pickTsplibFile,
  sensorCsvPath,
  setSensorCsvPath,
  pickSensorFile,
  projectRoot,
  previewLoading,
  previewDataset,
  previewDatasetPath,
  previewStats,
  logScale,
}: {
  dataSource: "synthetic" | "tsplib" | "sensor";
  setDataSource: (v: "synthetic" | "tsplib" | "sensor") => void;
  tsplibPath: string;
  setTsplibPath: (v: string) => void;
  pickTsplibFile: () => void;
  sensorCsvPath: string;
  setSensorCsvPath: (v: string) => void;
  pickSensorFile: () => void;
  projectRoot: string | null;
  previewLoading: boolean;
  previewDataset: () => void;
  previewDatasetPath: string;
  previewStats: DatasetPreviewStats | null;
  logScale: boolean;
}) {
  const demandHistOption = useMemo(() => {
    if (!previewStats?.demand_histogram.length) return null;
    const yKey = "count";
    return {
      backgroundColor: "transparent",
      grid: { left: 40, right: 10, top: 10, bottom: 30 },
      xAxis: {
        type: "category",
        data: ["0–0.25", "0.25–0.5", "0.5–0.75", "0.75–1.0", "1.0+"],
        axisLabel: { color: "#9090b0", fontSize: 9 },
      },
      yAxis: {
        type: chartMetricYAxisType(yKey, logScale),
        logBase: 10,
        axisLabel: { color: "#9090b0", fontSize: 9 },
        minorSplitLine: { show: false },
      },
      series: [
        {
          type: "bar",
          data: previewStats.demand_histogram.map((v) =>
            displayBarValue(v, yKey, logScale)
          ),
          itemStyle: { color: "#6366f1" },
        },
      ],
      tooltip: { trigger: "axis" },
    };
  }, [previewStats, logScale]);

  return (
    <>
      <div className="card space-y-3">
        <h2 className="text-sm font-semibold text-gray-200">Data Source</h2>
        <div className="flex flex-wrap gap-4">
          {(["synthetic", "tsplib", "sensor"] as const).map((src) => (
            <label
              key={src}
              className="flex items-center gap-2 cursor-pointer text-sm text-gray-300"
            >
              <input
                type="radio"
                name="dataSource"
                className="accent-accent-primary"
                checked={dataSource === src}
                onChange={() => setDataSource(src)}
              />
              {src === "synthetic"
                ? "Synthetic"
                : src === "tsplib"
                  ? "TSPLIB"
                  : "Sensor CSV"}
            </label>
          ))}
        </div>
        {dataSource === "tsplib" && (
          <div className="space-y-1">
            <div className="flex gap-2">
              <input
                type="text"
                className="input-base font-mono text-xs flex-1"
                value={tsplibPath}
                onChange={(e) => setTsplibPath(e.target.value)}
                placeholder="path/to/instance.vrp"
              />
              <button
                onClick={pickTsplibFile}
                className="btn-ghost p-1.5 text-canvas-muted hover:text-gray-200"
              >
                <FolderOpen size={13} />
              </button>
            </div>
            {tsplibPath.trim() ? (
              <OpenPathToolbar
                path={tsplibPath.trim()}
                projectRoot={projectRoot}
                chipClassName="max-w-full"
                handoff={false}
              />
            ) : null}
          </div>
        )}
        {dataSource === "sensor" && (
          <div className="space-y-1">
            <div className="flex gap-2">
              <input
                type="text"
                className="input-base font-mono text-xs flex-1"
                value={sensorCsvPath}
                onChange={(e) => setSensorCsvPath(e.target.value)}
                placeholder="path/to/bins.csv (timestamp,bin_id,fill_level,waste_type)"
              />
              <button
                onClick={pickSensorFile}
                className="btn-ghost p-1.5 text-canvas-muted hover:text-gray-200"
              >
                <FolderOpen size={13} />
              </button>
            </div>
            {sensorCsvPath.trim() ? (
              <OpenPathToolbar
                path={sensorCsvPath.trim()}
                projectRoot={projectRoot}
                kind="csv"
                labeled
                labeledIconSize={12}
                chipClassName="max-w-full"
              />
            ) : null}
            <p className="text-[10px] text-canvas-muted">
              CSV format: timestamp, bin_id, fill_level, waste_type (§12.3 sensor
              schema)
            </p>
          </div>
        )}
      </div>

      <div className="card space-y-3">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold text-gray-200">
            Instance Preview
          </h2>
          <button
            onClick={previewDataset}
            disabled={previewLoading || !projectRoot}
            className="btn-ghost text-xs flex items-center gap-1"
          >
            <BarChart2 size={12} />
            {previewLoading ? "Loading…" : "Preview .pkl/.pt"}
          </button>
        </div>
        {previewDatasetPath.trim() ? (
          <OpenPathToolbar
            path={previewDatasetPath.trim()}
            projectRoot={projectRoot}
            chipClassName="max-w-full"
            handoff={false}
          />
        ) : null}
        {previewStats ? (
          <div className="space-y-3">
            <div className="grid grid-cols-4 gap-2 text-xs">
              <div className="kpi-card">
                <p className="text-canvas-muted">Instances</p>
                <p className="font-mono text-gray-100">
                  {previewStats.num_instances}
                </p>
              </div>
              <div className="kpi-card">
                <p className="text-canvas-muted">Nodes</p>
                <p className="font-mono text-gray-100">
                  {previewStats.num_nodes ?? "—"}
                </p>
              </div>
              <div className="kpi-card">
                <p className="text-canvas-muted">Demand μ±σ</p>
                <p className="font-mono text-gray-100">
                  {previewStats.demand_mean != null
                    ? `${previewStats.demand_mean.toFixed(3)} ± ${(previewStats.demand_std ?? 0).toFixed(3)}`
                    : "—"}
                </p>
              </div>
              <div className="kpi-card">
                <p className="text-canvas-muted">File size</p>
                <p className="font-mono text-gray-100">
                  {formatBytes(previewStats.file_size_bytes)}
                </p>
              </div>
            </div>
            {demandHistOption && (
              <DemandHistogram option={demandHistOption} logScale={logScale} />
            )}
          </div>
        ) : (
          <p className="text-xs text-canvas-muted">
            Pick a generated dataset to inspect node count, demand statistics,
            and histogram.
          </p>
        )}
      </div>
    </>
  );
}
