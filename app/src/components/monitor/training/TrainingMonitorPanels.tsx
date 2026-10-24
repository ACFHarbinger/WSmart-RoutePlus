import { useEffect, useRef, useState } from "react";
import ReactECharts from "echarts-for-react";
import type EChartsReact from "echarts-for-react";
import { invoke } from "@tauri-apps/api/core";
import { ChevronDown, ChevronRight } from "lucide-react";
import { OpenPathToolbar } from "../../common/OpenPathToolbar";
import { GradNormSparkline, LrSparkline } from "../live/TrainingMetricSparklines";
import { ChartExportButtons } from "../../common/ChartExportButtons";
import {
  filterCheckpointEntries,
  parentRunBrushLabelFromCheckpointPath,
} from "../../../utils/training/checkpoints";
import type {
  DirEntry,
  TrainingRun,
  TrainingMetricsRow,
} from "../../../types";

export const RUN_COLORS = [
  "#6366f1", // indigo
  "#34d399", // emerald
  "#f87171", // red
  "#fbbf24", // amber
  "#a78bfa", // violet
  "#fb923c", // orange
  "#38bdf8", // sky
  "#f472b6", // pink
];

function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

export function MultiRunChart({
  runsMetrics,
  logScale,
}: {
  runsMetrics: { name: string; metrics: TrainingMetricsRow[] }[];
  logScale: boolean;
}) {
  const chartRef = useRef<EChartsReact | null>(null);
  if (runsMetrics.length === 0) return null;

  const series: object[] = [];
  const legendData: string[] = [];

  runsMetrics.forEach(({ name, metrics }, i) => {
    const color = RUN_COLORS[i % RUN_COLORS.length];
    const epochs = metrics.map((r) => r.epoch ?? r.step ?? 0);
    const scaleY = (y: number | null | undefined) =>
      y == null ? null : logScale ? Math.max(y, 1e-8) : y;
    const trainLoss = metrics.map((r, j) => [epochs[j], scaleY(r.train_loss)]);
    const valLoss = metrics.map((r, j) => [epochs[j], scaleY(r.val_loss)]);
    const reward = metrics.map((r, j) => [epochs[j], r.reward ?? null]);

    const hasReward = metrics.some((r) => r.reward != null);
    const shortName = name.length > 20 ? `…${name.slice(-18)}` : name;

    legendData.push(
      `${shortName} loss`,
      ...(hasReward ? [`${shortName} reward`] : [])
    );

    series.push({
      name: `${shortName} loss`,
      type: "line",
      data: trainLoss,
      smooth: true,
      lineStyle: { color, width: 2 },
      itemStyle: { color },
      symbol: "none",
      encode: { x: 0, y: 1 },
    });

    if (metrics.some((r) => r.val_loss != null)) {
      series.push({
        name: `${shortName} val`,
        type: "line",
        data: valLoss,
        smooth: true,
        lineStyle: { color, width: 1.5, type: "dashed" },
        itemStyle: { color },
        symbol: "none",
        encode: { x: 0, y: 1 },
      });
    }

    if (hasReward) {
      series.push({
        name: `${shortName} reward`,
        type: "line",
        yAxisIndex: 1,
        data: reward,
        smooth: true,
        lineStyle: { color, width: 2, type: "dotted" },
        itemStyle: { color },
        symbol: "none",
        encode: { x: 0, y: 1 },
      });
    }
  });

  return (
    <div className="card space-y-2">
      <div className="flex items-center justify-between">
        <p className="text-xs text-canvas-muted">
          {logScale ? "Multi-run overlay (log-scale loss)" : "Multi-run overlay"}
        </p>
        <ChartExportButtons
          chartRef={{ current: chartRef.current }}
          filenameStem="training-overlay"
        />
      </div>
      <ReactECharts
        ref={chartRef}
        option={{
          backgroundColor: "transparent",
          legend: {
            data: legendData,
            textStyle: { color: "#9090b0", fontSize: 10 },
            type: "scroll",
          },
          grid: { left: 50, right: 60, top: 45, bottom: 30 },
          xAxis: {
            type: "value",
            name: "Epoch",
            nameTextStyle: { color: "#9090b0" },
            axisLabel: { color: "#9090b0", fontSize: 10 },
          },
          yAxis: [
            {
              type: (logScale ? "log" : "value") as "log" | "value",
              logBase: 10,
              name: logScale ? "Loss (log)" : "Loss",
              nameTextStyle: { color: "#9090b0" },
              axisLabel: { color: "#9090b0", fontSize: 10 },
              minorSplitLine: { show: false },
            },
            {
              type: "value",
              name: "Reward",
              nameTextStyle: { color: "#9090b0" },
              axisLabel: { color: "#9090b0", fontSize: 10 },
              splitLine: { show: false },
            },
          ],
          series,
          tooltip: { trigger: "axis" },
        }}
        style={{ height: 260 }}
      />
    </div>
  );
}

export function HparamsPanel({ runPath }: { runPath: string }) {
  const [pairs, setPairs] = useState<{ key: string; value: string }[] | null>(
    null
  );
  const [expanded, setExpanded] = useState(false);

  useEffect(() => {
    const hparamsPath = `${runPath}/hparams.yaml`;
    invoke<string>("read_text_file", { path: hparamsPath })
      .then((text) => {
        const parsed = text
          .split("\n")
          .map((line) => line.match(/^(\S[^:]*?):\s*(.*)$/))
          .filter(Boolean)
          .map((m) => ({ key: m![1].trim(), value: m![2].trim() }))
          .filter(({ key }) => !key.startsWith("#"));
        setPairs(parsed);
      })
      .catch(() => setPairs(null));
  }, [runPath]);

  if (!pairs?.length) return null;

  const visible = expanded ? pairs : pairs.slice(0, 8);

  return (
    <div className="rounded-xl border border-canvas-border overflow-hidden">
      <button
        className="w-full flex items-center gap-2 px-4 py-2.5 bg-canvas-surface hover:bg-canvas-hover text-left"
        onClick={() => setExpanded((v) => !v)}
      >
        {expanded ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
        <span className="text-xs font-semibold text-canvas-muted uppercase tracking-wider">
          Hyperparameters
        </span>
        <span className="ml-auto text-xs text-canvas-muted">
          {pairs.length} params
        </span>
      </button>
      {expanded && (
        <div className="px-4 py-2 bg-canvas-bg border-t border-canvas-border">
          <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-0.5">
            {visible.map(({ key, value }) => (
              <>
                <dt
                  key={`k-${key}`}
                  className="text-xs text-canvas-muted font-mono py-0.5"
                >
                  {key}
                </dt>
                <dd
                  key={`v-${key}`}
                  className="text-xs text-gray-300 font-mono py-0.5 truncate"
                >
                  {value || "—"}
                </dd>
              </>
            ))}
          </dl>
          {pairs.length > 8 && (
            <button
              className="text-xs text-accent-primary mt-2"
              onClick={() => setExpanded((v) => !v)}
            >
              {expanded && visible.length < pairs.length
                ? `Show all ${pairs.length} params`
                : "Collapse"}
            </button>
          )}
        </div>
      )}
    </div>
  );
}

export function CheckpointBrowser({
  runPath,
  projectRoot,
}: {
  runPath: string;
  projectRoot: string | null;
}) {
  const [checkpoints, setCheckpoints] = useState<DirEntry[] | null>(null);

  useEffect(() => {
    invoke<DirEntry[]>("list_dir", { path: `${runPath}/checkpoints` })
      .then((entries) => setCheckpoints(filterCheckpointEntries(entries)))
      .catch(() => setCheckpoints([]));
  }, [runPath]);

  if (!checkpoints?.length) return null;

  return (
    <div className="rounded-xl border border-canvas-border overflow-hidden">
      <div className="px-4 py-2.5 bg-canvas-surface">
        <p className="text-xs font-semibold text-canvas-muted uppercase tracking-wider">
          Checkpoints ({checkpoints.length})
        </p>
      </div>
      <div className="divide-y divide-canvas-border/40 bg-canvas-bg">
        {checkpoints.map((ckpt) => (
          <div key={ckpt.path} className="flex items-center gap-3 px-4 py-2">
            <OpenPathToolbar
              path={ckpt.path}
              projectRoot={projectRoot}
              kind="checkpoint"
              label={ckpt.name}
              brushLabel={parentRunBrushLabelFromCheckpointPath(
                ckpt.path,
                projectRoot
              )}
              chipClassName="flex-1 min-w-0 max-w-none"
              labeled
              labeledIconSize={12}
              className="flex-1 min-w-0"
            >
              <span className="text-xs text-canvas-muted shrink-0">
                {formatBytes(ckpt.size_bytes)}
              </span>
            </OpenPathToolbar>
          </div>
        ))}
      </div>
    </div>
  );
}

export function RunPanel({
  run,
  metrics,
  color,
  projectRoot,
  logScale = false,
}: {
  run: TrainingRun;
  metrics: TrainingMetricsRow[];
  color: string;
  projectRoot: string | null;
  logScale?: boolean;
}) {
  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        <span
          className="w-2.5 h-2.5 rounded-full shrink-0"
          style={{ backgroundColor: color }}
        />
        <OpenPathToolbar
          path={run.path}
          projectRoot={projectRoot}
          kind="training"
          label={run.name}
          brushLabel={run.name}
          storedLabel={run.name}
          className="flex-1 min-w-0"
        >
          <span className="text-xs text-canvas-muted shrink-0">
            {metrics.length} epochs
          </span>
        </OpenPathToolbar>
      </div>
      <GradNormSparkline
        metrics={metrics}
        logScale={logScale}
        exportName="training-monitor-grad-norm"
      />
      <LrSparkline
        metrics={metrics}
        logScale={logScale}
        exportName="training-monitor-lr"
      />
      {run.has_hparams && <HparamsPanel runPath={run.path} />}
      <CheckpointBrowser runPath={run.path} projectRoot={projectRoot} />
    </div>
  );
}
