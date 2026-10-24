import { Activity, Cpu, HardDrive, Info, Zap } from "lucide-react";

export interface TaskMetricPoint {
  timestamp_ms: number;
  cpu_percent: number;
  memory_rss_mb: number;
  disk_io_mbps?: number;
}

export interface TaskMetricsData {
  task_id: string;
  points: TaskMetricPoint[];
  peak_cpu?: number;
  peak_memory_mb?: number;
}

export interface TaskMetricsPanelProps {
  /**
   * Task metrics data.
   * If null, undefined, or empty points, renders the empty state
   * (no resource samples received yet for this task).
   */
  metrics?: TaskMetricsData | null;
  className?: string;
}

export function TaskMetricsPanel({
  metrics,
  className = "",
}: TaskMetricsPanelProps) {
  const hasData = metrics != null && metrics.points && metrics.points.length > 0;

  if (!hasData) {
    return (
      <div
        data-testid="task-metrics-empty"
        className={`bg-noc-panel border border-noc-border rounded-xl p-5 space-y-4 ${className}`}
      >
        <div className="flex items-center justify-between border-b border-noc-border pb-3">
          <div className="flex items-center gap-2">
            <Activity size={16} className="text-noc-textMuted" />
            <h4 className="text-sm font-semibold text-noc-textPrimary">
              Task Resource Telemetry
            </h4>
          </div>
          <span className="text-[11px] font-medium px-2 py-0.5 rounded-full bg-noc-deep text-noc-textMuted border border-noc-border">
            No samples yet
          </span>
        </div>

        <div className="flex items-start gap-3 py-2">
          <div className="p-2 rounded-lg bg-noc-deep border border-noc-border text-noc-textMuted">
            <Info size={18} />
          </div>
          <div className="space-y-1">
            <p className="text-xs text-noc-textPrimary font-medium">
              No resource samples for this task yet.
            </p>
            <p className="text-xs text-noc-textMuted leading-relaxed">
              The task runtime samples CPU load and memory RSS every few seconds
              while the simulator runs; samples appear here after the first flush
              to the control plane. Tasks run before this telemetry existed have none.
            </p>
          </div>
        </div>

        {/* Placeholder metric cards */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 pt-1">
          <div className="rounded-lg p-3 bg-noc-deep border border-noc-border flex flex-col gap-1">
            <div className="flex items-center gap-1.5 text-xs text-noc-textMuted">
              <Cpu size={13} />
              <span>CPU Load</span>
            </div>
            <span className="text-base font-bold font-mono text-noc-textMuted">
              — %
            </span>
          </div>

          <div className="rounded-lg p-3 bg-noc-deep border border-noc-border flex flex-col gap-1">
            <div className="flex items-center gap-1.5 text-xs text-noc-textMuted">
              <HardDrive size={13} />
              <span>Memory RSS</span>
            </div>
            <span className="text-base font-bold font-mono text-noc-textMuted">
              — MB
            </span>
          </div>

          <div className="rounded-lg p-3 bg-noc-deep border border-noc-border flex flex-col gap-1">
            <div className="flex items-center gap-1.5 text-xs text-noc-textMuted">
              <Zap size={13} />
              <span>I/O Throughput</span>
            </div>
            <span className="text-base font-bold font-mono text-noc-textMuted">
              — MB/s
            </span>
          </div>
        </div>
      </div>
    );
  }

  // Active metrics rendering for CR-12
  const latestPoint = metrics.points[metrics.points.length - 1];

  return (
    <div
      data-testid="task-metrics-active"
      className={`bg-noc-panel border border-noc-border rounded-xl p-5 space-y-4 ${className}`}
    >
      <div className="flex items-center justify-between border-b border-noc-border pb-3">
        <div className="flex items-center gap-2">
          <Activity size={16} className="text-noc-cyan" />
          <h4 className="text-sm font-semibold text-noc-textPrimary">
            Task Resource Telemetry: {metrics.task_id}
          </h4>
        </div>
        <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-noc-cyan/15 text-noc-cyan border border-noc-cyan/30">
          Live Heartbeats
        </span>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
        <div className="rounded-lg p-3 bg-noc-deep border border-noc-border flex flex-col gap-1">
          <div className="flex items-center gap-1.5 text-xs text-noc-textMuted">
            <Cpu size={13} />
            <span>CPU Utilisation</span>
          </div>
          <span className="text-lg font-bold font-mono text-noc-textPrimary tabular-nums">
            {latestPoint.cpu_percent.toFixed(1)}%
          </span>
        </div>

        <div className="rounded-lg p-3 bg-noc-deep border border-noc-border flex flex-col gap-1">
          <div className="flex items-center gap-1.5 text-xs text-noc-textMuted">
            <HardDrive size={13} />
            <span>Memory RSS</span>
          </div>
          <span className="text-lg font-bold font-mono text-noc-textPrimary tabular-nums">
            {latestPoint.memory_rss_mb.toFixed(0)} MB
          </span>
        </div>

        <div className="rounded-lg p-3 bg-noc-deep border border-noc-border flex flex-col gap-1">
          <div className="flex items-center gap-1.5 text-xs text-noc-textMuted">
            <Zap size={13} />
            <span>I/O Throughput</span>
          </div>
          <span className="text-lg font-bold font-mono text-noc-textPrimary tabular-nums">
            {(latestPoint.disk_io_mbps ?? 0).toFixed(1)} MB/s
          </span>
        </div>
      </div>
    </div>
  );
}
