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
        className={`card bg-canvas-elevated/30 border border-canvas-border rounded-xl p-5 space-y-4 ${className}`}
      >
        <div className="flex items-center justify-between border-b border-canvas-border pb-3">
          <div className="flex items-center gap-2">
            <Activity size={16} className="text-canvas-muted" />
            <h4 className="text-sm font-semibold text-gray-200">
              Task Resource Telemetry
            </h4>
          </div>
          <span className="text-[11px] font-medium px-2 py-0.5 rounded-full bg-canvas-surface text-canvas-muted border border-canvas-border">
            No samples yet
          </span>
        </div>

        <div className="flex items-start gap-3 py-2">
          <div className="p-2 rounded-lg bg-canvas-surface border border-canvas-border text-canvas-muted">
            <Info size={18} />
          </div>
          <div className="space-y-1">
            <p className="text-xs text-gray-300 font-medium">
              No resource samples for this task yet.
            </p>
            <p className="text-xs text-canvas-muted leading-relaxed">
              The task runtime samples CPU load and memory RSS every few seconds
              while the simulator runs; samples appear here after the first flush
              to the control plane. Tasks run before this telemetry existed have none.
            </p>
          </div>
        </div>

        {/* Placeholder metric cards */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 pt-1">
          <div className="rounded-lg p-3 bg-canvas-surface/60 border border-canvas-border/60 flex flex-col gap-1">
            <div className="flex items-center gap-1.5 text-xs text-canvas-muted">
              <Cpu size={13} />
              <span>CPU Load</span>
            </div>
            <span className="text-base font-bold font-mono text-canvas-muted">
              — %
            </span>
          </div>

          <div className="rounded-lg p-3 bg-canvas-surface/60 border border-canvas-border/60 flex flex-col gap-1">
            <div className="flex items-center gap-1.5 text-xs text-canvas-muted">
              <HardDrive size={13} />
              <span>Memory RSS</span>
            </div>
            <span className="text-base font-bold font-mono text-canvas-muted">
              — MB
            </span>
          </div>

          <div className="rounded-lg p-3 bg-canvas-surface/60 border border-canvas-border/60 flex flex-col gap-1">
            <div className="flex items-center gap-1.5 text-xs text-canvas-muted">
              <Zap size={13} />
              <span>I/O Throughput</span>
            </div>
            <span className="text-base font-bold font-mono text-canvas-muted">
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
      className={`card bg-canvas-elevated/40 border border-canvas-border rounded-xl p-5 space-y-4 ${className}`}
    >
      <div className="flex items-center justify-between border-b border-canvas-border pb-3">
        <div className="flex items-center gap-2">
          <Activity size={16} className="text-accent-primary" />
          <h4 className="text-sm font-semibold text-gray-200">
            Task Resource Telemetry: {metrics.task_id}
          </h4>
        </div>
        <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-accent-primary/10 text-accent-primary border border-accent-primary/20">
          Live Heartbeats
        </span>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
        <div className="rounded-lg p-3 bg-canvas-surface border border-canvas-border flex flex-col gap-1">
          <div className="flex items-center gap-1.5 text-xs text-canvas-muted">
            <Cpu size={13} />
            <span>CPU Utilisation</span>
          </div>
          <span className="text-lg font-bold font-mono text-gray-100">
            {latestPoint.cpu_percent.toFixed(1)}%
          </span>
        </div>

        <div className="rounded-lg p-3 bg-canvas-surface border border-canvas-border flex flex-col gap-1">
          <div className="flex items-center gap-1.5 text-xs text-canvas-muted">
            <HardDrive size={13} />
            <span>Memory RSS</span>
          </div>
          <span className="text-lg font-bold font-mono text-gray-100">
            {latestPoint.memory_rss_mb.toFixed(0)} MB
          </span>
        </div>

        <div className="rounded-lg p-3 bg-canvas-surface border border-canvas-border flex flex-col gap-1">
          <div className="flex items-center gap-1.5 text-xs text-canvas-muted">
            <Zap size={13} />
            <span>I/O Throughput</span>
          </div>
          <span className="text-lg font-bold font-mono text-gray-100">
            {(latestPoint.disk_io_mbps ?? 0).toFixed(1)} MB/s
          </span>
        </div>
      </div>
    </div>
  );
}
