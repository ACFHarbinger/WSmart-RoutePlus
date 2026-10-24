import {
  Copy,
  Download,
  RefreshCw,
  RotateCcw,
  Square,
  Terminal,
} from "lucide-react";
import type {
  CloudJobStatus,
  CloudTaskMetrics,
  CloudTaskSummary,
} from "../../types/cloud";
import { CloudStatusBadge } from "./CloudStatusBadge";
import { TaskMetricsPanel } from "./TaskMetricsPanel";

export function CloudJobDetailHeader({
  activeJob,
  loadingDetail,
  handleCopyJobId,
  handleRetryJob,
  retrying,
  taskCounts,
  setDownloadJobId,
  isTerminal,
  handleCancelJob,
  cancelling,
}: {
  activeJob: CloudJobStatus;
  loadingDetail: boolean;
  handleCopyJobId: () => void;
  handleRetryJob: () => void;
  retrying: boolean;
  taskCounts: {
    total: number;
    succeeded: number;
    running: number;
    failed: number;
    queued: number;
  };
  setDownloadJobId: (id: string) => void;
  isTerminal: boolean;
  handleCancelJob: () => void;
  cancelling: boolean;
}) {
  return (
    <div className="card bg-canvas-surface border border-canvas-border rounded-xl p-5 space-y-4">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-canvas-border pb-4">
        <div>
          <div className="flex items-center gap-2.5">
            <h2 className="text-base font-bold text-gray-100">
              {activeJob.name || activeJob.job_id}
            </h2>
            <CloudStatusBadge state={activeJob.state} />
          </div>
          <div className="flex items-center gap-2 text-xs text-canvas-muted mt-1 font-mono">
            <span>{activeJob.job_id}</span>
            {loadingDetail && (
              <RefreshCw size={11} className="animate-spin text-accent-primary" />
            )}
            <button
              onClick={handleCopyJobId}
              className="hover:text-gray-200 transition-colors"
              title="Copy Job ID"
            >
              <Copy size={12} />
            </button>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <button
            onClick={handleRetryJob}
            disabled={
              retrying ||
              (taskCounts.failed === 0 && activeJob.state !== "cancelled")
            }
            className="btn-ghost text-xs flex items-center gap-1.5 px-3 py-1.5 border border-canvas-border disabled:opacity-40"
            title="Requeue failed and cancelled tasks"
          >
            <RotateCcw size={13} className={retrying ? "animate-spin" : ""} />
            <span>Retry Failed</span>
          </button>

          <button
            onClick={() => setDownloadJobId(activeJob.job_id)}
            className="btn-ghost text-xs flex items-center gap-1.5 px-3 py-1.5 border border-canvas-border"
            title="Download results index and daily checkpoints"
          >
            <Download size={13} />
            <span>Download</span>
          </button>

          {!isTerminal && (
            <button
              onClick={handleCancelJob}
              disabled={cancelling}
              className="btn-ghost text-xs text-accent-danger hover:bg-accent-danger/10 flex items-center gap-1.5 px-3 py-1.5 border border-accent-danger/30"
              title="Stop all active and queued tasks"
            >
              <Square size={13} className={cancelling ? "animate-spin" : ""} />
              <span>Cancel Job</span>
            </button>
          )}
        </div>
      </div>

      <div className="space-y-1.5">
        <div className="flex items-center justify-between text-xs text-canvas-muted">
          <span>
            Task Progress: {taskCounts.succeeded} of {taskCounts.total} completed
          </span>
          <span className="font-mono">
            {taskCounts.total > 0
              ? `${Math.round((taskCounts.succeeded / taskCounts.total) * 100)}%`
              : "0%"}
          </span>
        </div>

        <div className="w-full bg-canvas-elevated rounded-full h-2.5 overflow-hidden flex">
          {taskCounts.total > 0 && (
            <>
              <div
                style={{
                  width: `${(taskCounts.succeeded / taskCounts.total) * 100}%`,
                }}
                className="bg-emerald-500 transition-all duration-300"
                title={`Succeeded: ${taskCounts.succeeded}`}
              />
              <div
                style={{
                  width: `${(taskCounts.running / taskCounts.total) * 100}%`,
                }}
                className="bg-accent-success transition-all duration-300"
                title={`Running: ${taskCounts.running}`}
              />
              <div
                style={{
                  width: `${(taskCounts.failed / taskCounts.total) * 100}%`,
                }}
                className="bg-accent-danger transition-all duration-300"
                title={`Failed: ${taskCounts.failed}`}
              />
            </>
          )}
        </div>

        <div className="flex items-center gap-4 text-[11px] text-canvas-muted pt-1">
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-emerald-500" />
            {taskCounts.succeeded} Succeeded
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-accent-success" />
            {taskCounts.running} Running
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-accent-danger" />
            {taskCounts.failed} Failed / Preempted
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-canvas-muted" />
            {taskCounts.queued} Queued
          </span>
        </div>
      </div>
    </div>
  );
}

export function CloudJobDetailTabs({
  activeJob,
  activeTab,
  setActiveTab,
  taskFilter,
  setTaskFilter,
  filteredTasks,
  setLogTaskId,
  metricsTask,
  setMetricsTaskId,
  taskMetrics,
}: {
  activeJob: CloudJobStatus;
  activeTab: "tasks" | "metrics" | "json";
  setActiveTab: (t: "tasks" | "metrics" | "json") => void;
  taskFilter: string;
  setTaskFilter: (f: string) => void;
  filteredTasks: CloudTaskSummary[];
  setLogTaskId: (id: string) => void;
  metricsTask: CloudTaskSummary | null;
  setMetricsTaskId: (id: string) => void;
  taskMetrics: CloudTaskMetrics | null;
}) {
  return (
    <div className="card bg-canvas-surface border border-canvas-border rounded-xl p-4 space-y-4">
      <div className="flex items-center justify-between border-b border-canvas-border pb-3">
        <div className="flex items-center gap-2">
          <button
            onClick={() => setActiveTab("tasks")}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-colors ${
              activeTab === "tasks"
                ? "bg-accent-primary text-white"
                : "text-canvas-muted hover:text-gray-200"
            }`}
          >
            Tasks ({activeJob.tasks.length})
          </button>
          <button
            onClick={() => setActiveTab("metrics")}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-colors ${
              activeTab === "metrics"
                ? "bg-accent-primary text-white"
                : "text-canvas-muted hover:text-gray-200"
            }`}
          >
            Resource Telemetry
          </button>
          <button
            onClick={() => setActiveTab("json")}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-colors ${
              activeTab === "json"
                ? "bg-accent-primary text-white"
                : "text-canvas-muted hover:text-gray-200"
            }`}
          >
            Raw Status JSON
          </button>
        </div>

        {activeTab === "tasks" && (
          <div className="flex items-center gap-1 text-[11px]">
            {["all", "running", "failed", "succeeded"].map((tState) => (
              <button
                key={tState}
                onClick={() => setTaskFilter(tState)}
                className={`px-2 py-0.5 rounded capitalize ${
                  taskFilter === tState
                    ? "bg-canvas-elevated text-gray-100 font-medium border border-canvas-border"
                    : "text-canvas-muted hover:text-gray-200"
                }`}
              >
                {tState}
              </button>
            ))}
          </div>
        )}
      </div>

      {activeTab === "tasks" && (
        <div className="space-y-2">
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs border-collapse">
              <thead>
                <tr className="border-b border-canvas-border text-canvas-muted font-medium">
                  <th className="py-2 px-3">Task ID</th>
                  <th className="py-2 px-3">Cell / Config</th>
                  <th className="py-2 px-3">State</th>
                  <th className="py-2 px-3 text-center">Attempt</th>
                  <th className="py-2 px-3">Message</th>
                  <th className="py-2 px-3 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-canvas-border/50">
                {filteredTasks.length === 0 ? (
                  <tr>
                    <td
                      colSpan={6}
                      className="py-6 text-center text-canvas-muted"
                    >
                      No tasks match the selected filter.
                    </td>
                  </tr>
                ) : (
                  filteredTasks.map((t) => (
                    <tr
                      key={t.task_id}
                      className="hover:bg-canvas-elevated/40 transition-colors"
                    >
                      <td className="py-2 px-3 font-mono text-gray-200">
                        {t.task_id}
                      </td>
                      <td className="py-2 px-3 text-canvas-muted font-mono text-[11px] truncate max-w-[150px]">
                        {t.cell_id}
                      </td>
                      <td className="py-2 px-3">
                        <CloudStatusBadge state={t.state} size="sm" />
                      </td>
                      <td className="py-2 px-3 text-center font-mono text-gray-300">
                        #{t.attempt}
                      </td>
                      <td className="py-2 px-3 text-canvas-muted text-[11px] truncate max-w-[200px]">
                        {t.message || "—"}
                      </td>
                      <td className="py-2 px-3 text-right">
                        <button
                          onClick={() => setLogTaskId(t.task_id)}
                          className="btn-ghost text-[11px] py-1 px-2 border border-canvas-border hover:text-accent-primary flex items-center gap-1 ml-auto"
                        >
                          <Terminal size={12} />
                          <span>Logs</span>
                        </button>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {activeTab === "metrics" && (
        <div className="space-y-3">
          {activeJob.tasks.length > 1 && (
            <select
              data-testid="metrics-task-select"
              value={metricsTask?.task_id ?? ""}
              onChange={(e) => setMetricsTaskId(e.target.value)}
              className="input text-xs py-1"
            >
              {activeJob.tasks.map((t) => (
                <option key={t.task_id} value={t.task_id}>
                  {t.cell_id} ({t.state})
                </option>
              ))}
            </select>
          )}
          {taskMetrics?.progress && (
            <p data-testid="metrics-progress" className="text-xs text-gray-400">
              {taskMetrics.progress.policy}: day {taskMetrics.progress.day}
              {taskMetrics.progress.days
                ? ` of ${taskMetrics.progress.days}`
                : ""}
            </p>
          )}
          <TaskMetricsPanel
            metrics={
              taskMetrics
                ? {
                    task_id: taskMetrics.task_id,
                    points: taskMetrics.points,
                    peak_cpu: taskMetrics.peak_cpu ?? undefined,
                    peak_memory_mb: taskMetrics.peak_memory_mb ?? undefined,
                  }
                : null
            }
          />
        </div>
      )}

      {activeTab === "json" && (
        <pre className="p-3 bg-canvas-elevated/70 border border-canvas-border rounded-lg text-xs font-mono text-gray-300 max-h-96 overflow-y-auto whitespace-pre-wrap">
          {JSON.stringify(activeJob, null, 2)}
        </pre>
      )}
    </div>
  );
}
