/**
 * Cloud Monitor — Live monitoring and manager actions for cloud routing jobs (CR-15 / §J.1).
 *
 * Displays:
 * - Jobs master list with live status indicators and state filtering.
 * - Live detail view with background polling via `watchCloudJob` (CR-13).
 * - Full manager actions: Cancel, Retry, Step Scheduler (Worker Tick), Download results.
 * - Task list breakdown with attempt numbers, states, and inline log viewer.
 * - Task resource telemetry panel (CR-12 pending empty state).
 */

import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Activity,
  Copy,
  Download,
  Layers,
  Play,
  RefreshCw,
  RotateCcw,
  Search,
  Square,
  Terminal,
  Zap,
} from "lucide-react";
import { toast } from "sonner";
import { useAppStore } from "../../store/app";
import { useCloudMonitorStore } from "../../store/cloudMonitor";
import type {
  CloudJobStatus,
  CloudJobSummary,
} from "../../types/cloud";
import {
  cancelCloudJob,
  getCloudJobStatus,
  listCloudJobs,
  retryCloudJob,
  tickCloudWorker,
  watchCloudJob,
} from "../../utils/cloud";
import { CloudStatusBadge } from "../../components/cloud/CloudStatusBadge";
import { DownloadModal } from "../../components/cloud/DownloadModal";
import { TaskLogsModal } from "../../components/cloud/TaskLogsModal";
import { TaskMetricsPanel } from "../../components/cloud/TaskMetricsPanel";

export function CloudMonitor() {
  const { projectRoot, setMode } = useAppStore();
  const {
    selectedJobId,
    stateFilter,
    searchQuery,
    autoRefresh,
    root,
    activeTab,
    setSelectedJobId,
    setStateFilter,
    setSearchQuery,
    setAutoRefresh,
    setRoot,
    setActiveTab,
  } = useCloudMonitorStore();

  const [jobs, setJobs] = useState<CloudJobSummary[]>([]);
  const [loadingJobs, setLoadingJobs] = useState<boolean>(false);
  const [activeJob, setActiveJob] = useState<CloudJobStatus | null>(null);
  const [loadingDetail, setLoadingDetail] = useState<boolean>(false);
  const [taskFilter, setTaskFilter] = useState<string>("all");

  // Modals state
  const [logTaskId, setLogTaskId] = useState<string | null>(null);
  const [downloadJobId, setDownloadJobId] = useState<string | null>(null);

  // Manager action in-progress states
  const [cancelling, setCancelling] = useState<boolean>(false);
  const [retrying, setRetrying] = useState<boolean>(false);
  const [ticking, setTicking] = useState<boolean>(false);

  // Fetch job list
  const fetchJobs = useCallback(async () => {
    setLoadingJobs(true);
    try {
      const list = await listCloudJobs({ projectRoot, root });
      setJobs(list);

      // Default select the first job if none selected
      if (!selectedJobId && list.length > 0) {
        setSelectedJobId(list[0].job_id);
      }
    } catch {
      // Ignore or log error
    } finally {
      setLoadingJobs(false);
    }
  }, [projectRoot, root, selectedJobId, setSelectedJobId]);

  useEffect(() => {
    fetchJobs();
  }, [fetchJobs]);

  // Periodic polling for the job list when auto-refresh is active
  useEffect(() => {
    if (!autoRefresh) return;
    const interval = setInterval(() => {
      fetchJobs();
    }, 5000);
    return () => clearInterval(interval);
  }, [autoRefresh, fetchJobs]);

  // Watch the selected job using CR-13's background poller
  useEffect(() => {
    if (!selectedJobId) {
      setActiveJob(null);
      return;
    }

    setLoadingDetail(true);

    // Initial fetch of detailed status
    getCloudJobStatus(selectedJobId, { projectRoot, root })
      .then((status) => {
        setActiveJob(status);
        setLoadingDetail(false);
      })
      .catch(() => {
        setLoadingDetail(false);
      });

    // Start background watcher
    let unlistenFn: (() => void) | undefined;
    watchCloudJob(
      selectedJobId,
      (event) => {
        if (event.job_id === selectedJobId) {
          setActiveJob(event.status);
        }
      },
      2000,
      { projectRoot, root }
    )
      .then((unlisten) => {
        unlistenFn = unlisten;
      })
      .catch(() => {});

    return () => {
      if (unlistenFn) unlistenFn();
    };
  }, [selectedJobId, projectRoot, root]);

  // Filtered jobs
  const filteredJobs = useMemo(() => {
    return jobs.filter((job) => {
      if (stateFilter !== "all" && job.state.toLowerCase() !== stateFilter.toLowerCase()) {
        return false;
      }
      if (searchQuery.trim()) {
        const q = searchQuery.toLowerCase();
        return (
          job.job_id.toLowerCase().includes(q) ||
          job.name.toLowerCase().includes(q)
        );
      }
      return true;
    });
  }, [jobs, stateFilter, searchQuery]);

  // Filtered tasks for active job
  const filteredTasks = useMemo(() => {
    if (!activeJob) return [];
    if (taskFilter === "all") return activeJob.tasks;
    return activeJob.tasks.filter(
      (t) => t.state.toLowerCase() === taskFilter.toLowerCase()
    );
  }, [activeJob, taskFilter]);

  // Calculate task summary counts for active job
  const taskCounts = useMemo(() => {
    if (!activeJob) return { total: 0, succeeded: 0, running: 0, failed: 0, queued: 0 };
    const counts = { total: activeJob.tasks.length, succeeded: 0, running: 0, failed: 0, queued: 0 };
    for (const t of activeJob.tasks) {
      const s = t.state.toLowerCase();
      if (s === "succeeded") counts.succeeded++;
      else if (s === "running" || s === "starting" || s === "placing") counts.running++;
      else if (s === "failed" || s === "preempted") counts.failed++;
      else counts.queued++;
    }
    return counts;
  }, [activeJob]);

  // Manager Actions
  const handleCancelJob = async () => {
    if (!activeJob) return;
    if (!confirm(`Cancel job ${activeJob.job_id} and all remaining tasks?`)) return;

    setCancelling(true);
    try {
      await cancelCloudJob(activeJob.job_id, { projectRoot, root });
      toast.success(`Cancellation requested for job ${activeJob.job_id}`);
      fetchJobs();
    } catch (err: unknown) {
      toast.error("Failed to cancel job", {
        description: err instanceof Error ? err.message : String(err),
      });
    } finally {
      setCancelling(false);
    }
  };

  const handleRetryJob = async () => {
    if (!activeJob) return;

    setRetrying(true);
    try {
      const res = await retryCloudJob(activeJob.job_id, { projectRoot, root });
      toast.success(`Requeued ${res.retried} task(s) for job ${activeJob.job_id}`);
      fetchJobs();
    } catch (err: unknown) {
      toast.error("Failed to retry job", {
        description: err instanceof Error ? err.message : String(err),
      });
    } finally {
      setRetrying(false);
    }
  };

  const handleWorkerTick = async () => {
    setTicking(true);
    try {
      const res = await tickCloudWorker(activeJob?.job_id, { projectRoot, root });
      toast.success(`Scheduler step complete: ${res.changes} state change(s)`);
      fetchJobs();
    } catch (err: unknown) {
      toast.error("Scheduler tick failed", {
        description: err instanceof Error ? err.message : String(err),
      });
    } finally {
      setTicking(false);
    }
  };

  const handleCopyJobId = () => {
    if (!activeJob) return;
    navigator.clipboard.writeText(activeJob.job_id);
    toast.success("Job ID copied to clipboard");
  };

  const isTerminal =
    activeJob?.state === "succeeded" ||
    activeJob?.state === "failed" ||
    activeJob?.state === "cancelled";

  return (
    <div className="p-6 max-w-7xl mx-auto space-y-6">
      {/* Top Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-canvas-border pb-4">
        <div>
          <div className="flex items-center gap-2">
            <Activity size={24} className="text-accent-primary" />
            <h1 className="text-xl font-bold tracking-tight text-white">
              Cloud Job Monitor
            </h1>
          </div>
          <p className="text-xs text-canvas-muted mt-1">
            Real-time status tracking, cancellation, task log inspection, and results
            downloads across local and cloud environments.
          </p>
        </div>

        <div className="flex items-center gap-2">
          {/* Root state directory */}
          <div className="flex items-center gap-1.5 text-xs text-canvas-muted">
            <span>Root:</span>
            <input
              type="text"
              className="input-base font-mono text-xs w-32 py-1 px-2"
              value={root}
              onChange={(e) => setRoot(e.target.value)}
              placeholder="outputs/cloud"
            />
          </div>

          {/* Auto Refresh toggle */}
          <label className="flex items-center gap-1.5 text-xs text-canvas-muted cursor-pointer select-none px-2 py-1 bg-canvas-elevated border border-canvas-border rounded-lg">
            <input
              type="checkbox"
              checked={autoRefresh}
              onChange={(e) => setAutoRefresh(e.target.checked)}
              className="rounded border-canvas-border bg-canvas-surface text-accent-primary"
            />
            <span>Auto Refresh</span>
          </label>

          {/* Refresh button */}
          <button
            onClick={fetchJobs}
            disabled={loadingJobs}
            className="btn-ghost text-xs p-1.5 border border-canvas-border"
            title="Refresh job list"
          >
            <RefreshCw size={14} className={loadingJobs ? "animate-spin" : ""} />
          </button>

          {/* Launch shortcut */}
          <button
            onClick={() => setMode("cloud_launcher" as any)}
            className="btn-primary text-xs flex items-center gap-1 px-3 py-1.5"
          >
            <Play size={13} />
            <span>Launch New Job</span>
          </button>
        </div>
      </div>

      {/* Main Grid: Master List (Left) + Job Detail (Right) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Left Column: Job Master List */}
        <div className="lg:col-span-4 card bg-canvas-surface border border-canvas-border rounded-xl p-4 space-y-3">
          <div className="flex items-center justify-between pb-2 border-b border-canvas-border">
            <h3 className="text-sm font-semibold text-gray-200">
              Cloud Jobs ({jobs.length})
            </h3>
            <button
              onClick={handleWorkerTick}
              disabled={ticking}
              className="text-xs text-canvas-muted hover:text-accent-primary flex items-center gap-1 transition-colors"
              title="Run a single scheduler placement tick"
            >
              <Zap size={12} className={ticking ? "animate-spin text-accent-primary" : ""} />
              <span>Step Scheduler</span>
            </button>
          </div>

          {/* Search & Filter */}
          <div className="space-y-2">
            <div className="relative">
              <input
                type="text"
                placeholder="Search job ID or name..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="input-base text-xs pl-8 py-1.5"
              />
              <Search
                size={14}
                className="absolute left-2.5 top-2 text-canvas-muted"
              />
            </div>

            {/* State filter buttons */}
            <div className="flex flex-wrap gap-1 text-[11px]">
              {["all", "running", "succeeded", "failed", "queued"].map((st) => (
                <button
                  key={st}
                  onClick={() => setStateFilter(st)}
                  className={`px-2 py-0.5 rounded capitalize transition-colors ${
                    stateFilter === st
                      ? "bg-accent-primary text-white font-medium"
                      : "bg-canvas-elevated text-canvas-muted hover:text-gray-200"
                  }`}
                >
                  {st}
                </button>
              ))}
            </div>
          </div>

          {/* Jobs list */}
          <div className="space-y-2 max-h-[600px] overflow-y-auto pr-1">
            {filteredJobs.length === 0 ? (
              <div className="text-center py-10 text-xs text-canvas-muted">
                {jobs.length === 0
                  ? "No cloud jobs found in state directory."
                  : "No jobs matching search filter."}
              </div>
            ) : (
              filteredJobs.map((j) => {
                const isSelected = selectedJobId === j.job_id;
                const totalTasks = Object.values(j.counts).reduce(
                  (a, b) => a + b,
                  0
                );
                const succeededTasks = j.counts["succeeded"] ?? 0;
                return (
                  <button
                    key={j.job_id}
                    onClick={() => setSelectedJobId(j.job_id)}
                    className={`w-full text-left p-3 rounded-lg border transition-all ${
                      isSelected
                        ? "bg-canvas-elevated border-accent-primary shadow-sm"
                        : "bg-canvas-elevated/30 border-canvas-border hover:border-canvas-muted/60 hover:bg-canvas-elevated/60"
                    }`}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <span className="font-semibold text-xs text-gray-200 truncate">
                        {j.name || j.job_id}
                      </span>
                      <CloudStatusBadge state={j.state} size="sm" />
                    </div>

                    <div className="flex items-center justify-between text-[11px] text-canvas-muted mt-2">
                      <span className="font-mono">{j.job_id.slice(0, 16)}</span>
                      <span>
                        {succeededTasks}/{totalTasks} tasks done
                      </span>
                    </div>
                  </button>
                );
              })
            )}
          </div>
        </div>

        {/* Right Column: Selected Job Details & Actions */}
        <div className="lg:col-span-8 space-y-4">
          {!activeJob ? (
            <div className="card bg-canvas-surface border border-canvas-border rounded-xl p-12 text-center space-y-3">
              <Layers size={36} className="mx-auto text-canvas-muted" />
              <h3 className="text-base font-semibold text-gray-200">
                No Cloud Job Selected
              </h3>
              <p className="text-xs text-canvas-muted max-w-sm mx-auto">
                Select a job from the list on the left to monitor live task execution,
                inspect logs, or trigger management actions.
              </p>
            </div>
          ) : (
            <div className="space-y-4">
              {/* Job Header Card with Manager Actions */}
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

                  {/* Action Toolbar */}
                  <div className="flex flex-wrap items-center gap-2">
                    {/* Retry button */}
                    <button
                      onClick={handleRetryJob}
                      disabled={retrying || (taskCounts.failed === 0 && activeJob.state !== "cancelled")}
                      className="btn-ghost text-xs flex items-center gap-1.5 px-3 py-1.5 border border-canvas-border disabled:opacity-40"
                      title="Requeue failed and cancelled tasks"
                    >
                      <RotateCcw
                        size={13}
                        className={retrying ? "animate-spin" : ""}
                      />
                      <span>Retry Failed</span>
                    </button>

                    {/* Download button */}
                    <button
                      onClick={() => setDownloadJobId(activeJob.job_id)}
                      className="btn-ghost text-xs flex items-center gap-1.5 px-3 py-1.5 border border-canvas-border"
                      title="Download results index and daily checkpoints"
                    >
                      <Download size={13} />
                      <span>Download</span>
                    </button>

                    {/* Cancel button */}
                    {!isTerminal && (
                      <button
                        onClick={handleCancelJob}
                        disabled={cancelling}
                        className="btn-ghost text-xs text-accent-danger hover:bg-accent-danger/10 flex items-center gap-1.5 px-3 py-1.5 border border-accent-danger/30"
                        title="Stop all active and queued tasks"
                      >
                        <Square
                          size={13}
                          className={cancelling ? "animate-spin" : ""}
                        />
                        <span>Cancel Job</span>
                      </button>
                    )}
                  </div>
                </div>

                {/* Progress bar */}
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

              {/* Tabs Container */}
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

                {/* Tab 1: Tasks Table */}
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

                {/* Tab 2: Resource Telemetry (CR-12 pending) */}
                {activeTab === "metrics" && (
                  <TaskMetricsPanel metrics={null} />
                )}

                {/* Tab 3: Raw Status JSON */}
                {activeTab === "json" && (
                  <pre className="p-3 bg-canvas-elevated/70 border border-canvas-border rounded-lg text-xs font-mono text-gray-300 max-h-96 overflow-y-auto whitespace-pre-wrap">
                    {JSON.stringify(activeJob, null, 2)}
                  </pre>
                )}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Task Logs Modal */}
      <TaskLogsModal
        taskId={logTaskId}
        isOpen={logTaskId != null}
        onClose={() => setLogTaskId(null)}
        projectRoot={projectRoot}
        root={root}
      />

      {/* Download Results Modal */}
      <DownloadModal
        jobId={downloadJobId}
        isOpen={downloadJobId != null}
        onClose={() => setDownloadJobId(null)}
        projectRoot={projectRoot}
        root={root}
      />
    </div>
  );
}
