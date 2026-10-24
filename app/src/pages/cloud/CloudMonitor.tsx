/**
 * Cloud Monitor — Live monitoring and manager actions for cloud routing jobs (CR-15 / §J.1).
 */
import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Activity,
  Layers,
  Play,
  RefreshCw,
  Search,
  Zap,
} from "lucide-react";
import { toast } from "sonner";
import { useAppStore } from "../../store/app";
import { useCloudMonitorStore } from "../../store/cloudMonitor";
import type {
  CloudJobStatus,
  CloudJobSummary,
  CloudTaskMetrics,
} from "../../types/cloud";
import {
  cancelCloudJob,
  getCloudJobStatus,
  getCloudTaskMetrics,
  listCloudJobs,
  retryCloudJob,
  tickCloudWorker,
  watchCloudJob,
} from "../../utils/cloud";
import { CloudStatusBadge } from "../../components/cloud/CloudStatusBadge";
import { DownloadModal } from "../../components/cloud/DownloadModal";
import { TaskLogsModal } from "../../components/cloud/TaskLogsModal";
import {
  CloudJobDetailHeader,
  CloudJobDetailTabs,
} from "../../components/cloud/CloudMonitorDetail";

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

  const [logTaskId, setLogTaskId] = useState<string | null>(null);
  const [metricsTaskId, setMetricsTaskId] = useState<string | null>(null);
  const [taskMetrics, setTaskMetrics] = useState<CloudTaskMetrics | null>(null);
  const [downloadJobId, setDownloadJobId] = useState<string | null>(null);

  const [cancelling, setCancelling] = useState<boolean>(false);
  const [retrying, setRetrying] = useState<boolean>(false);
  const [ticking, setTicking] = useState<boolean>(false);

  const fetchJobs = useCallback(async () => {
    setLoadingJobs(true);
    try {
      const list = await listCloudJobs({ projectRoot, root });
      setJobs(list);
      if (!selectedJobId && list.length > 0) {
        setSelectedJobId(list[0].job_id);
      }
    } catch {
      // Ignore
    } finally {
      setLoadingJobs(false);
    }
  }, [projectRoot, root, selectedJobId, setSelectedJobId]);

  useEffect(() => {
    void fetchJobs();
  }, [fetchJobs]);

  const metricsTask = useMemo(() => {
    const tasks = activeJob?.tasks ?? [];
    return (
      tasks.find((t) => t.task_id === metricsTaskId) ??
      tasks.find((t) => t.state === "running") ??
      tasks[0] ??
      null
    );
  }, [activeJob, metricsTaskId]);

  useEffect(() => {
    if (activeTab !== "metrics" || !metricsTask) {
      setTaskMetrics(null);
      return;
    }
    let cancelled = false;
    const load = () =>
      getCloudTaskMetrics(metricsTask.task_id, undefined, { projectRoot, root })
        .then((m) => {
          if (!cancelled) setTaskMetrics(m);
        })
        .catch(() => {
          if (!cancelled) setTaskMetrics(null);
        });
    void load();
    const interval = autoRefresh ? setInterval(() => void load(), 10000) : undefined;
    return () => {
      cancelled = true;
      if (interval) clearInterval(interval);
    };
  }, [activeTab, metricsTask, autoRefresh, projectRoot, root]);

  useEffect(() => {
    if (!autoRefresh) return;
    const interval = setInterval(() => {
      void fetchJobs();
    }, 5000);
    return () => clearInterval(interval);
  }, [autoRefresh, fetchJobs]);

  useEffect(() => {
    if (!selectedJobId) {
      setActiveJob(null);
      return;
    }

    setLoadingDetail(true);

    getCloudJobStatus(selectedJobId, { projectRoot, root })
      .then((status) => {
        setActiveJob(status);
        setLoadingDetail(false);
      })
      .catch(() => {
        setLoadingDetail(false);
      });

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

  const filteredJobs = useMemo(() => {
    return jobs.filter((job) => {
      if (
        stateFilter !== "all" &&
        job.state.toLowerCase() !== stateFilter.toLowerCase()
      ) {
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

  const filteredTasks = useMemo(() => {
    if (!activeJob) return [];
    if (taskFilter === "all") return activeJob.tasks;
    return activeJob.tasks.filter(
      (t) => t.state.toLowerCase() === taskFilter.toLowerCase()
    );
  }, [activeJob, taskFilter]);

  const taskCounts = useMemo(() => {
    if (!activeJob) {
      return { total: 0, succeeded: 0, running: 0, failed: 0, queued: 0 };
    }
    const counts = {
      total: activeJob.tasks.length,
      succeeded: 0,
      running: 0,
      failed: 0,
      queued: 0,
    };
    for (const t of activeJob.tasks) {
      const s = t.state.toLowerCase();
      if (s === "succeeded") counts.succeeded++;
      else if (s === "running" || s === "starting" || s === "placing") {
        counts.running++;
      } else if (s === "failed" || s === "preempted") counts.failed++;
      else counts.queued++;
    }
    return counts;
  }, [activeJob]);

  const handleCancelJob = async () => {
    if (!activeJob) return;
    if (!confirm(`Cancel job ${activeJob.job_id} and all remaining tasks?`)) {
      return;
    }

    setCancelling(true);
    try {
      await cancelCloudJob(activeJob.job_id, { projectRoot, root });
      toast.success(`Cancellation requested for job ${activeJob.job_id}`);
      void fetchJobs();
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
      toast.success(
        `Requeued ${res.retried} task(s) for job ${activeJob.job_id}`
      );
      void fetchJobs();
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
      const res = await tickCloudWorker(activeJob?.job_id, {
        projectRoot,
        root,
      });
      toast.success(
        `Scheduler step complete: ${res.changes} state change(s)`
      );
      void fetchJobs();
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
    void navigator.clipboard.writeText(activeJob.job_id);
    toast.success("Job ID copied to clipboard");
  };

  const isTerminal =
    activeJob?.state === "succeeded" ||
    activeJob?.state === "failed" ||
    activeJob?.state === "cancelled";

  return (
    <div className="p-6 max-w-7xl mx-auto space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-canvas-border pb-4">
        <div>
          <div className="flex items-center gap-2">
            <Activity size={24} className="text-accent-primary" />
            <h1 className="text-xl font-bold tracking-tight text-white">
              Cloud Job Monitor
            </h1>
          </div>
          <p className="text-xs text-canvas-muted mt-1">
            Real-time status tracking, cancellation, task log inspection, and
            results downloads across local and cloud environments.
          </p>
        </div>

        <div className="flex items-center gap-2">
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

          <label className="flex items-center gap-1.5 text-xs text-canvas-muted cursor-pointer select-none px-2 py-1 bg-canvas-elevated border border-canvas-border rounded-lg">
            <input
              type="checkbox"
              checked={autoRefresh}
              onChange={(e) => setAutoRefresh(e.target.checked)}
              className="rounded border-canvas-border bg-canvas-surface text-accent-primary"
            />
            <span>Auto Refresh</span>
          </label>

          <button
            onClick={() => void fetchJobs()}
            disabled={loadingJobs}
            className="btn-ghost text-xs p-1.5 border border-canvas-border"
            title="Refresh job list"
          >
            <RefreshCw size={14} className={loadingJobs ? "animate-spin" : ""} />
          </button>

          <button
            onClick={() => setMode("cloud_launcher")}
            className="btn-primary text-xs flex items-center gap-1 px-3 py-1.5"
          >
            <Play size={13} />
            <span>Launch New Job</span>
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        <div className="lg:col-span-4 card bg-canvas-surface border border-canvas-border rounded-xl p-4 space-y-3">
          <div className="flex items-center justify-between pb-2 border-b border-canvas-border">
            <h3 className="text-sm font-semibold text-gray-200">
              Cloud Jobs ({jobs.length})
            </h3>
            <button
              onClick={() => void handleWorkerTick()}
              disabled={ticking}
              className="text-xs text-canvas-muted hover:text-accent-primary flex items-center gap-1 transition-colors"
              title="Run a single scheduler placement tick"
            >
              <Zap
                size={12}
                className={ticking ? "animate-spin text-accent-primary" : ""}
              />
              <span>Step Scheduler</span>
            </button>
          </div>

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

        <div className="lg:col-span-8 space-y-4">
          {!activeJob ? (
            <div className="card bg-canvas-surface border border-canvas-border rounded-xl p-12 text-center space-y-3">
              <Layers size={36} className="mx-auto text-canvas-muted" />
              <h3 className="text-base font-semibold text-gray-200">
                No Cloud Job Selected
              </h3>
              <p className="text-xs text-canvas-muted max-w-sm mx-auto">
                Select a job from the list on the left to monitor live task
                execution, inspect logs, or trigger management actions.
              </p>
            </div>
          ) : (
            <div className="space-y-4">
              <CloudJobDetailHeader
                activeJob={activeJob}
                loadingDetail={loadingDetail}
                handleCopyJobId={handleCopyJobId}
                handleRetryJob={() => void handleRetryJob()}
                retrying={retrying}
                taskCounts={taskCounts}
                setDownloadJobId={setDownloadJobId}
                isTerminal={isTerminal}
                handleCancelJob={() => void handleCancelJob()}
                cancelling={cancelling}
              />

              <CloudJobDetailTabs
                activeJob={activeJob}
                activeTab={activeTab}
                setActiveTab={setActiveTab}
                taskFilter={taskFilter}
                setTaskFilter={setTaskFilter}
                filteredTasks={filteredTasks}
                setLogTaskId={setLogTaskId}
                metricsTask={metricsTask}
                setMetricsTaskId={setMetricsTaskId}
                taskMetrics={taskMetrics}
              />
            </div>
          )}
        </div>
      </div>

      <TaskLogsModal
        taskId={logTaskId}
        isOpen={logTaskId != null}
        onClose={() => setLogTaskId(null)}
        projectRoot={projectRoot}
        root={root}
      />
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
