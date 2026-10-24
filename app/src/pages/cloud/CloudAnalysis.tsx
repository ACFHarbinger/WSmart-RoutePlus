/**
 * Cloud Analysis Dashboard — Post-run aggregate analytics, execution timeline,
 * cost accrual (CR-10), runtime prediction accuracy (CR-11), and task table (CR-16 / §J.1).
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Activity,
  BarChart3,
  DollarSign,
  Download,
  ExternalLink,
  FileSpreadsheet,
  Gauge,
  Layers,
  LineChart,
  RefreshCw,
} from "lucide-react";
import { toast } from "sonner";
import { useAppStore } from "../../store/app";
import { useCloudAnalysisStore, type AnalysisTab } from "../../store/cloudAnalysis";
import type { CloudJobStatus, CloudJobSummary } from "../../types/cloud";
import {
  computeJobAnalysis,
  exportTasksToCsv,
  getCloudJobStatus,
  listCloudJobs,
} from "../../utils/cloud";
import { CloudStatusBadge } from "../../components/cloud/CloudStatusBadge";
import {
  CloudAnalysisKpiStrip,
  CloudAnalysisTabPanels,
} from "../../components/cloud/CloudAnalysisSections";
import { DownloadModal } from "../../components/cloud/DownloadModal";
import { TaskLogsModal } from "../../components/cloud/TaskLogsModal";

export function CloudAnalysis() {
  const { projectRoot, setMode } = useAppStore();
  const {
    selectedJobId,
    activeTab,
    taskFilter,
    searchQuery,
    sortBy,
    sortDirection,
    root,
    setSelectedJobId,
    setActiveTab,
    setTaskFilter,
    setSearchQuery,
    setSort,
  } = useCloudAnalysisStore();

  const [jobs, setJobs] = useState<CloudJobSummary[]>([]);
  const [loadingJobs, setLoadingJobs] = useState<boolean>(false);
  const [activeJob, setActiveJob] = useState<CloudJobStatus | null>(null);
  const [loadingDetail, setLoadingDetail] = useState<boolean>(false);

  const [downloadJobId, setDownloadJobId] = useState<string | null>(null);
  const [logTaskId, setLogTaskId] = useState<string | null>(null);

  const fetchJobs = useCallback(async () => {
    setLoadingJobs(true);
    try {
      const list = await listCloudJobs({ projectRoot, root });
      setJobs(list);
      if (!selectedJobId && list.length > 0) {
        setSelectedJobId(list[0].job_id);
      }
    } catch (err: any) {
      toast.error(`Failed to list cloud jobs: ${err?.message || err}`);
    } finally {
      setLoadingJobs(false);
    }
  }, [projectRoot, root, selectedJobId, setSelectedJobId]);

  useEffect(() => {
    fetchJobs();
  }, [fetchJobs]);

  const detailGeneration = useRef(0);

  const fetchJobDetail = useCallback(
    async (jobId: string) => {
      const generation = ++detailGeneration.current;
      setActiveJob(null);
      setLoadingDetail(true);
      try {
        const detail = await getCloudJobStatus(jobId, { projectRoot, root });
        if (generation === detailGeneration.current) setActiveJob(detail);
      } catch (err: any) {
        toast.error(`Failed to load job status for ${jobId}: ${err?.message || err}`);
      } finally {
        if (generation === detailGeneration.current) setLoadingDetail(false);
      }
    },
    [projectRoot, root]
  );

  useEffect(() => {
    if (selectedJobId) {
      fetchJobDetail(selectedJobId);
    } else {
      setActiveJob(null);
    }
    return () => {
      detailGeneration.current++;
    };
  }, [selectedJobId, fetchJobDetail]);

  const analysis = useMemo(() => {
    if (!activeJob) return null;
    return computeJobAnalysis(activeJob);
  }, [activeJob]);

  const filteredTasks = useMemo(() => {
    if (!analysis) return [];
    let list = [...analysis.tasks];

    if (taskFilter !== "all") {
      list = list.filter((t) => t.state.toLowerCase() === taskFilter.toLowerCase());
    }

    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase().trim();
      list = list.filter(
        (t) =>
          t.task_id.toLowerCase().includes(q) ||
          t.cell_id.toLowerCase().includes(q) ||
          (t.policy && t.policy.toLowerCase().includes(q)) ||
          (t.message && t.message.toLowerCase().includes(q))
      );
    }

    list.sort((a, b) => {
      let comparison = 0;
      if (sortBy === "task_id") comparison = a.task_id.localeCompare(b.task_id);
      else if (sortBy === "cell_id") comparison = a.cell_id.localeCompare(b.cell_id);
      else if (sortBy === "state") comparison = a.state.localeCompare(b.state);
      else if (sortBy === "duration") comparison = (a.duration_seconds ?? -1) - (b.duration_seconds ?? -1);
      else if (sortBy === "cost") comparison = (a.cost_usd ?? -1) - (b.cost_usd ?? -1);
      return sortDirection === "asc" ? comparison : -comparison;
    });

    return list;
  }, [analysis, taskFilter, searchQuery, sortBy, sortDirection]);

  const handleExportCsv = () => {
    if (!analysis || filteredTasks.length === 0) {
      toast.warning("No task records available to export");
      return;
    }
    const filename = `cloud_job_${analysis.job_id}_tasks.csv`;
    exportTasksToCsv(filteredTasks, filename);
    toast.success(`Exported ${filteredTasks.length} tasks to ${filename}`);
  };

  const policySummary = useMemo(() => {
    if (!analysis) return [];
    const map = new Map<string, { count: number; computeSec: number | null; costUsd: number | null }>();

    for (const t of analysis.tasks) {
      if (!t.policy) continue;
      const pol = t.policy;
      const curr = map.get(pol) || { count: 0, computeSec: 0, costUsd: 0 };
      curr.count += 1;
      curr.computeSec =
        curr.computeSec == null || t.duration_seconds == null
          ? null
          : curr.computeSec + t.duration_seconds;
      curr.costUsd =
        curr.costUsd == null || t.cost_usd == null ? null : curr.costUsd + t.cost_usd;
      map.set(pol, curr);
    }

    return Array.from(map.entries()).map(([policy, data]) => ({
      policy,
      ...data,
    }));
  }, [analysis]);

  return (
    <div className="flex flex-col h-full bg-neutral-950 text-neutral-100 overflow-y-auto">
      <header className="px-6 py-4 bg-neutral-900 border-b border-neutral-800 flex flex-wrap items-center justify-between gap-4 sticky top-0 z-10">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded bg-indigo-500/10 text-indigo-400">
            <BarChart3 size={20} />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-base font-semibold text-neutral-100">Cloud Analysis Dashboard</h1>
              <span className="text-[10px] uppercase font-mono px-1.5 py-0.5 rounded bg-neutral-800 text-neutral-400 border border-neutral-700">
                CR-16
              </span>
            </div>
            <p className="text-xs text-neutral-400">
              Post-run performance metrics, task duration waterfalls, cost telemetry & runtime predictions
            </p>
          </div>
        </div>

        <div className="flex items-center flex-wrap gap-2.5">
          <div className="relative">
            <select
              value={selectedJobId || ""}
              onChange={(e) => setSelectedJobId(e.target.value || null)}
              className="bg-neutral-950 border border-neutral-700 text-xs rounded px-3 py-1.5 text-neutral-200 focus:outline-none focus:border-indigo-500 font-mono pr-8 cursor-pointer"
            >
              {jobs.length === 0 ? (
                <option value="">No cloud jobs found</option>
              ) : (
                jobs.map((j) => (
                  <option key={j.job_id} value={j.job_id}>
                    {j.name} ({j.job_id}) — [{j.state}]
                  </option>
                ))
              )}
            </select>
          </div>

          {activeJob && <CloudStatusBadge state={activeJob.state} />}

          <button
            onClick={() => {
              fetchJobs();
              if (selectedJobId) fetchJobDetail(selectedJobId);
            }}
            disabled={loadingJobs || loadingDetail}
            className="p-1.5 text-neutral-400 hover:text-neutral-200 bg-neutral-800 hover:bg-neutral-700 rounded transition-colors"
            title="Refresh Analysis"
          >
            <RefreshCw size={15} className={loadingJobs || loadingDetail ? "animate-spin" : ""} />
          </button>

          <button
            onClick={() => {
              if (selectedJobId) setDownloadJobId(selectedJobId);
            }}
            disabled={!selectedJobId}
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium bg-neutral-800 hover:bg-neutral-700 text-neutral-200 rounded transition-colors disabled:opacity-50"
          >
            <Download size={14} />
            <span>Download Results</span>
          </button>

          <button
            onClick={() => setMode("simulation_summary")}
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium bg-indigo-600/20 hover:bg-indigo-600/30 text-indigo-300 border border-indigo-500/30 rounded transition-colors"
            title="Switch view to Simulation Summary"
          >
            <LineChart size={14} />
            <span>Sim Summary</span>
            <ExternalLink size={12} className="opacity-70" />
          </button>

          <button
            onClick={() => setMode("cloud_monitor")}
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium bg-neutral-800 hover:bg-neutral-700 text-neutral-300 rounded transition-colors"
            title="Switch to Live Monitor"
          >
            <Activity size={14} />
            <span>Live Monitor</span>
          </button>
        </div>
      </header>

      <main className="p-6 flex-1 flex flex-col gap-6">
        {!activeJob || !analysis ? (
          <div className="flex-1 flex flex-col items-center justify-center py-20 text-center">
            <Layers className="w-12 h-12 text-neutral-600 mb-3" />
            <h3 className="text-base font-semibold text-neutral-300">No Cloud Job Selected</h3>
            <p className="text-xs text-neutral-500 max-w-sm mt-1">
              Select a job from the top dropdown or launch a new run in Cloud Launcher to view detailed analysis.
            </p>
          </div>
        ) : (
          <>
            <CloudAnalysisKpiStrip analysis={analysis} />

            <div className="flex items-center justify-between border-b border-neutral-800">
              <nav className="flex items-center gap-1">
                {[
                  { id: "overview" as AnalysisTab, label: "Overview & Telemetry", icon: Gauge },
                  { id: "waterfall" as AnalysisTab, label: "Duration Comparison", icon: BarChart3 },
                  { id: "tasks" as AnalysisTab, label: "Task Inspection Table", icon: Layers },
                  { id: "cost" as AnalysisTab, label: "Financial & Spot Breakdown", icon: DollarSign },
                ].map((tab) => {
                  const Icon = tab.icon;
                  const isActive = activeTab === tab.id;
                  return (
                    <button
                      key={tab.id}
                      onClick={() => setActiveTab(tab.id)}
                      className={`flex items-center gap-2 px-4 py-2.5 text-xs font-medium border-b-2 transition-colors ${
                        isActive
                          ? "border-indigo-500 text-indigo-400 bg-neutral-900/40"
                          : "border-transparent text-neutral-400 hover:text-neutral-200 hover:border-neutral-700"
                      }`}
                    >
                      <Icon size={14} />
                      <span>{tab.label}</span>
                    </button>
                  );
                })}
              </nav>

              {activeTab === "tasks" && (
                <button
                  onClick={handleExportCsv}
                  className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium bg-neutral-800 hover:bg-neutral-700 text-neutral-200 rounded transition-colors"
                  title="Export filtered tasks to CSV"
                >
                  <FileSpreadsheet size={14} className="text-emerald-400" />
                  <span>Export CSV</span>
                </button>
              )}
            </div>

            <CloudAnalysisTabPanels
              activeTab={activeTab}
              analysis={analysis}
              policySummary={policySummary}
              searchQuery={searchQuery}
              setSearchQuery={setSearchQuery}
              taskFilter={taskFilter}
              setTaskFilter={setTaskFilter}
              filteredTasks={filteredTasks}
              setSort={setSort}
              setLogTaskId={setLogTaskId}
            />
          </>
        )}
      </main>

      <DownloadModal
        jobId={downloadJobId}
        isOpen={Boolean(downloadJobId)}
        onClose={() => setDownloadJobId(null)}
        projectRoot={projectRoot}
        root={root}
      />

      <TaskLogsModal
        taskId={logTaskId}
        isOpen={Boolean(logTaskId)}
        onClose={() => setLogTaskId(null)}
        projectRoot={projectRoot}
        root={root}
      />
    </div>
  );
}
