import { formatObserved } from "../../utils/cloud/analysis";
/**
 * Cloud Analysis Dashboard — Post-run aggregate analytics, execution timeline,
 * cost accrual (CR-10), runtime prediction accuracy (CR-11), and task table (CR-16 / §J.1).
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Activity,
  ArrowUpDown,
  BarChart3,
  Clock,
  Cpu,
  DollarSign,
  Download,
  ExternalLink,
  FileSpreadsheet,
  Gauge,
  Layers,
  LineChart,
  RefreshCw,
  Search,
  Terminal,
  Zap,
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
import { CostAccrualCard } from "../../components/cloud/CostAccrualCard";
import { DownloadModal } from "../../components/cloud/DownloadModal";
import { PredictionAccuracyCard } from "../../components/cloud/PredictionAccuracyCard";
import { TaskDurationWaterfall } from "../../components/cloud/TaskDurationWaterfall";
import { TaskLogsModal } from "../../components/cloud/TaskLogsModal";

function formatDuration(seconds: number | null): string {
  if (seconds == null) return "Unavailable";
  if (!seconds || seconds <= 0) return "0.0s";
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  if (mins < 60) return `${mins}m ${secs}s`;
  const hours = Math.floor(mins / 60);
  const remMins = mins % 60;
  return `${hours}h ${remMins}m ${secs}s`;
}

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

  // Modals
  const [downloadJobId, setDownloadJobId] = useState<string | null>(null);
  const [logTaskId, setLogTaskId] = useState<string | null>(null);

  // Load jobs list
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

  // Load active job details
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
    return () => { detailGeneration.current++; };
  }, [selectedJobId, fetchJobDetail]);

  // Derived analysis calculations
  const analysis = useMemo(() => {
    if (!activeJob) return null;
    return computeJobAnalysis(activeJob);
  }, [activeJob]);

  // Filtered and sorted tasks
  const filteredTasks = useMemo(() => {
    if (!analysis) return [];
    let list = [...analysis.tasks];

    // State filter
    if (taskFilter !== "all") {
      list = list.filter((t) => t.state.toLowerCase() === taskFilter.toLowerCase());
    }

    // Search query
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

    // Sorting
    list.sort((a, b) => {
      let comparison = 0;
      if (sortBy === "task_id") {
        comparison = a.task_id.localeCompare(b.task_id);
      } else if (sortBy === "cell_id") {
        comparison = a.cell_id.localeCompare(b.cell_id);
      } else if (sortBy === "state") {
        comparison = a.state.localeCompare(b.state);
      } else if (sortBy === "duration") {
        comparison = (a.duration_seconds ?? -1) - (b.duration_seconds ?? -1);
      } else if (sortBy === "cost") {
        comparison = (a.cost_usd ?? -1) - (b.cost_usd ?? -1);
      }
      return sortDirection === "asc" ? comparison : -comparison;
    });

    return list;
  }, [analysis, taskFilter, searchQuery, sortBy, sortDirection]);

  // CSV export handler
  const handleExportCsv = () => {
    if (!analysis || filteredTasks.length === 0) {
      toast.warning("No task records available to export");
      return;
    }
    const filename = `cloud_job_${analysis.job_id}_tasks.csv`;
    exportTasksToCsv(filteredTasks, filename);
    toast.success(`Exported ${filteredTasks.length} tasks to ${filename}`);
  };

  // Grouped policy metrics
  const policySummary = useMemo(() => {
    if (!analysis) return [];
    const map = new Map<string, { count: number; computeSec: number | null; costUsd: number | null }>();

    for (const t of analysis.tasks) {
      if (!t.policy) continue;
      const pol = t.policy;
      const curr = map.get(pol) || { count: 0, computeSec: 0, costUsd: 0 };
      curr.count += 1;
      curr.computeSec = curr.computeSec == null || t.duration_seconds == null ? null : curr.computeSec + t.duration_seconds;
      curr.costUsd = curr.costUsd == null || t.cost_usd == null ? null : curr.costUsd + t.cost_usd;
      map.set(pol, curr);
    }

    return Array.from(map.entries()).map(([policy, data]) => ({
      policy,
      ...data,
    }));
  }, [analysis]);

  return (
    <div className="flex flex-col h-full bg-neutral-950 text-neutral-100 overflow-y-auto">
      {/* Top Header / Master Controls */}
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

        {/* Master Job Selector & Actions */}
        <div className="flex items-center flex-wrap gap-2.5">
          {/* Job Select Dropdown */}
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

          {/* Download Results */}
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

          {/* Simulation Summary Link */}
          <button
            onClick={() => setMode("simulation_summary")}
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium bg-indigo-600/20 hover:bg-indigo-600/30 text-indigo-300 border border-indigo-500/30 rounded transition-colors"
            title="Switch view to Simulation Summary"
          >
            <LineChart size={14} />
            <span>Sim Summary</span>
            <ExternalLink size={12} className="opacity-70" />
          </button>

          {/* Live Monitor Link */}
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

      {/* Main Body */}
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
            {/* Top-Level KPI Summary Cards Strip */}
            <div className="grid grid-cols-2 md:grid-cols-5 gap-3.5">
              {/* Total Tasks */}
              <div className="bg-neutral-900 border border-neutral-800 rounded-lg p-3.5">
                <div className="flex items-center justify-between text-neutral-400 text-xs">
                  <span>Total Tasks</span>
                  <Layers size={14} />
                </div>
                <div className="text-2xl font-bold font-mono text-neutral-100 mt-1">
                  {analysis.total_tasks}
                </div>
                <div className="flex items-center gap-1.5 text-[11px] font-mono mt-1 text-neutral-400">
                  <span className="text-emerald-400">{analysis.succeeded_tasks} succ</span>
                  <span>•</span>
                  <span className={analysis.failed_tasks > 0 ? "text-red-400" : "text-neutral-500"}>
                    {analysis.failed_tasks} fail
                  </span>
                  <span>•</span>
                  <span className="text-zinc-500">{analysis.cancelled_tasks} canc</span>
                </div>
              </div>

              {/* Wall Time */}
              <div className="bg-neutral-900 border border-neutral-800 rounded-lg p-3.5">
                <div className="flex items-center justify-between text-neutral-400 text-xs">
                  <span>Wall-Clock Time</span>
                  <Clock size={14} />
                </div>
                <div className="text-2xl font-bold font-mono text-neutral-100 mt-1">
                  {formatDuration(analysis.total_wall_seconds)}
                </div>
                <div className="text-[11px] text-neutral-500 mt-1">
                  Total job turnaround span
                </div>
              </div>

              {/* Total Compute Time */}
              <div className="bg-neutral-900 border border-neutral-800 rounded-lg p-3.5">
                <div className="flex items-center justify-between text-neutral-400 text-xs">
                  <span>Total Compute Time</span>
                  <Cpu size={14} />
                </div>
                <div className="text-2xl font-bold font-mono text-neutral-100 mt-1">
                  {formatDuration(analysis.total_compute_seconds)}
                </div>
                <div className="text-[11px] text-neutral-500 mt-1">
                  Cumulative across all tasks
                </div>
              </div>

              {/* Average Attempt Concurrency */}
              <div className="bg-neutral-900 border border-neutral-800 rounded-lg p-3.5">
                <div className="flex items-center justify-between text-neutral-400 text-xs">
                  <span>Average Attempt Concurrency</span>
                  <Zap size={14} className="text-amber-400" />
                </div>
                <div className="text-2xl font-bold font-mono text-amber-400 mt-1">
                  {analysis.total_compute_seconds == null || !analysis.total_wall_seconds
                    ? "Unavailable"
                    : `${(analysis.total_compute_seconds / analysis.total_wall_seconds).toFixed(1)} tasks`}
                </div>
                <div className="text-[11px] text-neutral-500 mt-1">
                  Average active parallel tasks
                </div>
              </div>

              {/* Estimated Compute Cost */}
              <div className="bg-neutral-900 border border-neutral-800 rounded-lg p-3.5">
                <div className="flex items-center justify-between text-neutral-400 text-xs">
                  <span>Estimated Compute Cost</span>
                  <DollarSign size={14} className="text-emerald-400" />
                </div>
                <div className="text-2xl font-bold font-mono text-emerald-400 mt-1">
                  ${formatObserved(analysis.accrued_cost_usd, 4)}
                </div>
                <div className="text-[11px] text-neutral-500 mt-1">
                  {analysis.spot == null ? "Pricing unavailable" : analysis.spot ? "Spot discounted tier" : "Standard on-demand rate"}
                </div>
              </div>
            </div>

            {/* Navigation Tabs Bar */}
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

            {/* Tab 1: Overview & Telemetry */}
            {activeTab === "overview" && (
              <div className="flex flex-col gap-6">
                {/* Cost Accrual Card (CR-10) */}
                <CostAccrualCard analysis={analysis} />

                {/* Prediction Accuracy Card (CR-11) */}
                <PredictionAccuracyCard analysis={analysis} />

                {/* Policy Distribution Mini-Summary */}
                <div className="bg-neutral-900 border border-neutral-800 rounded-lg p-5">
                  <div className="flex items-center justify-between mb-3 pb-2 border-b border-neutral-800">
                    <h3 className="text-sm font-semibold text-neutral-200">
                      Workload Policy Distribution
                    </h3>
                    <span className="text-xs text-neutral-500">
                      {policySummary.length ? `${policySummary.length} distinct policies` : "Policy metadata unavailable"}
                    </span>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                    {policySummary.map((p) => (
                      <div
                        key={p.policy}
                        className="bg-neutral-950/60 border border-neutral-800/80 rounded p-3 flex flex-col justify-between"
                      >
                        <div className="flex items-center justify-between">
                          <span className="text-xs font-mono font-semibold text-indigo-300 uppercase">
                            {p.policy}
                          </span>
                          <span className="text-xs font-mono text-neutral-400">
                            {p.count} tasks
                          </span>
                        </div>
                        <div className="mt-2 pt-2 border-t border-neutral-800/60 flex items-center justify-between text-[11px] text-neutral-500 font-mono">
                          <span>Compute: {formatDuration(p.computeSec)}</span>
                          <span>Cost: ${formatObserved(p.costUsd, 4)}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {/* Tab 2: Execution Waterfall Chart */}
            {activeTab === "waterfall" && (
              <div className="flex flex-col gap-4 bg-neutral-900 border border-neutral-800 rounded-lg p-5">
                <div className="flex items-center justify-between pb-3 border-b border-neutral-800">
                  <div>
                    <h3 className="text-sm font-semibold text-neutral-200">
                      Task Execution Duration Waterfall
                    </h3>
                    <p className="text-xs text-neutral-500">
                      Visual distribution of task execution durations and terminal states
                    </p>
                  </div>
                  <div className="flex items-center gap-3 text-xs">
                    <span className="flex items-center gap-1.5 text-emerald-400">
                      <span className="w-2.5 h-2.5 rounded-sm bg-emerald-500" /> Succeeded
                    </span>
                    <span className="flex items-center gap-1.5 text-red-400">
                      <span className="w-2.5 h-2.5 rounded-sm bg-red-500" /> Failed
                    </span>
                    <span className="flex items-center gap-1.5 text-purple-400">
                      <span className="w-2.5 h-2.5 rounded-sm bg-purple-500" /> Preempted
                    </span>
                    <span className="flex items-center gap-1.5 text-zinc-400">
                      <span className="w-2.5 h-2.5 rounded-sm bg-zinc-500" /> Cancelled
                    </span>
                  </div>
                </div>

                <TaskDurationWaterfall tasks={analysis.tasks} height={380} />
              </div>
            )}

            {/* Tab 3: Tasks Inspection Table */}
            {activeTab === "tasks" && (
              <div className="flex flex-col gap-4 bg-neutral-900 border border-neutral-800 rounded-lg p-5">
                {/* Search & State Filter Controls */}
                <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-neutral-800">
                  <div className="flex items-center gap-2">
                    <div className="relative">
                      <Search className="w-3.5 h-3.5 absolute left-3 top-2.5 text-neutral-500" />
                      <input
                        type="text"
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        placeholder="Search task ID, cell, policy..."
                        className="bg-neutral-950 border border-neutral-700 text-xs rounded pl-8 pr-3 py-1.5 text-neutral-200 placeholder-neutral-500 focus:outline-none focus:border-indigo-500 w-64"
                      />
                    </div>

                    <select
                      value={taskFilter}
                      onChange={(e) => setTaskFilter(e.target.value)}
                      className="bg-neutral-950 border border-neutral-700 text-xs rounded px-2.5 py-1.5 text-neutral-200 focus:outline-none focus:border-indigo-500"
                    >
                      <option value="all">All States ({analysis.tasks.length})</option>
                      <option value="succeeded">Succeeded ({analysis.succeeded_tasks})</option>
                      <option value="failed">Failed ({analysis.failed_tasks})</option>
                      <option value="cancelled">Cancelled ({analysis.cancelled_tasks})</option>
                      <option value="preempted">Preempted ({analysis.preempted_tasks})</option>
                    </select>
                  </div>

                  <span className="text-xs text-neutral-400 font-mono">
                    Showing {filteredTasks.length} of {analysis.tasks.length} tasks
                  </span>
                </div>

                {/* Table */}
                <div className="overflow-x-auto">
                  <table className="w-full text-left text-xs border-collapse">
                    <thead>
                      <tr className="border-b border-neutral-800 text-neutral-400 font-mono">
                        <th
                          onClick={() => setSort("task_id")}
                          className="py-2.5 px-3 cursor-pointer hover:text-neutral-200 select-none"
                        >
                          <div className="flex items-center gap-1">
                            <span>Task ID</span>
                            <ArrowUpDown size={12} />
                          </div>
                        </th>
                        <th
                          onClick={() => setSort("cell_id")}
                          className="py-2.5 px-3 cursor-pointer hover:text-neutral-200 select-none"
                        >
                          <div className="flex items-center gap-1">
                            <span>Cell Spec</span>
                            <ArrowUpDown size={12} />
                          </div>
                        </th>
                        <th
                          onClick={() => setSort("state")}
                          className="py-2.5 px-3 cursor-pointer hover:text-neutral-200 select-none"
                        >
                          <div className="flex items-center gap-1">
                            <span>State</span>
                            <ArrowUpDown size={12} />
                          </div>
                        </th>
                        <th className="py-2.5 px-3 text-center">Attempt</th>
                        <th
                          onClick={() => setSort("duration")}
                          className="py-2.5 px-3 text-right cursor-pointer hover:text-neutral-200 select-none"
                        >
                          <div className="flex items-center justify-end gap-1">
                            <span>Duration</span>
                            <ArrowUpDown size={12} />
                          </div>
                        </th>
                        <th
                          onClick={() => setSort("cost")}
                          className="py-2.5 px-3 text-right cursor-pointer hover:text-neutral-200 select-none"
                        >
                          <div className="flex items-center justify-end gap-1">
                            <span>Cost ($)</span>
                            <ArrowUpDown size={12} />
                          </div>
                        </th>
                        <th className="py-2.5 px-3 text-center">Actions</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-neutral-800/60 font-mono">
                      {filteredTasks.length === 0 ? (
                        <tr>
                          <td colSpan={7} className="text-center py-8 text-neutral-500">
                            No tasks match the active filters
                          </td>
                        </tr>
                      ) : (
                        filteredTasks.map((task) => (
                          <tr key={task.task_id} className="hover:bg-neutral-800/30">
                            <td className="py-2.5 px-3 font-semibold text-neutral-200">
                              {task.task_id}
                            </td>
                            <td className="py-2.5 px-3 text-neutral-400 max-w-xs truncate" title={task.cell_id}>
                              {task.cell_id}
                            </td>
                            <td className="py-2.5 px-3">
                              <span
                                className={`inline-block px-2 py-0.5 rounded text-[11px] font-medium capitalize ${
                                  task.state === "succeeded"
                                    ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20"
                                    : task.state === "failed"
                                    ? "bg-red-500/10 text-red-400 border border-red-500/20"
                                    : task.state === "cancelled"
                                    ? "bg-zinc-800 text-zinc-400 border border-zinc-700"
                                    : "bg-amber-500/10 text-amber-400 border border-amber-500/20"
                                }`}
                              >
                                {task.state}
                              </span>
                            </td>
                            <td className="py-2.5 px-3 text-center text-neutral-400">
                              #{task.attempt}
                            </td>
                            <td className="py-2.5 px-3 text-right text-neutral-200">
                              {formatObserved(task.duration_seconds, 2)}s
                            </td>
                            <td className="py-2.5 px-3 text-right text-emerald-400">
                              ${formatObserved(task.cost_usd, 4)}
                            </td>
                            <td className="py-2.5 px-3 text-center">
                              <button
                                onClick={() => setLogTaskId(task.task_id)}
                                className="inline-flex items-center gap-1 px-2 py-1 bg-neutral-800 hover:bg-neutral-700 text-neutral-300 rounded text-[11px] transition-colors"
                                title="Inspect task execution logs"
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

            {/* Tab 4: Financial & Spot Breakdown */}
            {activeTab === "cost" && (
              <div className="flex flex-col gap-6">
                <CostAccrualCard analysis={analysis} />

                <div className="bg-neutral-900 border border-neutral-800 rounded-lg p-5">
                  <h3 className="text-sm font-semibold text-neutral-200 mb-3 pb-2 border-b border-neutral-800">
                    Detailed Infrastructure Pricing Breakdown
                  </h3>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="bg-neutral-950/60 border border-neutral-800/80 rounded p-4 flex flex-col gap-2.5 text-xs">
                      <div className="flex justify-between items-center text-neutral-400">
                        <span>Compute Engine Model</span>
                        <span className="font-mono text-neutral-200 font-semibold">{analysis.machine_type}</span>
                      </div>
                      <div className="flex justify-between items-center text-neutral-400">
                        <span>Cloud Region</span>
                        <span className="font-mono text-neutral-200">{analysis.region}</span>
                      </div>
                      <div className="flex justify-between items-center text-neutral-400">
                        <span>Provisioning Tier</span>
                        <span className="font-mono text-neutral-200">
                          {analysis.spot == null ? "Unavailable" : analysis.spot ? "Preemptible / Spot" : "Standard On-Demand"}
                        </span>
                      </div>
                      <div className="flex justify-between items-center text-neutral-400">
                        <span>Baseline On-Demand Cost</span>
                        <span className="font-mono text-neutral-200">
                          ${formatObserved(analysis.on_demand_baseline_cost_usd, 4)}
                        </span>
                      </div>
                      <div className="flex justify-between items-center text-neutral-400">
                        <span>Effective Estimated Compute Cost</span>
                        <span className="font-mono text-emerald-400 font-bold">
                          ${formatObserved(analysis.accrued_cost_usd, 4)}
                        </span>
                      </div>
                      <div className="flex justify-between items-center text-neutral-400 pt-2 border-t border-neutral-800">
                        <span>Net Spot Savings</span>
                        <span className="font-mono text-emerald-400 font-bold">
                          -${formatObserved(analysis.spot_savings_usd, 4)} ({analysis.spot_savings_pct}%)
                        </span>
                      </div>
                    </div>

                    <div className="bg-neutral-950/60 border border-neutral-800/80 rounded p-4 flex flex-col justify-between text-xs text-neutral-400">
                      <div>
                        <h4 className="font-semibold text-neutral-300 mb-1">Billing Governance Note (§J.1 / CR-10)</h4>
                        <p className="text-[11px] text-neutral-500 leading-relaxed">
                          Billing depends on the provider, resource and all attempts, including retries.
                          Cost and savings require recorded usage and applicable prices.
                        </p>
                      </div>

                      <div className="mt-4 pt-3 border-t border-neutral-800 flex items-center justify-between text-[11px] font-mono text-neutral-500">
                        <span>Control Plane: Local/Remote</span>
                        <span>Billing: Hourly Prorated</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </>
        )}
      </main>

      {/* Modals */}
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
