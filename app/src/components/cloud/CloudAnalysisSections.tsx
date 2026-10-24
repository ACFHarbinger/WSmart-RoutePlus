import {
  ArrowUpDown,
  Clock,
  Cpu,
  DollarSign,
  Layers,
  Search,
  Terminal,
  Zap,
} from "lucide-react";
import { formatObserved } from "../../utils/cloud/analysis";
import type { CloudJobAnalysis, CloudTaskAnalysis } from "../../types/cloud";
import type { AnalysisTab } from "../../store/cloudAnalysis";
import { CostAccrualCard } from "./CostAccrualCard";
import { PredictionAccuracyCard } from "./PredictionAccuracyCard";
import { TaskDurationWaterfall } from "./TaskDurationWaterfall";

export function formatDuration(seconds: number | null): string {
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

export function CloudAnalysisKpiStrip({
  analysis,
}: {
  analysis: CloudJobAnalysis;
}) {
  return (
    <div className="grid grid-cols-2 md:grid-cols-5 gap-3.5">
      <div className="bg-neutral-900 border border-neutral-800 rounded-lg p-3.5">
        <div className="flex items-center justify-between text-neutral-400 text-xs">
          <span>Total Tasks</span>
          <Layers size={14} />
        </div>
        <div className="text-2xl font-bold font-mono text-neutral-100 mt-1">
          {analysis.total_tasks}
        </div>
        <div className="flex items-center gap-1.5 text-[11px] font-mono mt-1 text-neutral-400">
          <span className="text-emerald-400">
            {analysis.succeeded_tasks} succ
          </span>
          <span>•</span>
          <span
            className={
              analysis.failed_tasks > 0 ? "text-red-400" : "text-neutral-500"
            }
          >
            {analysis.failed_tasks} fail
          </span>
          <span>•</span>
          <span className="text-zinc-500">{analysis.cancelled_tasks} canc</span>
        </div>
      </div>

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

      <div className="bg-neutral-900 border border-neutral-800 rounded-lg p-3.5">
        <div className="flex items-center justify-between text-neutral-400 text-xs">
          <span>Average Attempt Concurrency</span>
          <Zap size={14} className="text-amber-400" />
        </div>
        <div className="text-2xl font-bold font-mono text-amber-400 mt-1">
          {analysis.total_compute_seconds == null ||
          !analysis.total_wall_seconds
            ? "Unavailable"
            : `${(
                analysis.total_compute_seconds / analysis.total_wall_seconds
              ).toFixed(1)} tasks`}
        </div>
        <div className="text-[11px] text-neutral-500 mt-1">
          Average active parallel tasks
        </div>
      </div>

      <div className="bg-neutral-900 border border-neutral-800 rounded-lg p-3.5">
        <div className="flex items-center justify-between text-neutral-400 text-xs">
          <span>Estimated Compute Cost</span>
          <DollarSign size={14} className="text-emerald-400" />
        </div>
        <div className="text-2xl font-bold font-mono text-emerald-400 mt-1">
          ${formatObserved(analysis.accrued_cost_usd, 4)}
        </div>
        <div className="text-[11px] text-neutral-500 mt-1">
          {analysis.spot == null
            ? "Pricing unavailable"
            : analysis.spot
              ? "Spot discounted tier"
              : "Standard on-demand rate"}
        </div>
      </div>
    </div>
  );
}

export function CloudAnalysisTabPanels({
  activeTab,
  analysis,
  policySummary,
  searchQuery,
  setSearchQuery,
  taskFilter,
  setTaskFilter,
  filteredTasks,
  setSort,
  setLogTaskId,
}: {
  activeTab: AnalysisTab;
  analysis: CloudJobAnalysis;
  policySummary: Array<{
    policy: string;
    count: number;
    computeSec: number | null;
    costUsd: number | null;
  }>;
  searchQuery: string;
  setSearchQuery: (q: string) => void;
  taskFilter: string;
  setTaskFilter: (f: string) => void;
  filteredTasks: CloudTaskAnalysis[];
  setSort: (col: "task_id" | "cell_id" | "state" | "duration" | "cost") => void;
  setLogTaskId: (id: string) => void;
}) {
  return (
    <>
      {activeTab === "overview" && (
        <div className="flex flex-col gap-6">
          <CostAccrualCard analysis={analysis} />
          <PredictionAccuracyCard analysis={analysis} />

          <div className="bg-neutral-900 border border-neutral-800 rounded-lg p-5">
            <div className="flex items-center justify-between mb-3 pb-2 border-b border-neutral-800">
              <h3 className="text-sm font-semibold text-neutral-200">
                Workload Policy Distribution
              </h3>
              <span className="text-xs text-neutral-500">
                {policySummary.length
                  ? `${policySummary.length} distinct policies`
                  : "Policy metadata unavailable"}
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

      {activeTab === "waterfall" && (
        <div className="flex flex-col gap-4 bg-neutral-900 border border-neutral-800 rounded-lg p-5">
          <div className="flex items-center justify-between pb-3 border-b border-neutral-800">
            <div>
              <h3 className="text-sm font-semibold text-neutral-200">
                Task Execution Duration Waterfall
              </h3>
              <p className="text-xs text-neutral-500">
                Visual distribution of task execution durations and terminal
                states
              </p>
            </div>
            <div className="flex items-center gap-3 text-xs">
              <span className="flex items-center gap-1.5 text-emerald-400">
                <span className="w-2.5 h-2.5 rounded-sm bg-emerald-500" />{" "}
                Succeeded
              </span>
              <span className="flex items-center gap-1.5 text-red-400">
                <span className="w-2.5 h-2.5 rounded-sm bg-red-500" /> Failed
              </span>
              <span className="flex items-center gap-1.5 text-purple-400">
                <span className="w-2.5 h-2.5 rounded-sm bg-purple-500" />{" "}
                Preempted
              </span>
              <span className="flex items-center gap-1.5 text-zinc-400">
                <span className="w-2.5 h-2.5 rounded-sm bg-zinc-500" />{" "}
                Cancelled
              </span>
            </div>
          </div>

          <TaskDurationWaterfall tasks={analysis.tasks} height={380} />
        </div>
      )}

      {activeTab === "tasks" && (
        <div className="flex flex-col gap-4 bg-neutral-900 border border-neutral-800 rounded-lg p-5">
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
                <option value="all">
                  All States ({analysis.tasks.length})
                </option>
                <option value="succeeded">
                  Succeeded ({analysis.succeeded_tasks})
                </option>
                <option value="failed">Failed ({analysis.failed_tasks})</option>
                <option value="cancelled">
                  Cancelled ({analysis.cancelled_tasks})
                </option>
                <option value="preempted">
                  Preempted ({analysis.preempted_tasks})
                </option>
              </select>
            </div>

            <span className="text-xs text-neutral-400 font-mono">
              Showing {filteredTasks.length} of {analysis.tasks.length} tasks
            </span>
          </div>

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
                    <td
                      colSpan={7}
                      className="text-center py-8 text-neutral-500"
                    >
                      No tasks match the active filters
                    </td>
                  </tr>
                ) : (
                  filteredTasks.map((task) => (
                    <tr
                      key={task.task_id}
                      className="hover:bg-neutral-800/30"
                    >
                      <td className="py-2.5 px-3 font-semibold text-neutral-200">
                        {task.task_id}
                      </td>
                      <td
                        className="py-2.5 px-3 text-neutral-400 max-w-xs truncate"
                        title={task.cell_id}
                      >
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
                  <span className="font-mono text-neutral-200 font-semibold">
                    {analysis.machine_type}
                  </span>
                </div>
                <div className="flex justify-between items-center text-neutral-400">
                  <span>Cloud Region</span>
                  <span className="font-mono text-neutral-200">
                    {analysis.region}
                  </span>
                </div>
                <div className="flex justify-between items-center text-neutral-400">
                  <span>Provisioning Tier</span>
                  <span className="font-mono text-neutral-200">
                    {analysis.spot == null
                      ? "Unavailable"
                      : analysis.spot
                        ? "Preemptible / Spot"
                        : "Standard On-Demand"}
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
                    -${formatObserved(analysis.spot_savings_usd, 4)} (
                    {analysis.spot_savings_pct}%)
                  </span>
                </div>
              </div>

              <div className="bg-neutral-950/60 border border-neutral-800/80 rounded p-4 flex flex-col justify-between text-xs text-neutral-400">
                <div>
                  <h4 className="font-semibold text-neutral-300 mb-1">
                    Billing Governance Note (§J.1 / CR-10)
                  </h4>
                  <p className="text-[11px] text-neutral-500 leading-relaxed">
                    Billing depends on the provider, resource and all attempts,
                    including retries. Cost and savings require recorded usage
                    and applicable prices.
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
  );
}
