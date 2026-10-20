import { formatObserved } from "../../utils/cloud/analysis";
/**
 * Task Duration Waterfall / Gantt chart — visualizes execution duration and status
 * distribution across tasks in a cloud job (§J.1 / CR-16).
 */

import { useMemo } from "react";
import ReactECharts from "echarts-for-react";
import { BarChart3 } from "lucide-react";
import type { CloudTaskAnalysis } from "../../types/cloud";

interface TaskDurationWaterfallProps {
  tasks: CloudTaskAnalysis[];
  height?: number;
}

const STATE_COLORS: Record<string, string> = {
  succeeded: "#10b981", // emerald-500
  running: "#f59e0b",   // amber-500
  failed: "#ef4444",    // red-500
  preempted: "#a855f7", // purple-500
  cancelled: "#71717a", // zinc-500
  queued: "#3b82f6",    // blue-500
};

export function TaskDurationWaterfall({ tasks, height = 340 }: TaskDurationWaterfallProps) {
  const option = useMemo(() => {
    if (!tasks.some((task) => task.duration_seconds != null)) return null;

    // Show top 30 tasks sorted by duration descending or execution order
    const displayTasks = tasks.filter((task) => task.duration_seconds != null).slice(0, 30);
    const taskLabels = displayTasks.map((t) => t.task_id.replace("task-", ""));
    const durations = displayTasks.map((t) => ({
      value: t.duration_seconds,
      itemStyle: {
        color: STATE_COLORS[t.state.toLowerCase()] || "#64748b",
        borderRadius: [0, 4, 4, 0],
      },
      task: t,
    }));

    return {
      backgroundColor: "transparent",
      tooltip: {
        renderMode: "richText",
        trigger: "item",
        backgroundColor: "#18181b",
        borderColor: "#27272a",
        textStyle: { color: "#e4e4e7", fontSize: 12 },
        formatter: (params: any) => {
          const t: CloudTaskAnalysis = params.data.task;
          return [
            t.task_id, `Cell: ${t.cell_id}`, `State: ${t.state}`,
            `Duration: ${formatObserved(t.duration_seconds, 2)}s`,
            `Attempt: #${t.attempt}`, `Cost: ${formatObserved(t.cost_usd, 4)}`,
          ].join("\n");
        },
      },
      grid: {
        top: 20,
        right: 30,
        bottom: 25,
        left: 80,
        containLabel: true,
      },
      xAxis: {
        type: "value",
        name: "Duration (s)",
        nameLocation: "middle",
        nameGap: 24,
        nameTextStyle: { color: "#71717a", fontSize: 11 },
        splitLine: { lineStyle: { color: "#27272a", type: "dashed" } },
        axisLabel: { color: "#a1a1aa", fontSize: 10, formatter: "{value}s" },
      },
      yAxis: {
        type: "category",
        data: taskLabels,
        inverse: true,
        axisLine: { lineStyle: { color: "#3f3f46" } },
        axisLabel: { color: "#a1a1aa", fontSize: 10, fontFamily: "monospace" },
      },
      series: [
        {
          name: "Task Duration",
          type: "bar",
          data: durations,
          barMaxWidth: 18,
          label: {
            show: true,
            position: "right",
            color: "#a1a1aa",
            fontSize: 10,
            fontFamily: "monospace",
            formatter: (p: any) => `${p.value.toFixed(1)}s`,
          },
        },
      ],
    };
  }, [tasks]);

  if (!option) {
    return (
      <div className="bg-neutral-900 border border-neutral-800 rounded-lg p-8 flex flex-col items-center justify-center text-center">
        <BarChart3 className="text-neutral-600 mb-2" size={32} />
        <h4 className="text-sm font-medium text-neutral-300">No task durations available</h4>
        <p className="text-xs text-neutral-500 mt-1">
          Duration waterfall will populate as task attempts finish execution.
        </p>
      </div>
    );
  }

  return (
    <div className="bg-neutral-900 border border-neutral-800 rounded-lg p-4">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <BarChart3 size={16} className="text-blue-400" />
          <h3 className="text-sm font-semibold text-neutral-200">
            Task Execution Duration Waterfall
          </h3>
          <span className="text-xs text-neutral-500">
            ({tasks.length} total tasks)
          </span>
        </div>

        {/* Legend */}
        <div className="flex items-center gap-3 text-xs">
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-sm bg-emerald-500" />
            <span className="text-neutral-400">Succeeded</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-sm bg-amber-500" />
            <span className="text-neutral-400">Running</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-sm bg-purple-500" />
            <span className="text-neutral-400">Preempted</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-sm bg-red-500" />
            <span className="text-neutral-400">Failed</span>
          </div>
        </div>
      </div>

      <div style={{ height }}>
        {option && <ReactECharts option={option} style={{ height: "100%", width: "100%" }} />}
      </div>
    </div>
  );
}
