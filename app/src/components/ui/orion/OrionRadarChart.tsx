import React, { useMemo, useState } from "react";
import ReactECharts from "echarts-for-react";
import {
  SIM_METRICS,
  normalizeCohortMetric,
  type SimKpiKey,
} from "../telemetry/metrics";
import { getPolicyColor } from "../../../utils/charts/palettes";
import { nocPanelStyle } from "../telemetry/panel";
import { NOC_COLORS } from "../telemetry/tokens";

export interface RadarPolicyCohortEntry {
  policy: string;
  /** Raw metric means keyed by SimKpiKey. */
  means: Partial<Record<SimKpiKey, number>>;
  /** Optional sample standard deviations keyed by SimKpiKey. */
  stds?: Partial<Record<SimKpiKey, number>>;
}

export interface OrionRadarChartProps {
  title?: string;
  subtitle?: string;
  cohort: readonly RadarPolicyCohortEntry[];
  axes?: readonly SimKpiKey[];
  /** Maximum number of policies initially toggled visible (default 4 per plan). */
  maxInitialVisible?: number;
  height?: number;
}

const DEFAULT_AXES: readonly SimKpiKey[] = [
  "profit",
  "kg",
  "kg/km",
  "km",
  "overflows",
  "cost",
];

/**
 * Direction-Aware Multi-Objective Policy Radar Chart (Slice V3, Owner Ruling #4).
 *
 * Enforces the universal direction contract (`1.0 = best in cohort`, outer ring =
 * best) where lower-is-better metrics (`km`, `overflows`, `cost`, `kg_lost`) are
 * inverted via `normalizeCohortMetric`. Caps default visible polygons at 4 with
 * interactive pill toggles and displays raw means ± std with units in tooltips.
 */
export const OrionRadarChart: React.FC<OrionRadarChartProps> = ({
  title = "Multi-Objective Policy Radar (1.0 = Best in Cohort)",
  subtitle = "Outer boundary strictly represents optimal cohort performance on every axis",
  cohort,
  axes = DEFAULT_AXES,
  maxInitialVisible = 4,
  height = 320,
}) => {
  const [selectedPolicies, setSelectedPolicies] = useState<string[]>(() =>
    cohort.slice(0, maxInitialVisible).map((c) => c.policy)
  );

  // Keep selection valid if cohort prop changes
  const activePolicies = useMemo(() => {
    const valid = selectedPolicies.filter((p) =>
      cohort.some((c) => c.policy === p)
    );
    if (valid.length > 0) return valid;
    return cohort.slice(0, maxInitialVisible).map((c) => c.policy);
  }, [cohort, maxInitialVisible, selectedPolicies]);

  const plottedAxes = useMemo(
    () =>
      axes.filter((axis) =>
        cohort.every(
          (entry) =>
            entry.means[axis] != null && Number.isFinite(entry.means[axis])
        )
      ),
    [axes, cohort]
  );

  const { normalizedByPolicy, indicators } = useMemo(() => {
    const byAxis: Record<SimKpiKey, number[]> = {} as Record<SimKpiKey, number[]>;
    for (const axis of plottedAxes) {
      const contract = SIM_METRICS[axis];
      const rawValues = cohort.map((c) => c.means[axis] as number);
      byAxis[axis] = normalizeCohortMetric(rawValues, contract.higherIsBetter);
    }

    const normMap = new Map<string, number[]>();
    cohort.forEach((entry, idx) => {
      const vec = plottedAxes.map((axis) =>
        Number((byAxis[axis]?.[idx] ?? 0).toFixed(3))
      );
      normMap.set(entry.policy, vec);
    });

    const inds = plottedAxes.map((axis) => {
      const c = SIM_METRICS[axis];
      return {
        name: `${c.shortLabel} (${c.directionArrow} best)`,
        max: 1.0,
        min: 0.0,
      };
    });

    return { normalizedByPolicy: normMap, indicators: inds };
  }, [cohort, plottedAxes]);

  const option = useMemo(() => {
    const visibleEntries = cohort.filter((c) =>
      activePolicies.includes(c.policy)
    );

    return {
      animation: false,
      tooltip: {
        trigger: "item",
        formatter: (params: { name?: string }) => {
          const policyName = params.name ?? "";
          const entry = cohort.find((c) => c.policy === policyName);
          if (!entry) return policyName;
          const lines = plottedAxes.map((axis, i) => {
            const c = SIM_METRICS[axis];
            const raw = entry.means[axis];
            const std = entry.stds?.[axis];
            const norm = normalizedByPolicy.get(policyName)?.[i] ?? 0;
            const rawStr = c.format(raw);
            const stdStr =
              std !== undefined && Number.isFinite(std)
                ? ` ± ${c.format(std)}`
                : "";
            return `${c.shortLabel}: <strong>${rawStr}${stdStr} ${c.unit}</strong> (score ${norm.toFixed(2)})`;
          });
          return `<div style="font-family:monospace;font-size:11px">
            <strong>${policyName.toUpperCase()}</strong><br/>
            ${lines.join("<br/>")}
          </div>`;
        },
      },
      radar: {
        indicator: indicators,
        radius: "66%",
        splitNumber: 4,
        axisName: {
          color: "var(--noc-text-secondary)",
          fontSize: 10,
        },
        splitArea: {
          areaStyle: {
            color: ["rgba(15,23,42,0.15)", "rgba(30,41,59,0.25)"],
          },
        },
        splitLine: {
          lineStyle: { color: "rgba(148,163,184,0.22)" },
        },
        axisLine: {
          lineStyle: { color: "rgba(148,163,184,0.22)" },
        },
      },
      series: [
        {
          type: "radar",
          data: visibleEntries.map((entry, idx) => {
            const color = getPolicyColor(entry.policy, idx);
            return {
              name: entry.policy,
              value: normalizedByPolicy.get(entry.policy) ?? [],
              itemStyle: { color },
              lineStyle: { width: 2, color },
              areaStyle: { color, opacity: 0.16 },
            };
          }),
        },
      ],
    };
  }, [activePolicies, cohort, indicators, normalizedByPolicy, plottedAxes]);

  const togglePolicy = (policy: string) => {
    setSelectedPolicies((prev) =>
      prev.includes(policy)
        ? prev.filter((p) => p !== policy)
        : [...prev, policy]
    );
  };

  return (
    <div
      className="rounded-xl border p-4 space-y-3"
      style={nocPanelStyle}
      data-testid="orion-radar-chart"
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <h3 className="text-xs font-bold uppercase tracking-wider text-noc-textPrimary">
            {title}
          </h3>
          <p className="text-[11px] text-noc-textMuted">{subtitle}</p>
        </div>
        <span className="text-[11px] font-mono text-noc-textMuted tabular-nums">
          Showing {activePolicies.length} / {cohort.length} policies
        </span>
      </div>

      {cohort.length > 0 && (
        <div
          className="flex flex-wrap items-center gap-1.5"
          role="group"
          aria-label="Policy radar visibility toggles"
        >
          {cohort.map((entry, idx) => {
            const active = activePolicies.includes(entry.policy);
            const color = getPolicyColor(entry.policy, idx);
            return (
              <button
                key={entry.policy}
                type="button"
                aria-pressed={active}
                onClick={() => togglePolicy(entry.policy)}
                style={{
                  borderColor: active ? color : "var(--noc-border)",
                  backgroundColor: active ? `${color}22` : "var(--noc-bg-deep)",
                }}
                className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full border text-xs font-mono uppercase transition-colors"
              >
                <span
                  className="w-2 h-2 rounded-full"
                  style={{
                    backgroundColor: active ? color : NOC_COLORS.idle,
                  }}
                />
                <span className="text-noc-textPrimary">{entry.policy}</span>
              </button>
            );
          })}
        </div>
      )}

      {cohort.length === 0 ? (
        <div
          style={{ height: 200 }}
          className="flex items-center justify-center text-xs text-noc-textMuted border border-dashed border-noc-border rounded-lg"
        >
          No policy cohort data available
        </div>
      ) : (
        <ReactECharts
          option={option}
          style={{ height, width: "100%" }}
          notMerge
          lazyUpdate
        />
      )}
    </div>
  );
};

export default OrionRadarChart;
