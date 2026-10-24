import { useMemo, useRef, useState } from "react";
import ReactECharts from "echarts-for-react";
import type EChartsReact from "echarts-for-react";
import { ChartExportButtons } from "../../common/ChartExportButtons";
import { StrategyLegend } from "../telemetry/StrategyLegend";
import {
  errorBarBounds,
} from "../../../utils/charts/chartLogScale";
import { isHighlighted } from "../../../utils/charts/chartHighlight";
import type { LogPathMeta, PolicyMeta } from "../../../utils/sim/simMetadata";
import {
  buildPolicyHierarchy,
  buildPortfolioHierarchy,
  childrenAtPath,
  enrichDrillChildren,
  policiesAtPath,
  resolveDrillBarColor,
  type HierarchyColorMode,
  type PortfolioHierarchyRun,
} from "../../../utils/benchmark/policyHierarchy";
import {
  fmt,
  type HierarchyView,
  type PolicyStats,
} from "./SummaryCoreWidgets";

function HierarchyBreadcrumb({
  path,
  onNavigate,
}: {
  path: string[];
  onNavigate: (depth: number) => void;
}) {
  if (path.length === 0) return null;
  return (
    <div className="flex flex-wrap items-center gap-1 text-xs">
      <button
        onClick={() => onNavigate(0)}
        className="text-accent-primary hover:underline font-medium"
      >
        All
      </button>
      {path.map((seg, i) => (
        <span key={`${seg}-${i}`} className="flex items-center gap-1">
          <span className="text-canvas-muted">›</span>
          <button
            onClick={() => onNavigate(i + 1)}
            className="text-accent-secondary hover:underline font-mono"
          >
            {seg}
          </button>
        </span>
      ))}
    </div>
  );
}

export function PolicyHierarchyPanel({
  stats,
  policies,
  policyMeta,
  logMeta,
  portfolioRuns,
  brushed,
  onBrushPolicies,
  showErrorBars = false,
  logScale = false,
}: {
  stats: Record<string, PolicyStats>;
  policies: string[];
  policyMeta: Record<string, PolicyMeta>;
  logMeta: LogPathMeta;
  portfolioRuns?: PortfolioHierarchyRun[];
  brushed?: string[] | null;
  onBrushPolicies: (ps: string[]) => void;
  showErrorBars?: boolean;
  logScale?: boolean;
}) {
  const chartRef = useRef<EChartsReact | null>(null);
  const [view, setView] = useState<HierarchyView>("sunburst");
  const [colorMode, setColorMode] = useState<HierarchyColorMode>("kgkm");
  const [drillPath, setDrillPath] = useState<string[]>([]);

  const tree = useMemo(() => {
    if (portfolioRuns && portfolioRuns.length >= 2) {
      return buildPortfolioHierarchy(portfolioRuns, colorMode);
    }
    return buildPolicyHierarchy(policies, stats, policyMeta, logMeta, colorMode);
  }, [portfolioRuns, colorMode, policies, stats, policyMeta, logMeta]);

  const drillChildren = useMemo(
    () => enrichDrillChildren(childrenAtPath(tree, drillPath), stats, policyMeta),
    [tree, drillPath, stats, policyMeta]
  );

  const showDrillMorph = drillPath.length > 0 && drillChildren.length > 0;

  const hierarchyOption = useMemo(
    () => ({
      backgroundColor: "transparent",
      animationDurationUpdate: 750,
      animationEasingUpdate: "cubicOut" as const,
      tooltip: {
        formatter: (p: { name: string; value: number }) =>
          `${p.name}<br/>Profit: ${fmt(p.value, 1)} €`,
      },
      series: showDrillMorph
        ? [
            {
              id: "hierarchy-drill",
              type: "bar" as const,
              data: drillChildren.map((c) => ({
                value: logScale ? Math.max(c.profit, 0.001) : c.profit,
                name: c.name,
                itemStyle: {
                  color: resolveDrillBarColor(c.name, c.policies, stats, colorMode),
                  opacity: c.policies.some((p) => isHighlighted(p, brushed ?? null))
                    ? 1
                    : 0.25,
                },
              })),
              universalTransition: { enabled: true },
            },
            ...(showErrorBars
              ? [
                  {
                    type: "custom" as const,
                    renderItem: (
                      params: { dataIndex: number },
                      api: {
                        coord: (v: [number, number]) => [number, number];
                        style: (s: object) => object;
                      }
                    ) => {
                      const c = drillChildren[params.dataIndex];
                      const err = Math.max(c.profitStd, c.distSpread);
                      const bounds = errorBarBounds(c.profit, err, "profit", logScale);
                      const y = api.coord([bounds.center, params.dataIndex])[1];
                      const xLeft = api.coord([bounds.low, params.dataIndex])[0];
                      const xRight = api.coord([bounds.high, params.dataIndex])[0];
                      const cap = 4;
                      return {
                        type: "group",
                        children: [
                          {
                            type: "line",
                            shape: { x1: xLeft, y1: y, x2: xRight, y2: y },
                            style: api.style({ stroke: "#9090b0", lineWidth: 1.5 }),
                          },
                          {
                            type: "line",
                            shape: {
                              x1: xLeft,
                              y1: y - cap,
                              x2: xLeft,
                              y2: y + cap,
                            },
                            style: api.style({ stroke: "#9090b0", lineWidth: 1.5 }),
                          },
                          {
                            type: "line",
                            shape: {
                              x1: xRight,
                              y1: y - cap,
                              x2: xRight,
                              y2: y + cap,
                            },
                            style: api.style({ stroke: "#9090b0", lineWidth: 1.5 }),
                          },
                        ],
                      };
                    },
                    data: drillChildren.map((_, i) => i),
                    z: 10,
                  },
                ]
              : []),
          ]
        : [
            view === "sunburst"
              ? {
                  id: "hierarchy-sunburst",
                  type: "sunburst" as const,
                  data: tree,
                  radius: ["12%", "90%"],
                  label: { color: "#c0c0d8", fontSize: 9 },
                  itemStyle: { borderWidth: 1, borderColor: "#1a1a2e" },
                  emphasis: { focus: "ancestor" as const },
                  universalTransition: { enabled: true },
                }
              : {
                  id: "hierarchy-treemap",
                  type: "treemap" as const,
                  data: tree,
                  leafDepth: 2,
                  label: { color: "#c0c0d8", fontSize: 9 },
                  upperLabel: { show: true, height: 22, color: "#9090b0" },
                  itemStyle: { borderWidth: 1, borderColor: "#1a1a2e" },
                  universalTransition: { enabled: true },
                },
          ],
      ...(showDrillMorph
        ? {
            grid: { left: 110, right: 24, top: 12, bottom: 12 },
            xAxis: {
              type: (logScale ? "log" : "value") as "log" | "value",
              logBase: 10,
              name: logScale ? "Profit (€, log)" : "Profit (€)",
              nameTextStyle: { color: "#9090b0", fontSize: 9 },
              axisLabel: { color: "#9090b0", fontSize: 9 },
              minorSplitLine: { show: false },
            },
            yAxis: {
              type: "category" as const,
              data: drillChildren.map((c) => c.name),
              inverse: true,
              axisLabel: { color: "#9090b0", fontSize: 9 },
            },
          }
        : {}),
    }),
    [
      tree,
      view,
      showDrillMorph,
      drillChildren,
      brushed,
      showErrorBars,
      logScale,
      stats,
      colorMode,
    ]
  );

  const handleSegmentClick = (path: string[]) => {
    setDrillPath(path);
    onBrushPolicies(policiesAtPath(tree, path));
  };

  return (
    <div className="card space-y-3">
      <div className="flex items-center justify-between flex-wrap gap-2">
        <div>
          <p className="text-xs font-semibold text-gray-300">
            Policy Hierarchy (§G.2)
          </p>
          <p className="text-[10px] text-canvas-muted">
            Span = profit · color ={" "}
            {colorMode === "kgkm" ? "kg/km efficiency" : "mean overflows"} ·
            strategy ring borders + drill bars use packing palette
          </p>
        </div>
        <div className="flex items-center gap-2">
          <div className="flex items-center gap-1 bg-canvas-elevated rounded-lg p-0.5">
            {(["kgkm", "overflows"] as const).map((m) => (
              <button
                key={m}
                onClick={() => setColorMode(m)}
                className={`text-xs px-2.5 py-1 rounded-md transition-colors ${
                  colorMode === m
                    ? "bg-accent-primary text-white"
                    : "text-canvas-muted hover:text-gray-200"
                }`}
              >
                {m === "kgkm" ? "kg/km" : "Overflows"}
              </button>
            ))}
          </div>
          <div className="flex items-center gap-1 bg-canvas-elevated rounded-lg p-0.5">
            {(["sunburst", "treemap"] as const).map((v) => (
              <button
                key={v}
                onClick={() => setView(v)}
                className={`text-xs px-2.5 py-1 rounded-md capitalize transition-colors ${
                  view === v
                    ? "bg-accent-primary text-white"
                    : "text-canvas-muted hover:text-gray-200"
                }`}
              >
                {v}
              </button>
            ))}
          </div>
          <ChartExportButtons
            chartRef={{ current: chartRef.current }}
            filenameStem={`summary-${view}`}
          />
        </div>
      </div>

      <HierarchyBreadcrumb
        path={drillPath}
        onNavigate={(depth) => {
          const next = drillPath.slice(0, depth);
          setDrillPath(next);
          onBrushPolicies(next.length ? policiesAtPath(tree, next) : policies);
        }}
      />

      <StrategyLegend />

      <ReactECharts
        ref={chartRef}
        option={hierarchyOption}
        notMerge={false}
        style={{
          height: showDrillMorph ? Math.max(180, drillChildren.length * 32) : 300,
        }}
        onEvents={{
          click: (params: {
            treePathInfo?: Array<{ name: string }>;
            name?: string;
          }) => {
            if (params.treePathInfo?.length) {
              const path = params.treePathInfo.map((n) => n.name);
              handleSegmentClick(path);
              return;
            }
            if (showDrillMorph && params.name) {
              const child = drillChildren.find((c) => c.name === params.name);
              if (child) onBrushPolicies(child.policies);
            }
          },
        }}
      />
    </div>
  );
}
