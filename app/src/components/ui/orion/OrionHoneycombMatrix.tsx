import React, { useId, useMemo, useState } from "react";
import { OrionGlowDefs } from "./OrionGlowDefs";
import { nocPanelStyle } from "../telemetry/panel";
import { NOC_COLORS } from "../telemetry/tokens";

export interface HoneycombCellData {
  id: string;
  row: number;
  col: number;
  label: string;
  xParam: string;
  yParam: string;
  value: number;
  unit?: string;
  secondaryLabel?: string;
}

export interface OrionHoneycombMatrixProps {
  title?: string;
  subtitle?: string;
  xAxisLabel?: string;
  yAxisLabel?: string;
  cells: readonly HoneycombCellData[];
  higherIsBetter?: boolean;
  hexRadius?: number;
  onSelectCell?: (cell: HoneycombCellData) => void;
}

function hexPolygonPoints(cx: number, cy: number, r: number): string {
  const pts: string[] = [];
  for (let i = 0; i < 6; i++) {
    const angle = (Math.PI / 180) * (60 * i - 30);
    const px = cx + r * Math.cos(angle);
    const py = cy + r * Math.sin(angle);
    pts.push(`${px.toFixed(1)},${py.toFixed(1)}`);
  }
  return pts.join(" ");
}

/**
 * Hexagonal Parameter & HPO Sensitivity Matrix (Slice V3, Owner Ruling #3).
 *
 * Strictly reserved for abstract 2D parameter/hyperparameter sweeps (e.g.
 * learning-rate × KL-weight or threshold × fleet-size). Supports full keyboard
 * focus navigation (`role="gridcell"`, `tabIndex={0}`) and zero `feGaussianBlur`.
 */
export const OrionHoneycombMatrix: React.FC<OrionHoneycombMatrixProps> = ({
  title = "2D Parameter Sensitivity Honeycomb",
  subtitle = "Abstract hyperparameter / threshold sweep surface",
  xAxisLabel = "Parameter X",
  yAxisLabel = "Parameter Y",
  cells,
  higherIsBetter = true,
  hexRadius = 26,
  onSelectCell,
}) => {
  const gradPrefix = useId().replace(/:/g, "");
  const [activeId, setActiveId] = useState<string | null>(
    cells[0]?.id ?? null
  );

  const { positioned, svgWidth, svgHeight, bestId } = useMemo(() => {
    if (cells.length === 0) {
      return { positioned: [], svgWidth: 320, svgHeight: 180, bestId: null };
    }
    const vals = cells.map((c) => c.value);
    const minVal = Math.min(...vals);
    const maxVal = Math.max(...vals);
    const span = maxVal - minVal || 1;

    const hexW = Math.sqrt(3) * hexRadius;
    const vertStep = 1.5 * hexRadius;
    const padX = hexRadius + 20;
    const padY = hexRadius + 20;

    let maxX = 0;
    let maxY = 0;
    let optimalCellId: string | null = null;
    let bestScore = -1;

    const items = cells.map((cell) => {
      const offset = cell.row % 2 === 1 ? hexW / 2 : 0;
      const cx = padX + cell.col * (hexW + 4) + offset;
      const cy = padY + cell.row * (vertStep + 4);
      if (cx > maxX) maxX = cx;
      if (cy > maxY) maxY = cy;

      const norm = higherIsBetter
        ? (cell.value - minVal) / span
        : (maxVal - cell.value) / span;
      if (norm > bestScore) {
        bestScore = norm;
        optimalCellId = cell.id;
      }

      const tier: "optimal" | "nominal" | "warn" | "crit" =
        norm >= 0.8
          ? "optimal"
          : norm >= 0.45
          ? "nominal"
          : norm >= 0.2
          ? "warn"
          : "crit";

      return { cell, cx, cy, norm, tier };
    });

    return {
      positioned: items,
      svgWidth: Math.max(320, Math.ceil(maxX + padX)),
      svgHeight: Math.max(180, Math.ceil(maxY + padY)),
      bestId: optimalCellId,
    };
  }, [cells, hexRadius, higherIsBetter]);

  const activeItem = useMemo(
    () => positioned.find((p) => p.cell.id === activeId) ?? positioned[0],
    [activeId, positioned]
  );

  return (
    <div
      className="rounded-xl border p-4 space-y-3"
      style={nocPanelStyle}
      data-testid="orion-honeycomb-matrix"
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <h3 className="text-xs font-bold uppercase tracking-wider text-noc-textPrimary">
            {title}
          </h3>
          <p className="text-[11px] text-noc-textMuted">
            {subtitle} ({xAxisLabel} × {yAxisLabel})
          </p>
        </div>

        {activeItem && (
          <div className="flex items-center gap-3 px-3 py-1 rounded-lg border border-noc-border bg-noc-deep text-xs tabular-nums">
            <span className="font-medium text-noc-textSecondary">
              {activeItem.cell.xParam} × {activeItem.cell.yParam}
            </span>
            <span
              className="font-mono font-bold"
              style={{ color: NOC_COLORS.cyan }}
            >
              {activeItem.cell.value.toLocaleString(undefined, {
                maximumFractionDigits: 2,
              })}
              {activeItem.cell.unit ? ` ${activeItem.cell.unit}` : ""}
            </span>
          </div>
        )}
      </div>

      {positioned.length === 0 ? (
        <div className="flex items-center justify-center h-40 text-xs text-noc-textMuted border border-dashed border-noc-border rounded-lg">
          No parameter sweep cells available
        </div>
      ) : (
        <div className="overflow-x-auto">
          <svg
            viewBox={`0 0 ${svgWidth} ${svgHeight}`}
            width="100%"
            style={{ maxHeight: 340 }}
            role="grid"
            aria-label={title}
          >
            <OrionGlowDefs idPrefix={gradPrefix} />

            {positioned.map(({ cell, cx, cy, tier }) => {
              const isSelected = cell.id === activeItem?.cell.id;
              const isBest = cell.id === bestId;
              const fillUrl = `url(#${gradPrefix}-hex-${tier})`;

              return (
                <g
                  key={cell.id}
                  role="gridcell"
                  tabIndex={0}
                  aria-label={`${cell.label}: ${cell.value}${
                    cell.unit ? ` ${cell.unit}` : ""
                  }`}
                  aria-selected={isSelected}
                  onClick={() => {
                    setActiveId(cell.id);
                    onSelectCell?.(cell);
                  }}
                  onMouseEnter={() => setActiveId(cell.id)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === " ") {
                      e.preventDefault();
                      setActiveId(cell.id);
                      onSelectCell?.(cell);
                    }
                  }}
                  className="cursor-pointer focus:outline-none"
                >
                  {isBest && (
                    <circle
                      cx={cx}
                      cy={cy}
                      r={hexRadius * 1.35}
                      fill={`url(#${gradPrefix}-glow-ok)`}
                    />
                  )}
                  <polygon
                    points={hexPolygonPoints(cx, cy, hexRadius - 1.5)}
                    fill={fillUrl}
                    stroke={
                      isSelected
                        ? NOC_COLORS.cyan
                        : isBest
                        ? NOC_COLORS.emerald
                        : "var(--noc-border-strong)"
                    }
                    strokeWidth={isSelected ? 2.5 : isBest ? 2 : 1}
                  />
                  <text
                    x={cx}
                    y={cy - 3}
                    textAnchor="middle"
                    dominantBaseline="central"
                    fill="#f8fafc"
                    className="text-[10px] font-mono font-bold tabular-nums pointer-events-none"
                  >
                    {cell.value.toLocaleString(undefined, {
                      maximumFractionDigits: 1,
                    })}
                  </text>
                  <text
                    x={cx}
                    y={cy + 9}
                    textAnchor="middle"
                    dominantBaseline="central"
                    fill="rgba(248,250,252,0.82)"
                    className="text-[8px] font-medium pointer-events-none"
                  >
                    {cell.label}
                  </text>
                </g>
              );
            })}
          </svg>
        </div>
      )}
    </div>
  );
};

export default OrionHoneycombMatrix;
