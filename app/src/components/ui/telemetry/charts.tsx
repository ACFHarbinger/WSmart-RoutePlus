import { useId, useMemo, useState } from "react";
import { ProvenanceBadge, type ProvenanceTier } from "./gauges";
import { NOC_COLORS } from "./tokens";

export interface TimeSeriesPoint {
  x: number | string;
  y: number;
}

export interface TelemetryTimeSeriesCardProps {
  title: string;
  unit?: string;
  points: readonly TimeSeriesPoint[];
  color?: string;
  height?: number;
  provenance?: ProvenanceTier;
  showMeanLine?: boolean;
}

export function TelemetryTimeSeriesCard({
  title,
  unit = "",
  points,
  color = NOC_COLORS.cyan,
  height = 128,
  provenance,
  showMeanLine = true,
}: TelemetryTimeSeriesCardProps) {
  const gradId = useId();
  const [hoverIdx, setHoverIdx] = useState<number | null>(null);

  const width = 440;
  const padX = 12;
  const padY = 14;
  const plotW = width - padX * 2;
  const plotH = height - padY * 2;

  const { coords, meanY, meanVal, minVal, maxVal } = useMemo(() => {
    if (points.length === 0) {
      return { coords: [], meanY: null, meanVal: 0, minVal: 0, maxVal: 0 };
    }
    const ys = points.map((p) => p.y);
    const min = Math.min(...ys);
    const max = Math.max(...ys);
    const span = max - min || 1;
    const mean = ys.reduce((acc, v) => acc + v, 0) / ys.length;
    const mapY = (val: number) => padY + plotH - ((val - min) / span) * plotH;
    const mapX = (idx: number) =>
      points.length === 1 ? width / 2 : padX + (idx / (points.length - 1)) * plotW;

    return {
      coords: points.map((p, idx) => ({ px: mapX(idx), py: mapY(p.y), point: p })),
      meanY: mapY(mean),
      meanVal: mean,
      minVal: min,
      maxVal: max,
    };
  }, [points, plotH, plotW]);

  const activePoint =
    hoverIdx !== null && coords[hoverIdx] ? coords[hoverIdx]! : coords[coords.length - 1];

  const polyline = coords.map((c) => `${c.px.toFixed(1)},${c.py.toFixed(1)}`).join(" ");
  const areaPath =
    coords.length > 1
      ? `M ${coords[0]!.px.toFixed(1)} ${padY + plotH} L ${coords
          .map((c) => `${c.px.toFixed(1)} ${c.py.toFixed(1)}`)
          .join(" L ")} L ${coords[coords.length - 1]!.px.toFixed(1)} ${padY + plotH} Z`
      : "";

  return (
    <div className="card flex flex-col gap-2">
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2 min-w-0">
          <span className="kpi-label truncate">{title}</span>
          {provenance && <ProvenanceBadge tier={provenance} />}
        </div>
        {activePoint && (
          <span className="text-xs font-mono font-semibold text-noc-textPrimary tabular-nums">
            {activePoint.point.y.toLocaleString(undefined, { maximumFractionDigits: 2 })}
            {unit ? ` ${unit}` : ""}
          </span>
        )}
      </div>

      {coords.length === 0 ? (
        <div
          style={{ height }}
          className="flex items-center justify-center text-xs text-noc-textMuted"
        >
          No telemetry points
        </div>
      ) : (
        <svg
          viewBox={`0 0 ${width} ${height}`}
          className="w-full overflow-visible"
          style={{ height }}
          role="img"
          aria-label={`${title} time series`}
          onMouseLeave={() => setHoverIdx(null)}
        >
          <defs>
            <linearGradient id={gradId} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={color} stopOpacity={0.32} />
              <stop offset="100%" stopColor={color} stopOpacity={0.0} />
            </linearGradient>
          </defs>

          {areaPath && <path d={areaPath} fill={`url(#${gradId})`} />}

          {showMeanLine && meanY !== null && (
            <line
              x1={padX}
              y1={meanY}
              x2={width - padX}
              y2={meanY}
              stroke="var(--noc-border-strong)"
              strokeDasharray="4 4"
              strokeWidth={1}
            />
          )}

          {coords.length > 1 && (
            <polyline
              fill="none"
              stroke={color}
              strokeWidth={2}
              strokeLinejoin="round"
              strokeLinecap="round"
              points={polyline}
            />
          )}

          {hoverIdx !== null && coords[hoverIdx] && (
            <g>
              <line
                x1={coords[hoverIdx]!.px}
                y1={padY}
                x2={coords[hoverIdx]!.px}
                y2={height - padY}
                stroke={color}
                strokeDasharray="2 2"
                strokeWidth={1}
              />
              <circle
                cx={coords[hoverIdx]!.px}
                cy={coords[hoverIdx]!.py}
                r={4}
                fill={color}
                stroke="var(--noc-bg-panel)"
                strokeWidth={1.5}
              />
            </g>
          )}

          {coords.map((c, idx) => (
            <rect
              key={idx}
              x={c.px - plotW / Math.max(2, coords.length) / 2}
              y={0}
              width={plotW / Math.max(1, coords.length)}
              height={height}
              fill="transparent"
              onMouseEnter={() => setHoverIdx(idx)}
            />
          ))}
        </svg>
      )}

      {coords.length > 0 && (
        <div className="flex items-center justify-between text-[11px] text-noc-textMuted tabular-nums">
          <span>
            Min: {minVal.toLocaleString(undefined, { maximumFractionDigits: 1 })}
            {unit ? ` ${unit}` : ""}
          </span>
          <span>
            Mean: {meanVal.toLocaleString(undefined, { maximumFractionDigits: 1 })}
            {unit ? ` ${unit}` : ""}
          </span>
          <span>
            Max: {maxVal.toLocaleString(undefined, { maximumFractionDigits: 1 })}
            {unit ? ` ${unit}` : ""}
          </span>
        </div>
      )}
    </div>
  );
}
