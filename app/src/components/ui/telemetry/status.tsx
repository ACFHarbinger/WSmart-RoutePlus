import { useEffect, useState, type ReactNode } from "react";
import { KpiCard, type KpiCardProps } from "../KpiCard";
import { NOC_COLORS, type NocTone } from "./tokens";

export { KpiCard, type KpiCardProps };

export function KpiCardGrid({
  items,
  columns = 4,
}: {
  items: KpiCardProps[];
  columns?: 2 | 3 | 4 | 6;
}) {
  const colClass =
    columns === 2
      ? "grid-cols-1 sm:grid-cols-2"
      : columns === 3
      ? "grid-cols-1 sm:grid-cols-3"
      : columns === 6
      ? "grid-cols-2 sm:grid-cols-3 lg:grid-cols-6"
      : "grid-cols-2 lg:grid-cols-4";

  return (
    <div className={`grid ${colClass} gap-3`}>
      {items.map((item) => (
        <KpiCard key={item.label} {...item} />
      ))}
    </div>
  );
}

const TONE_HEX: Record<NocTone, string> = {
  ok: NOC_COLORS.ok,
  warn: NOC_COLORS.warn,
  crit: NOC_COLORS.crit,
  idle: NOC_COLORS.idle,
  cyan: NOC_COLORS.cyan,
  indigo: NOC_COLORS.indigoText,
  purple: NOC_COLORS.purpleText,
  amber: NOC_COLORS.amber,
};

export function HudStatusPill({
  label,
  value,
  tone = "cyan",
  onClick,
  title,
  pulse = false,
}: {
  label: string;
  value?: string | number;
  tone?: NocTone;
  onClick?: () => void;
  title?: string;
  pulse?: boolean;
}) {
  const color = TONE_HEX[tone];
  const Tag = onClick ? "button" : "div";

  return (
    <Tag
      type={onClick ? "button" : undefined}
      onClick={onClick}
      title={title}
      style={{
        borderColor: `${color}40`,
        backgroundColor: `${color}14`,
      }}
      className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full border text-xs tabular-nums transition-colors ${
        onClick ? "hover:bg-canvas-hover cursor-pointer" : ""
      }`}
    >
      <span
        style={{ backgroundColor: color }}
        className={`w-2 h-2 rounded-full shrink-0 ${pulse ? "animate-pulse" : ""}`}
      />
      <span className="text-noc-textSecondary font-medium">{label}</span>
      {value !== undefined && (
        <span style={{ color }} className="font-mono font-semibold tabular-nums">
          {value}
        </span>
      )}
    </Tag>
  );
}

/**
 * Self-contained heartbeat indicator reading `lastEventAt` timestamp.
 * Manages its own 1 Hz timer internally so parent views never re-render.
 */
export function StalenessDot({
  lastEventAt,
  staleAfterSec = 10,
}: {
  lastEventAt: number | null | undefined;
  staleAfterSec?: number;
}) {
  const [now, setNow] = useState(() => Date.now());

  useEffect(() => {
    if (!lastEventAt) return;
    const id = setInterval(() => {
      if (typeof document === "undefined" || document.visibilityState !== "hidden") {
        setNow(Date.now());
      }
    }, 1000);
    return () => clearInterval(id);
  }, [lastEventAt]);

  if (!lastEventAt) {
    return (
      <span className="inline-flex items-center gap-1.5 text-xs text-noc-textMuted tabular-nums">
        <span
          style={{ backgroundColor: NOC_COLORS.idle }}
          className="w-2 h-2 rounded-full"
        />
        awaiting stream
      </span>
    );
  }

  const ageSec = Math.max(0, Math.floor((now - lastEventAt) / 1000));
  const isStale = ageSec >= staleAfterSec;
  const color = isStale ? NOC_COLORS.warn : NOC_COLORS.ok;

  return (
    <span className="inline-flex items-center gap-1.5 text-xs text-noc-textSecondary tabular-nums">
      <span
        style={{ backgroundColor: color }}
        className={`w-2 h-2 rounded-full ${isStale ? "" : "animate-pulse"}`}
      />
      <span>
        {isStale ? "stale" : "live"} • {ageSec}s
      </span>
    </span>
  );
}

export interface MonitorTileItem {
  id: string;
  label: string;
  value?: string | number;
  tone: NocTone;
}

export function MonitorTileStrip({
  tiles,
  onSelect,
  selectedId,
}: {
  tiles: MonitorTileItem[];
  onSelect?: (id: string) => void;
  selectedId?: string | null;
}) {
  return (
    <div className="flex flex-wrap items-center gap-1.5" role="list">
      {tiles.map((tile) => {
        const color = TONE_HEX[tile.tone];
        const isSelected = selectedId === tile.id;
        return (
          <button
            key={tile.id}
            type="button"
            onClick={() => onSelect?.(tile.id)}
            aria-pressed={isSelected}
            style={{
              borderColor: isSelected ? color : "var(--noc-border)",
              backgroundColor: isSelected ? `${color}22` : "var(--noc-bg-card)",
            }}
            className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md border text-xs tabular-nums transition-colors hover:border-noc-borderStrong"
          >
            <span
              style={{ backgroundColor: color }}
              className="w-1.5 h-1.5 rounded-full shrink-0"
            />
            <span className="font-medium text-noc-textPrimary">{tile.label}</span>
            {tile.value !== undefined && (
              <span className="font-mono text-noc-textMuted tabular-nums">
                {tile.value}
              </span>
            )}
          </button>
        );
      })}
    </div>
  );
}

export function EmptyStateChip({
  title,
  description,
  tone = "idle",
  actionLabel,
  onAction,
  icon,
}: {
  title: string;
  description?: string;
  tone?: NocTone;
  actionLabel?: string;
  onAction?: () => void;
  icon?: ReactNode;
}) {
  const color = TONE_HEX[tone];
  return (
    <div
      style={{ borderColor: `${color}33` }}
      className="flex items-center justify-between gap-4 p-4 rounded-xl border bg-noc-panel/80 text-sm"
    >
      <div className="flex items-center gap-3 min-w-0">
        {icon && <div style={{ color }}>{icon}</div>}
        <div className="min-w-0">
          <div className="font-medium text-noc-textPrimary">{title}</div>
          {description && (
            <div className="text-xs text-noc-textMuted mt-0.5">{description}</div>
          )}
        </div>
      </div>
      {actionLabel && onAction && (
        <button
          type="button"
          onClick={onAction}
          className="btn-primary text-xs px-3 py-1.5 shrink-0"
        >
          {actionLabel}
        </button>
      )}
    </div>
  );
}
