import type { KeyboardEvent, ReactNode } from "react";
import { NOC_COLORS } from "./tokens";

export interface SubtabItem<T extends string = string> {
  id: T;
  label: string;
  count?: number;
  icon?: ReactNode;
}

export function SubtabBar<T extends string = string>({
  tabs,
  activeId,
  onChange,
  ariaLabel = "View modes",
}: {
  tabs: readonly SubtabItem<T>[];
  activeId: T;
  onChange: (id: T) => void;
  ariaLabel?: string;
}) {
  const onKeyDown = (e: KeyboardEvent<HTMLButtonElement>, idx: number) => {
    if (tabs.length === 0) return;
    if (e.key === "ArrowRight") {
      e.preventDefault();
      const next = tabs[(idx + 1) % tabs.length]!;
      onChange(next.id);
    } else if (e.key === "ArrowLeft") {
      e.preventDefault();
      const prev = tabs[(idx - 1 + tabs.length) % tabs.length]!;
      onChange(prev.id);
    }
  };

  return (
    <div
      role="tablist"
      aria-label={ariaLabel}
      className="inline-flex flex-wrap items-center gap-1 p-1 rounded-xl border border-noc-border bg-noc-deep"
    >
      {tabs.map((tab, idx) => {
        const isActive = tab.id === activeId;
        return (
          <button
            key={tab.id}
            role="tab"
            type="button"
            aria-selected={isActive}
            tabIndex={isActive ? 0 : -1}
            onClick={() => onChange(tab.id)}
            onKeyDown={(e) => onKeyDown(e, idx)}
            style={
              isActive
                ? {
                    backgroundColor: "var(--noc-bg-card)",
                    borderColor: `${NOC_COLORS.cyan}66`,
                    color: "var(--noc-text-primary)",
                  }
                : undefined
            }
            className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium border transition-colors ${
              isActive
                ? "shadow-sm"
                : "border-transparent text-noc-textMuted hover:text-noc-textPrimary hover:bg-canvas-hover"
            }`}
          >
            {tab.icon}
            <span>{tab.label}</span>
            {tab.count !== undefined && (
              <span
                style={
                  isActive
                    ? {
                        backgroundColor: `${NOC_COLORS.cyan}26`,
                        color: NOC_COLORS.cyan,
                      }
                    : undefined
                }
                className="px-1.5 py-0.2 rounded-full text-[10px] font-mono font-semibold tabular-nums bg-noc-panel text-noc-textMuted"
              >
                {tab.count}
              </span>
            )}
          </button>
        );
      })}
    </div>
  );
}
