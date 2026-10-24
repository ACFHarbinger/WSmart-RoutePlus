import React, { useState } from "react";
import { Copy, Check, Radio, Eye, X } from "lucide-react";
import { useLoadedRunStore } from "../../store/loadedRun";
import { useSimStore } from "../../store/sim";
import { NOC_COLORS } from "../ui/telemetry/tokens";

export const LoadedRunChip: React.FC = () => {
  const { path, label, days, entries, source, toggleSource, clearLoadedRun } =
    useLoadedRunStore();
  const liveEntriesCount = useSimStore((s) => s.entries.length);
  const [copied, setCopied] = useState(false);

  const hasLoaded = Boolean(path);
  const hasLive = liveEntriesCount > 0;

  if (!hasLoaded && !hasLive) {
    return null;
  }

  const isLiveMode = source === "live" || (!hasLoaded && hasLive);
  const badgeColor = isLiveMode ? NOC_COLORS.ok : NOC_COLORS.cyan;
  const displayLabel = isLiveMode
    ? `live_stream (${liveEntriesCount} rows)`
    : `${label ?? "run"}${days ? ` • ${days}d` : entries.length ? ` • ${entries.length}r` : ""}`;

  const handleCopyPath = async () => {
    if (!path || typeof navigator === "undefined" || !navigator.clipboard) return;
    try {
      await navigator.clipboard.writeText(path);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // Ignore clipboard permission errors in headless tests
    }
  };

  return (
    <div
      data-testid="loaded-run-chip"
      className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full border border-noc-border bg-noc-deep text-xs tabular-nums"
    >
      <button
        type="button"
        onClick={toggleSource}
        title="Click to toggle between Loaded Run [Viewing] and [Live Stream]"
        style={{
          color: badgeColor,
          backgroundColor: `${badgeColor}1a`,
          borderColor: `${badgeColor}4d`,
        }}
        className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded border text-[10px] font-semibold uppercase tracking-wider cursor-pointer"
      >
        {isLiveMode ? <Radio size={10} /> : <Eye size={10} />}
        <span>{isLiveMode ? "[Live Stream]" : "[Viewing]"}</span>
      </button>

      <span
        className="font-mono text-noc-textPrimary truncate max-w-[200px]"
        title={path ?? "Active live simulation stream"}
      >
        {displayLabel}
      </span>

      {path && (
        <button
          type="button"
          onClick={() => void handleCopyPath()}
          title="Copy run path"
          aria-label="Copy run path"
          className="text-noc-textMuted hover:text-noc-textPrimary p-0.5"
        >
          {copied ? <Check size={11} /> : <Copy size={11} />}
        </button>
      )}

      {path && (
        <button
          type="button"
          onClick={clearLoadedRun}
          title="Clear loaded run"
          aria-label="Clear loaded run"
          className="text-noc-textMuted hover:text-noc-textPrimary p-0.5"
        >
          <X size={11} />
        </button>
      )}
    </div>
  );
};

export default LoadedRunChip;
