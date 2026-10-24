import { useCallback, useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import {
  ArrowDown,
  ChevronDown,
  ChevronUp,
  Square,
  Terminal,
  Trash2,
} from "lucide-react";
import { OpenPathToolbar } from "../../common/OpenPathToolbar";
import { useProcessStore } from "../../../store/process";
import { StatusPill } from "../../ui/StatusPill";
import { LiveTrainProgressBar } from "../live/LiveTrainProgressBar";

/**
 * Try to parse a log line as structured JSON (e.g. Python's structlog or loguru JSON sink).
 * Renders level-coloured output when successful; falls back to plain text.
 */
export function LogLine({ line }: { line: string }) {
  const isStderr = line.startsWith("[stderr]");
  const text = isStderr ? line.slice(8) : line;

  try {
    const json = JSON.parse(text) as Record<string, unknown>;
    const level = String(
      json.level ?? json.levelname ?? json.severity ?? ""
    ).toUpperCase();
    const msg = String(json.msg ?? json.message ?? json.text ?? text);
    const ts = String(json.timestamp ?? json.time ?? json.ts ?? json.t ?? "");

    const levelColor =
      level.startsWith("ERR") ||
      level.startsWith("CRIT") ||
      level.startsWith("FATAL")
        ? "text-accent-danger"
        : level.startsWith("WARN")
        ? "text-accent-warning"
        : level.startsWith("DEBUG")
        ? "text-canvas-muted"
        : "text-gray-400";

    return (
      <span>
        {ts && (
          <span className="text-canvas-muted mr-2 text-[10px]">
            {ts.slice(0, 19)}
          </span>
        )}
        {level && (
          <span className={`font-semibold mr-2 ${levelColor}`}>
            [{level.slice(0, 5)}]
          </span>
        )}
        <span className="text-gray-300">{msg}</span>
      </span>
    );
  } catch {
    return (
      <span className={isStderr ? "text-accent-warning" : "text-gray-400"}>
        {line}
      </span>
    );
  }
}

function useLiveDuration(startTime: number, stopped: boolean): string {
  const [elapsed, setElapsed] = useState(Date.now() - startTime);

  useEffect(() => {
    if (stopped) return;
    const id = setInterval(() => setElapsed(Date.now() - startTime), 1000);
    return () => clearInterval(id);
  }, [startTime, stopped]);

  const s = Math.floor(elapsed / 1000);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ${s % 60}s`;
  return `${Math.floor(m / 60)}h ${m % 60}m`;
}

export function ProcessRow({
  id,
  selected,
  runBrushActive,
  logPath,
  brushLabel,
  projectRoot,
  onSelect,
}: {
  id: string;
  selected: boolean;
  runBrushActive?: boolean;
  logPath?: string | null;
  brushLabel?: string | null;
  projectRoot: string | null;
  onSelect: () => void;
}) {
  const proc = useProcessStore((s) => s.processes[id]);
  const removeProcess = useProcessStore((s) => s.removeProcess);
  const [expanded, setExpanded] = useState(false);
  const [autoScroll, setAutoScroll] = useState(true);
  const logRef = useRef<HTMLDivElement>(null);

  const stopped = proc.status !== "running";
  const duration = useLiveDuration(proc.startTime, stopped);

  const cancel = useCallback(async () => {
    await invoke("cancel_process", { id });
  }, [id]);

  useEffect(() => {
    if (expanded && autoScroll && logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [proc.logLines.length, expanded, autoScroll]);

  if (!proc) return null;

  return (
    <div
      className={`border rounded-xl overflow-hidden ${
        selected ? "border-accent-secondary/60" : "border-canvas-border"
      } ${runBrushActive ? "ring-1 ring-accent-secondary/40" : ""}`}
    >
      <div
        className={`flex items-center gap-3 px-4 py-2.5 hover:bg-canvas-hover cursor-pointer ${
          selected ? "bg-accent-primary/10" : "bg-canvas-surface"
        }`}
        onClick={onSelect}
      >
        <StatusPill status={proc.status} />

        <div className="flex-1 min-w-0 space-y-0.5">
          {logPath ? (
            <OpenPathToolbar
              path={logPath}
              projectRoot={projectRoot}
              brushLabel={brushLabel?.trim() ? brushLabel : undefined}
              chipClassName="max-w-full flex-1 min-w-0"
              className="max-w-full min-w-0"
            >
              <span className="text-[10px] text-canvas-muted font-mono shrink-0 truncate max-w-[8rem]">
                {id}
              </span>
            </OpenPathToolbar>
          ) : (
            <p className="text-xs font-mono text-gray-200 truncate">{id}</p>
          )}
          <p className="text-xs text-canvas-muted truncate">{proc.command}</p>
        </div>

        <div className="hidden sm:flex items-center gap-4 text-xs text-canvas-muted shrink-0 tabular-nums">
          <span title="PID">PID {proc.pid}</span>
          <span title="Elapsed">{duration}</span>
          {proc.exitCode !== undefined && (
            <span
              className={
                proc.exitCode === 0
                  ? "text-accent-success"
                  : "text-accent-danger"
              }
            >
              exit {proc.exitCode}
            </span>
          )}
        </div>

        <div className="flex items-center gap-2 shrink-0">
          {proc.status === "running" && (
            <button
              onClick={(e) => {
                e.stopPropagation();
                void cancel();
              }}
              className="flex items-center gap-1 text-xs text-accent-danger hover:text-accent-danger/80 btn-ghost py-1 px-2"
            >
              <Square size={11} />
              Cancel
            </button>
          )}
          {stopped && (
            <button
              onClick={(e) => {
                e.stopPropagation();
                removeProcess(id);
              }}
              className="btn-ghost p-1.5 text-canvas-muted hover:text-accent-danger"
              title="Remove from history"
            >
              <Trash2 size={12} />
            </button>
          )}
          <button
            className="btn-ghost p-1.5 text-canvas-muted"
            onClick={(e) => {
              e.stopPropagation();
              setExpanded((v) => !v);
            }}
            title={expanded ? "Collapse log" : "Expand log"}
          >
            {expanded ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
          </button>
        </div>
      </div>

      {proc.status === "running" && (
        <LiveTrainProgressBar
          processId={id}
          className="px-4 py-1.5 bg-canvas-bg border-t border-canvas-border/40"
        />
      )}

      {expanded && (
        <div className="bg-canvas-bg border-t border-canvas-border">
          <div className="flex items-center gap-2 px-4 py-1.5 border-b border-canvas-border/50">
            <Terminal size={11} className="text-canvas-muted" />
            <span className="text-xs text-canvas-muted flex-1 tabular-nums">
              {proc.logLines.length} lines
            </span>
            <label className="flex items-center gap-1.5 text-xs text-canvas-muted cursor-pointer">
              <input
                type="checkbox"
                checked={autoScroll}
                onChange={(e) => setAutoScroll(e.target.checked)}
                className="accent-accent-primary"
              />
              <ArrowDown size={10} />
              Auto-scroll
            </label>
          </div>
          <div
            ref={logRef}
            className="h-48 overflow-y-auto px-4 py-2 font-mono text-xs"
          >
            {proc.logLines.length === 0 && (
              <span className="text-canvas-muted">No output yet…</span>
            )}
            {proc.logLines.map((line, i) => (
              <div key={i} className="py-px">
                <LogLine line={line} />
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
