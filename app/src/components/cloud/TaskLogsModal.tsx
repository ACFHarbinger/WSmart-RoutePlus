import { useEffect, useRef, useState, useCallback } from "react";
import {
  Terminal,
  X,
  RefreshCw,
  Copy,
  Check,
  ArrowDown,
  AlertCircle,
} from "lucide-react";
import { getCloudTaskLogs } from "../../utils/cloud";

export interface TaskLogsModalProps {
  taskId: string | null;
  isOpen: boolean;
  onClose: () => void;
  projectRoot?: string;
  root?: string;
}

export function TaskLogsModal({
  taskId,
  isOpen,
  onClose,
  projectRoot,
  root,
}: TaskLogsModalProps) {
  const [logs, setLogs] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [tail, setTail] = useState<number>(200);
  const [autoRefresh, setAutoRefresh] = useState<boolean>(true);
  const [copied, setCopied] = useState<boolean>(false);
  const [autoScroll, setAutoScroll] = useState<boolean>(true);

  const logContainerRef = useRef<HTMLDivElement>(null);

  const fetchLogs = useCallback(async () => {
    if (!taskId) return;
    setLoading(true);
    setError(null);
    try {
      const res = await getCloudTaskLogs(taskId, tail > 0 ? tail : undefined, {
        projectRoot,
        root,
      });
      setLogs(res.log || "(Task log is currently empty)");
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }, [taskId, tail, projectRoot, root]);

  useEffect(() => {
    if (isOpen && taskId) {
      fetchLogs();
    }
  }, [isOpen, taskId, fetchLogs]);

  // Periodic polling if auto-refresh is active
  useEffect(() => {
    if (!isOpen || !taskId || !autoRefresh) return;
    const interval = setInterval(() => {
      fetchLogs();
    }, 3000);
    return () => clearInterval(interval);
  }, [isOpen, taskId, autoRefresh, fetchLogs]);

  // Auto-scroll to bottom on new logs
  useEffect(() => {
    if (autoScroll && logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
    }
  }, [logs, autoScroll]);

  if (!isOpen || !taskId) return null;

  const handleCopy = () => {
    navigator.clipboard.writeText(logs);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4">
      <div className="bg-canvas-surface border border-canvas-border rounded-xl shadow-2xl w-full max-w-4xl h-[80vh] flex flex-col overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-canvas-border shrink-0 bg-canvas-elevated/40">
          <div className="flex items-center gap-2">
            <Terminal size={18} className="text-accent-primary" />
            <h3 className="text-sm font-semibold text-gray-100 font-mono">
              Task Log: {taskId}
            </h3>
          </div>

          <div className="flex items-center gap-2">
            {/* Tail selector */}
            <select
              className="select-base text-xs py-1 px-2 w-28 bg-canvas-surface"
              value={tail}
              onChange={(e) => setTail(Number(e.target.value))}
            >
              <option value={50}>Last 50 lines</option>
              <option value={100}>Last 100 lines</option>
              <option value={200}>Last 200 lines</option>
              <option value={500}>Last 500 lines</option>
              <option value={0}>All output</option>
            </select>

            {/* Auto refresh toggle */}
            <label className="flex items-center gap-1.5 text-xs text-canvas-muted cursor-pointer select-none">
              <input
                type="checkbox"
                checked={autoRefresh}
                onChange={(e) => setAutoRefresh(e.target.checked)}
                className="rounded border-canvas-border bg-canvas-surface text-accent-primary"
              />
              <span>Live Poll</span>
            </label>

            {/* Manual Refresh */}
            <button
              onClick={fetchLogs}
              disabled={loading}
              className="btn-ghost p-1.5 text-canvas-muted hover:text-gray-100"
              title="Refresh logs"
            >
              <RefreshCw size={14} className={loading ? "animate-spin" : ""} />
            </button>

            {/* Copy button */}
            <button
              onClick={handleCopy}
              className="btn-ghost p-1.5 text-canvas-muted hover:text-gray-100"
              title="Copy to clipboard"
            >
              {copied ? (
                <Check size={14} className="text-accent-success" />
              ) : (
                <Copy size={14} />
              )}
            </button>

            {/* Close */}
            <button
              onClick={onClose}
              className="btn-ghost p-1.5 text-canvas-muted hover:text-gray-100"
            >
              <X size={16} />
            </button>
          </div>
        </div>

        {/* Content */}
        <div
          ref={logContainerRef}
          className="flex-1 p-4 bg-canvas-elevated font-mono text-xs overflow-y-auto leading-relaxed select-text"
        >
          {error ? (
            <div className="flex items-center gap-2 text-accent-danger p-3 bg-accent-danger/10 border border-accent-danger/20 rounded-lg">
              <AlertCircle size={16} />
              <span>{error}</span>
            </div>
          ) : (
            <pre className="text-gray-300 whitespace-pre-wrap break-all font-mono">
              {logs}
            </pre>
          )}
        </div>

        {/* Footer controls */}
        <div className="flex items-center justify-between px-4 py-2 border-t border-canvas-border shrink-0 bg-canvas-surface text-xs text-canvas-muted">
          <label className="flex items-center gap-1.5 cursor-pointer">
            <input
              type="checkbox"
              checked={autoScroll}
              onChange={(e) => setAutoScroll(e.target.checked)}
              className="rounded border-canvas-border bg-canvas-surface text-accent-primary"
            />
            <ArrowDown size={12} />
            <span>Auto-scroll to bottom</span>
          </label>

          <span>
            {logs ? `${logs.split("\n").length} lines loaded` : "No output"}
          </span>
        </div>
      </div>
    </div>
  );
}
