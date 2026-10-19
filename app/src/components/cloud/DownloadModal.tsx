import { useEffect, useState, type FormEvent } from "react";
import { Download, X, CheckCircle2, AlertCircle, RefreshCw, Folder } from "lucide-react";
import type { CloudDownloadProgressEvent } from "../../types/cloud";
import { downloadCloudJobResults, onCloudDownloadProgress } from "../../utils/cloud";

export interface DownloadModalProps {
  jobId: string | null;
  isOpen: boolean;
  onClose: () => void;
  projectRoot?: string;
  root?: string;
}

export function DownloadModal({
  jobId,
  isOpen,
  onClose,
  projectRoot,
  root,
}: DownloadModalProps) {
  const [dest, setDest] = useState<string>("");
  const [downloading, setDownloading] = useState<boolean>(false);
  const [progressEvent, setProgressEvent] = useState<CloudDownloadProgressEvent | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [completed, setCompleted] = useState<boolean>(false);
  const [fileCount, setFileCount] = useState<number>(0);

  useEffect(() => {
    if (jobId) {
      setDest(`outputs/cloud/results/${jobId}`);
      setProgressEvent(null);
      setError(null);
      setCompleted(false);
      setFileCount(0);
    }
  }, [jobId]);

  // Listen to download progress events
  useEffect(() => {
    if (!isOpen || !jobId) return;
    let unlistenFn: (() => void) | undefined;

    onCloudDownloadProgress((event) => {
      if (event.job_id === jobId) {
        setProgressEvent(event);
        if (event.stage === "completed") {
          setCompleted(true);
        } else if (event.stage === "failed") {
          setError(event.message || "Download failed");
        }
      }
    })
      .then((unlisten) => {
        unlistenFn = unlisten;
      })
      .catch(() => {});

    return () => {
      if (unlistenFn) unlistenFn();
    };
  }, [isOpen, jobId]);

  if (!isOpen || !jobId) return null;

  const handleStartDownload = async (e: FormEvent) => {
    e.preventDefault();
    setDownloading(true);
    setError(null);
    setCompleted(false);

    try {
      const res = await downloadCloudJobResults(jobId, dest.trim() || undefined, {
        projectRoot,
        root,
      });
      setFileCount(res.files);
      setCompleted(true);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setDownloading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4">
      <div className="bg-canvas-surface border border-canvas-border rounded-xl shadow-2xl w-full max-w-md p-5 space-y-4">
        <div className="flex items-center justify-between border-b border-canvas-border pb-3">
          <div className="flex items-center gap-2">
            <Download size={18} className="text-accent-primary" />
            <h3 className="text-sm font-semibold text-gray-100">
              Download Job Results
            </h3>
          </div>
          <button
            onClick={onClose}
            className="text-canvas-muted hover:text-gray-200 transition-colors"
          >
            <X size={16} />
          </button>
        </div>

        <form onSubmit={handleStartDownload} className="space-y-4">
          <div className="space-y-1">
            <label className="text-xs text-canvas-muted">
              Target Job ID
            </label>
            <input
              type="text"
              readOnly
              className="input-base font-mono text-xs bg-canvas-elevated/40 text-canvas-muted"
              value={jobId}
            />
          </div>

          <div className="space-y-1">
            <label className="text-xs text-canvas-muted">
              Destination Directory (--dest)
            </label>
            <div className="relative">
              <input
                type="text"
                disabled={downloading}
                className="input-base font-mono text-xs pr-8"
                value={dest}
                onChange={(e) => setDest(e.target.value)}
                placeholder="outputs/cloud/results/<job_id>"
              />
              <Folder
                size={14}
                className="absolute right-2.5 top-2.5 text-canvas-muted"
              />
            </div>
            <p className="text-[10px] text-canvas-muted">
              Daily simulation checkpoints, day logs, and results index will download
              into this directory with checksum verification.
            </p>
          </div>

          {/* Progress / Status display */}
          {downloading && (
            <div className="p-3 bg-canvas-elevated/50 border border-canvas-border rounded-lg space-y-2 text-xs">
              <div className="flex items-center justify-between text-gray-200">
                <span className="capitalize">
                  Stage: {progressEvent?.stage ?? "Starting download..."}
                </span>
                <RefreshCw size={13} className="animate-spin text-accent-primary" />
              </div>
              {progressEvent?.total_files && progressEvent.total_files > 0 ? (
                <div className="w-full bg-canvas-surface rounded-full h-1.5 overflow-hidden border border-canvas-border">
                  <div
                    className="bg-accent-primary h-full transition-all duration-300"
                    style={{
                      width: `${Math.min(
                        100,
                        Math.round(
                          (progressEvent.files_downloaded /
                            progressEvent.total_files) *
                            100
                        )
                      )}%`,
                    }}
                  />
                </div>
              ) : null}
              {progressEvent?.message && (
                <p className="text-[11px] text-canvas-muted">
                  {progressEvent.message}
                </p>
              )}
            </div>
          )}

          {completed && (
            <div className="flex items-center gap-2 p-3 bg-accent-success/10 border border-accent-success/20 rounded-lg text-xs text-accent-success">
              <CheckCircle2 size={16} className="shrink-0" />
              <div>
                <p className="font-medium">Download complete!</p>
                <p className="text-[11px] text-gray-300">
                  {fileCount} file(s) saved to <code className="font-mono">{dest}</code>.
                </p>
              </div>
            </div>
          )}

          {error && (
            <div className="flex items-start gap-2 p-3 bg-accent-danger/10 border border-accent-danger/20 rounded-lg text-xs text-accent-danger">
              <AlertCircle size={16} className="shrink-0 mt-0.5" />
              <span>{error}</span>
            </div>
          )}

          <div className="flex items-center justify-end gap-2 pt-2 border-t border-canvas-border">
            <button
              type="button"
              onClick={onClose}
              disabled={downloading}
              className="btn-ghost text-xs px-3 py-1.5"
            >
              {completed ? "Done" : "Cancel"}
            </button>
            {!completed && (
              <button
                type="submit"
                disabled={downloading}
                className="btn-primary text-xs flex items-center gap-1.5 px-4 py-1.5"
              >
                {downloading ? (
                  <>
                    <RefreshCw size={13} className="animate-spin" />
                    <span>Downloading...</span>
                  </>
                ) : (
                  <>
                    <Download size={13} />
                    <span>Download Results</span>
                  </>
                )}
              </button>
            )}
          </div>
        </form>
      </div>
    </div>
  );
}
