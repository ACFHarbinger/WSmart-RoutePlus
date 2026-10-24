import { RefreshCw, Archive, Save } from "lucide-react";
import { OpenPathToolbar } from "../common/OpenPathToolbar";
import { PolicyTelemetryTrendsPanel } from "../analysis/telemetry/PolicyTelemetryTrendsPanel";
import {
  formatBytes,
  CSV_EXTENSIONS,
  LOG_EXTENSIONS,
  CONFIG_EXTENSIONS,
} from "./OutputBrowserTree";
import {
  isCheckpointEntry,
  parentRunBrushLabelFromCheckpointPath,
} from "../../utils/training/checkpoints";
import { downloadParquetFromCsv } from "../../utils/charts/tableExport";
import { toast } from "sonner";
import type { DirEntry, OutputDir, WsrouteBundleInfo } from "../../types";

interface OutputBrowserDetailProps {
  selectedRun: OutputDir | null;
  theme: "dark" | "light";
  logScale: boolean;
  activePolicy: string | null;
  activeRunLabel: string | null;
  sourceRunLabel: string | null;
  derivedRunLabel: string | null;
  draggingBundle: boolean;
  viewingPath: string | null;
  viewingExt: string;
  fileLoading: boolean;
  fileContent: string | null;
  csvRows: Array<Record<string, string | number | null>> | null;
  csvHeaders: string[];
  wsrouteBundle: WsrouteBundleInfo | null;
  viewingCheckpoint: DirEntry | null;
  projectRoot: string;
  extractBundleAndOpen: () => void;
  bundleExtracting: boolean;
  parquetExporting: boolean;
  setParquetExporting: (val: boolean) => void;
}

export function OutputBrowserDetail({
  selectedRun,
  theme,
  logScale,
  activePolicy,
  activeRunLabel,
  sourceRunLabel,
  derivedRunLabel,
  draggingBundle,
  viewingPath,
  viewingExt,
  fileLoading,
  fileContent,
  csvRows,
  csvHeaders,
  wsrouteBundle,
  viewingCheckpoint,
  projectRoot,
  extractBundleAndOpen,
  bundleExtracting,
  parquetExporting,
  setParquetExporting,
}: OutputBrowserDetailProps) {
  return (
    <div className="flex-1 min-w-0 flex flex-col gap-2 relative overflow-y-auto">
      {selectedRun && (
        <PolicyTelemetryTrendsPanel
          theme={theme}
          logScale={logScale}
          initialPolicy={activePolicy}
          initialRunLabel={activeRunLabel ?? sourceRunLabel ?? derivedRunLabel}
        />
      )}

      {draggingBundle && (
        <div className="absolute inset-0 z-10 flex items-center justify-center rounded-lg border-2 border-dashed border-accent-primary bg-accent-primary/10 pointer-events-none">
          <p className="text-sm text-accent-primary font-medium">Drop .wsroute bundle to inspect</p>
        </div>
      )}
      {!viewingPath && !fileLoading && (
        <div className="card flex-1 flex flex-col items-center justify-center text-canvas-muted text-sm gap-2">
          <p>Select a file to view its contents.</p>
          <p className="text-xs">Or drag a <code className="font-mono">.wsroute</code> bundle anywhere in the app</p>
        </div>
      )}

      {fileLoading && (
        <div className="card flex-1 flex items-center justify-center gap-2 text-canvas-muted text-sm">
          <RefreshCw size={14} className="animate-spin" />
          Loading…
        </div>
      )}

      {!fileLoading && (fileContent !== null || csvRows !== null || wsrouteBundle !== null || viewingCheckpoint !== null) && (
        <div className="flex items-center gap-3 shrink-0">
          {viewingPath ? (
            <OpenPathToolbar
              path={viewingPath}
              projectRoot={projectRoot}
              kind={
                viewingCheckpoint
                  ? "checkpoint"
                  : LOG_EXTENSIONS.has(viewingExt)
                    ? "log"
                    : CSV_EXTENSIONS.has(viewingExt)
                      ? "csv"
                      : CONFIG_EXTENSIONS.has(viewingExt)
                        ? "config"
                        : null
              }
              handoff={
                viewingCheckpoint
                  ? "checkpoint"
                  : LOG_EXTENSIONS.has(viewingExt)
                    ? "log"
                    : CSV_EXTENSIONS.has(viewingExt)
                      ? "csv"
                      : CONFIG_EXTENSIONS.has(viewingExt)
                        ? "config"
                        : true
              }
              brushLabel={
                isCheckpointEntry({ is_dir: false, extension: viewingExt })
                  ? parentRunBrushLabelFromCheckpointPath(viewingPath, projectRoot)
                  : undefined
              }
              chipClassName="flex-1"
              labeled={
                LOG_EXTENSIONS.has(viewingExt) ||
                CSV_EXTENSIONS.has(viewingExt) ||
                CONFIG_EXTENSIONS.has(viewingExt) ||
                viewingCheckpoint != null
              }
              labeledIconSize={12}
              order="chip-first"
              className="flex-1 min-w-0"
            >
              {viewingExt === "wsroute" && (
                <button
                  onClick={extractBundleAndOpen}
                  disabled={bundleExtracting}
                  className="btn-ghost text-xs flex items-center gap-1.5 text-accent-primary shrink-0"
                >
                  {bundleExtracting ? (
                    <RefreshCw size={12} className="animate-spin" />
                  ) : (
                    <Archive size={12} />
                  )}
                  Extract & Open
                </button>
              )}
              {CSV_EXTENSIONS.has(viewingExt) && projectRoot && (
                <button
                  onClick={async () => {
                    setParquetExporting(true);
                    try {
                      const out = await downloadParquetFromCsv(
                        projectRoot,
                        viewingPath,
                        viewingPath.replace(/\.csv$/i, ".parquet")
                      );
                      if (out) toast.success("Parquet export complete", { description: out.split("/").pop() });
                    } catch (err) {
                      toast.error("Parquet export failed", { description: String(err) });
                    } finally {
                      setParquetExporting(false);
                    }
                  }}
                  disabled={parquetExporting}
                  className="btn-ghost text-xs flex items-center gap-1.5 text-accent-primary shrink-0"
                >
                  {parquetExporting ? (
                    <RefreshCw size={12} className="animate-spin" />
                  ) : (
                    <Save size={12} />
                  )}
                  Export Parquet
                </button>
              )}
            </OpenPathToolbar>
          ) : null}
        </div>
      )}

      {!fileLoading && viewingCheckpoint !== null && (
        <div className="card flex-1 flex flex-col items-center justify-center gap-3 text-sm">
          <OpenPathToolbar
            path={viewingCheckpoint.path}
            projectRoot={projectRoot}
            kind="checkpoint"
            label={viewingCheckpoint.name}
            brushLabel={parentRunBrushLabelFromCheckpointPath(
              viewingCheckpoint.path,
              projectRoot
            )}
            chipClassName="max-w-full px-4"
            labeled
            labeledIconSize={12}
            order="chip-first"
            className="flex-col items-center gap-3"
          />
          <p className="text-canvas-muted text-xs">
            {formatBytes(viewingCheckpoint.size_bytes)} checkpoint weight file
          </p>
        </div>
      )}

      {!fileLoading && fileContent !== null && (
        <div className="card flex-1 overflow-auto">
          <pre className="font-mono text-xs text-gray-300 whitespace-pre-wrap">{fileContent}</pre>
        </div>
      )}

      {!fileLoading && wsrouteBundle !== null && (
        <div className="card flex-1 overflow-auto space-y-3">
          <div className="flex flex-wrap gap-4 text-xs">
            <span className="text-canvas-muted">
              Version: <span className="font-mono text-gray-300">{wsrouteBundle.version ?? "—"}</span>
            </span>
            <span className="text-canvas-muted">
              Created: <span className="font-mono text-gray-300">{wsrouteBundle.created_at ?? "—"}</span>
            </span>
            <span className="text-canvas-muted">
              Files: <span className="font-mono text-gray-300">{wsrouteBundle.files.length}</span>
            </span>
            {wsrouteBundle.arrow_sidecars != null && wsrouteBundle.arrow_sidecars > 0 && (
              <span className="text-canvas-muted">
                Arrow sidecars:{" "}
                <span className="font-mono text-accent-success">{wsrouteBundle.arrow_sidecars}</span>
              </span>
            )}
          </div>
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-canvas-border">
                <th className="text-left py-2 px-3 text-canvas-muted font-medium">Path</th>
                <th className="text-right py-2 px-3 text-canvas-muted font-medium">Size</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-canvas-border">
              {wsrouteBundle.files.map((f) => (
                <tr key={f.path} className="hover:bg-canvas-hover">
                  <td className="py-1.5 px-3 max-w-md">
                    <OpenPathToolbar
                      path={f.path}
                      projectRoot={projectRoot}
                      brushLabel={sourceRunLabel ?? undefined}
                      chipClassName="max-w-full"
                      className="max-w-full min-w-0"
                    />
                  </td>
                  <td className="py-1.5 px-3 text-right text-canvas-muted">{formatBytes(f.size_bytes)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {!fileLoading && csvRows !== null && (
        <div className="card flex-1 overflow-auto">
          <table className="w-full text-xs">
            <thead className="bg-canvas-elevated sticky top-0">
              <tr>
                {csvHeaders.map((h) => (
                  <th key={h} className="px-3 py-2 text-left text-canvas-muted font-medium whitespace-nowrap">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-canvas-border">
              {csvRows.slice(0, 200).map((row, i) => (
                <tr key={i} className="hover:bg-canvas-hover">
                  {csvHeaders.map((h) => (
                    <td key={h} className="px-3 py-1.5 text-gray-300 whitespace-nowrap font-mono">
                      {row[h] ?? "—"}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
          {csvRows.length > 200 && (
            <p className="text-xs text-canvas-muted p-3">
              Showing 200 / {csvRows.length} rows. Use Data Explorer for full pagination.
            </p>
          )}
        </div>
      )}
    </div>
  );
}
