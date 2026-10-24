import { ExternalLink } from "lucide-react";
import { GlobalFilterBar } from "../layout/GlobalFilterBar";
import { LauncherLivePanel } from "../monitor/live/LauncherLivePanel";
import { ProcessIdFooter } from "../monitor/process/ProcessIdFooter";
import { PolicyTelemetryPanel } from "../analysis/telemetry/PolicyTelemetryPanel";
import { PolicyTelemetryTrendsPanel } from "../analysis/telemetry/PolicyTelemetryTrendsPanel";
import { simLivePanelTitle } from "../../utils/process/launcherProcess";
import type {
  DayLogEntry,
  PolicyVizEntry,
  ProcessStatus,
} from "../../types";

function fmt(n: number | undefined, decimals = 1): string {
  if (n == null || isNaN(n)) return "—";
  return n.toFixed(decimals);
}

export function PolicyLiveCard({
  policy,
  entry,
  brushed,
  onBrush,
}: {
  policy: string;
  entry: DayLogEntry;
  brushed?: boolean;
  onBrush?: () => void;
}) {
  const d = entry.data;
  return (
    <div className="rounded-lg border border-canvas-border bg-canvas-surface px-3 py-2 space-y-1">
      <div className="flex items-center justify-between">
        <span
          className={`text-xs font-mono font-semibold ${
            brushed ? "text-accent-secondary" : "text-gray-200"
          } ${onBrush ? "cursor-pointer hover:text-accent-primary" : ""}`}
          title={onBrush ? `${policy} — click to brush policy` : undefined}
          onClick={onBrush}
        >
          {policy}
        </span>
        <span className="text-xs text-canvas-muted">Day {entry.day}</span>
      </div>
      <div className="grid grid-cols-2 gap-x-4 gap-y-0.5 text-xs">
        <span className="text-canvas-muted">Profit</span>
        <span className="font-mono text-gray-200">{fmt(d.profit)} €</span>
        <span className="text-canvas-muted">Distance</span>
        <span className="font-mono text-gray-200">{fmt(d.km)} km</span>
        <span className="text-canvas-muted">Overflows</span>
        <span
          className={`font-mono ${
            (d.overflows ?? 0) > 0 ? "text-accent-danger" : "text-accent-success"
          }`}
        >
          {d.overflows ?? "—"}
        </span>
        <span className="text-canvas-muted">Waste</span>
        <span className="font-mono text-gray-200">{fmt(d.kg)} kg</span>
      </div>
    </div>
  );
}

export function SimLiveMonitorSection({
  displayProcessId,
  vizPolicies,
  liveRunLabel,
  activeRunLabel,
  isDone,
  simStatus,
  liveLogPath,
  projectRoot,
  outputRunPath,
  simJsonlPath,
  openCompletedInSummary,
  liveLogLines,
  liveEntries,
  activePolicy,
  setPolicy,
  policyVizEntries,
  selectedPolicy,
  theme,
  logScale,
  policyVizLive,
  telemetryTrendsKey,
}: {
  displayProcessId: string;
  vizPolicies: string[];
  liveRunLabel: string | null;
  activeRunLabel: string | null;
  isDone: boolean;
  simStatus: ProcessStatus | null;
  liveLogPath: string | null;
  projectRoot: string | null;
  outputRunPath: string | null;
  simJsonlPath: string | null;
  openCompletedInSummary: () => void;
  liveLogLines: string[];
  liveEntries: DayLogEntry[];
  activePolicy: string | null;
  setPolicy: (p: string | null) => void;
  policyVizEntries: PolicyVizEntry[];
  selectedPolicy: string | null;
  theme: "dark" | "light";
  logScale: boolean;
  policyVizLive: boolean;
  telemetryTrendsKey: number;
}) {
  return (
    <div className="space-y-3">
      <GlobalFilterBar
        policies={vizPolicies.length > 0 ? vizPolicies : undefined}
        runLabels={liveRunLabel ? [liveRunLabel] : []}
        showLogScale
      />
      <LauncherLivePanel
        header={{
          status: isDone ? (simStatus ?? "running") : "running",
          title: simLivePanelTitle({
            isRunning: !isDone,
            status: simStatus ?? undefined,
          }),
          runLabel: liveRunLabel,
          logPath: liveLogPath,
          projectRoot,
          navMesh: {
            kind: "sim",
            hideSelf: true,
            showPostRun: isDone && simStatus === "completed",
            showOutputBrowser: isDone && simStatus === "completed",
            outputRunPath,
            simLogPath: simJsonlPath,
          },
          navTrailing:
            isDone && simStatus === "completed" ? (
              <button
                type="button"
                onClick={openCompletedInSummary}
                className="btn-primary text-xs py-1 px-3 flex items-center gap-1.5"
              >
                <ExternalLink size={12} />
                Open in Summary
              </button>
            ) : undefined,
        }}
        progress={!isDone && displayProcessId ? { processId: displayProcessId } : undefined}
        footer={
          <ProcessIdFooter
            processId={displayProcessId}
            logPath={liveLogPath}
            runLabel={liveRunLabel}
            projectRoot={projectRoot}
          />
        }
        logLines={liveLogLines}
        logTailWaiting={!isDone}
      >
        {liveEntries.length === 0 ? (
          <p className="text-xs text-canvas-muted">
            {isDone ? "No day entries received." : "Waiting for first day log…"}
          </p>
        ) : (
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
            {liveEntries
              .sort((a, b) => a.policy.localeCompare(b.policy))
              .map((entry) => (
                <PolicyLiveCard
                  key={`${entry.policy}::${entry.sample_id}`}
                  policy={`${entry.policy}${
                    liveEntries.filter((e) => e.policy === entry.policy).length > 1
                      ? ` #${entry.sample_id}`
                      : ""
                  }`}
                  entry={entry}
                  brushed={activePolicy === entry.policy}
                  onBrush={() =>
                    setPolicy(activePolicy === entry.policy ? null : entry.policy)
                  }
                />
              ))}
          </div>
        )}
      </LauncherLivePanel>

      {policyVizEntries.length > 0 && (
        <div className="space-y-3">
          {vizPolicies.length > 1 && (
            <div className="flex flex-wrap gap-1.5">
              {vizPolicies.map((pol) => (
                <button
                  key={pol}
                  onClick={() => setPolicy(activePolicy === pol ? null : pol)}
                  className={`text-[10px] font-mono px-2 py-0.5 rounded border transition-colors ${
                    selectedPolicy === pol
                      ? "border-accent-secondary text-accent-secondary bg-accent-secondary/10"
                      : "border-canvas-border text-gray-400 hover:text-gray-200"
                  }`}
                >
                  {pol}
                </button>
              ))}
            </div>
          )}
          <PolicyTelemetryPanel
            entries={policyVizEntries}
            policy={selectedPolicy}
            sampleId={null}
            day={null}
            theme={theme}
            logScale={logScale}
            live={policyVizLive}
          />
          <PolicyTelemetryTrendsPanel
            theme={theme}
            logScale={logScale}
            refreshKey={telemetryTrendsKey}
            initialPolicy={selectedPolicy}
            initialRunLabel={liveRunLabel ?? activeRunLabel}
          />
        </div>
      )}
    </div>
  );
}
