import { FolderOpen, FolderTree, RefreshCw } from "lucide-react";
import { OpenPathToolbar } from "../../../components/common/OpenPathToolbar";
import { SimMonitorDayScrubber } from "../../../components/monitor/sim/SimMonitorHeaderControls";
import { StalenessDot } from "../../../components/ui/telemetry/status";
import type { SimulationDirectoryCatalog } from "../../../types";
import {
  formatPipelineTimingBadge,
  type ArrowPipelineTiming,
} from "../../../utils/duckdb/arrowPipeline";

export function SimulationMonitorToolbar({
  openLog,
  openDirectory,
  catalog,
  selectedRunKey,
  handleRunChange,
  isWatching,
  lastEventAt,
  watchPath,
  projectRoot,
  duckdbLoading,
  lastPipeline,
  policies,
  selectedPolicy,
  setPolicy,
  samples,
  selectedSample,
  setSampleId,
  dayRange,
  displayDay,
  setSelectedDay,
  isPlaying,
  setIsPlaying,
  playbackSpeed,
  setPlaybackSpeed,
  selectedDay,
}: {
  openLog: () => void;
  openDirectory: () => void;
  catalog: SimulationDirectoryCatalog | null;
  selectedRunKey: string | null;
  handleRunChange: (runKey: string) => void;
  isWatching: boolean;
  lastEventAt: number | null;
  watchPath: string | null;
  projectRoot: string;
  duckdbLoading: boolean;
  lastPipeline: ArrowPipelineTiming | null;
  policies: string[];
  selectedPolicy: string | null;
  setPolicy: (policy: string | null) => void;
  samples: number[];
  selectedSample: number | null;
  setSampleId: (sampleId: number | null) => void;
  dayRange: { min: number; max: number };
  displayDay: number;
  setSelectedDay: (day: number | null) => void;
  isPlaying: boolean;
  setIsPlaying: (playing: boolean) => void;
  playbackSpeed: 1 | 2 | 4;
  setPlaybackSpeed: (speed: 1 | 2 | 4) => void;
  selectedDay: number | null;
}) {
  return (
    <div className="flex items-center gap-3 flex-wrap">
      <button onClick={openLog} className="btn-primary flex items-center gap-2">
        <FolderOpen size={14} />
        Open Log File
      </button>

      <button onClick={openDirectory} className="btn-secondary flex items-center gap-2">
        <FolderTree size={14} />
        Open Directory
      </button>

      {catalog && catalog.runs.length > 1 && (
        <div className="flex items-center gap-1.5 bg-canvas-subtle px-2 py-1 rounded border border-border-default">
          <span className="text-xs text-canvas-muted whitespace-nowrap">Run:</span>
          <select
            className="select-base text-xs py-0.5 px-2 max-w-xs truncate"
            value={selectedRunKey ?? ""}
            onChange={(e) => void handleRunChange(e.target.value)}
            title={selectedRunKey ?? "Select run"}
          >
            <option value="__ALL__">All Runs ({catalog.runs.length})</option>
            {catalog.runs.map((r) => (
              <option key={r.key} value={r.key} title={r.label}>
                {r.label} ({r.logs.length} log{r.logs.length === 1 ? "" : "s"})
              </option>
            ))}
          </select>
        </div>
      )}

      {catalog && catalog.runs.length === 1 && (
        <span
          className="text-xs bg-canvas-subtle text-canvas-muted px-2 py-1 rounded border border-border-default truncate max-w-xs"
          title={catalog.runs[0].key}
        >
          {catalog.runs[0].label} ({catalog.runs[0].logs.length} log{catalog.runs[0].logs.length === 1 ? "" : "s"})
        </span>
      )}

      {isWatching && <StalenessDot lastEventAt={lastEventAt} />}

      {watchPath && (
        <OpenPathToolbar
          path={watchPath}
          projectRoot={projectRoot}
          kind="log"
          labeled
          labeledTargets={["summary"]}
          trailing={
            <>
              {isWatching && <RefreshCw size={11} className="animate-spin text-accent-success shrink-0" />}
              {duckdbLoading && <span className="shrink-0">· DuckDB ingesting…</span>}
              {!duckdbLoading && lastPipeline?.tableName === "monitor_sim" && (
                <span className="shrink-0">· {formatPipelineTimingBadge(lastPipeline)}</span>
              )}
            </>
          }
        />
      )}

      {policies.length > 0 && (
        <select
          className="select-base w-40"
          value={selectedPolicy ?? ""}
          onChange={(e) => setPolicy(e.target.value || null)}
        >
          <option value="">All policies</option>
          {policies.map((p) => (
            <option key={p} value={p}>{p}</option>
          ))}
        </select>
      )}

      {samples.length > 1 && (
        <select
          className="select-base w-32"
          value={selectedSample ?? ""}
          onChange={(e) => setSampleId(e.target.value ? Number(e.target.value) : null)}
        >
          <option value="">All samples</option>
          {samples.map((s) => (
            <option key={s} value={s}>Sample {s}</option>
          ))}
        </select>
      )}

      <SimMonitorDayScrubber
        dayRange={dayRange}
        displayDay={displayDay}
        setSelectedDay={setSelectedDay}
        isPlaying={isPlaying}
        setIsPlaying={setIsPlaying}
        playbackSpeed={playbackSpeed}
        setPlaybackSpeed={setPlaybackSpeed}
        selectedDay={selectedDay}
        isWatching={isWatching}
      />
    </div>
  );
}
