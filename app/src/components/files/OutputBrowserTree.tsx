import {
  Folder,
  FolderOpen,
  File,
  FileText,
  RefreshCw,
  ChevronRight,
  ChevronDown,
  BarChart2,
  Save,
  Trash2,
  Package,
} from "lucide-react";
import { OpenPathToolbar } from "../common/OpenPathToolbar";
import { LoadedRunRow } from "../common/LoadedRunRow";
import { parentRunBrushLabelFromCheckpointPath, isCheckpointEntry } from "../../utils/training/checkpoints";
import type { DirEntry, OutputDir } from "../../types";
import type { SessionProfile } from "../../store/sessionProfiles";

export function formatBytes(b: number) {
  if (b < 1024) return `${b} B`;
  if (b < 1024 ** 2) return `${(b / 1024).toFixed(1)} KB`;
  return `${(b / 1024 ** 2).toFixed(1)} MB`;
}

export const TEXT_EXTENSIONS = new Set(["yaml", "yml", "toml", "cfg", "ini", "txt", "log", "md", "json", "jsonl"]);
export const CSV_EXTENSIONS = new Set(["csv"]);
export const LOG_EXTENSIONS = new Set(["jsonl"]);
export const CONFIG_EXTENSIONS = new Set(["yaml", "yml", "toml", "cfg", "ini"]);

export function FileIcon({ entry }: { entry: DirEntry }) {
  if (entry.is_dir) return <Folder size={13} className="text-accent-warning" />;
  if (entry.extension === "wsroute") return <File size={13} className="text-accent-primary" />;
  if (CSV_EXTENSIONS.has(entry.extension)) return <File size={13} className="text-accent-success" />;
  if (TEXT_EXTENSIONS.has(entry.extension)) return <FileText size={13} className="text-accent-secondary" />;
  return <File size={13} className="text-canvas-muted" />;
}

export function parseYamlFlat(yaml: string): Array<{ key: string; value: string }> {
  const rows: Array<{ key: string; value: string }> = [];
  const stack: string[] = [];
  for (const line of yaml.split("\n")) {
    const stripped = line.replace(/\t/g, "  ");
    const trimmed = stripped.trimStart();
    if (!trimmed || trimmed.startsWith("#")) continue;
    const indent = (stripped.length - trimmed.length) / 2;
    stack.length = indent;
    const colonIdx = trimmed.indexOf(":");
    if (colonIdx === -1) continue;
    const key = trimmed.slice(0, colonIdx).trim();
    const value = trimmed.slice(colonIdx + 1).trim();
    stack[indent] = key;
    if (value && !value.startsWith("#")) {
      rows.push({ key: stack.filter(Boolean).join("."), value });
    }
  }
  return rows;
}

export const META_KEYS = [
  "task", "seed", "envs", "models", "model.encoder.type",
  "sim.graph.area", "sim.graph.num_loc", "sim.data_distribution",
  "sim.policies", "train.max_epochs", "hpo.n_trials",
];

export function sortEntries(list: DirEntry[]): DirEntry[] {
  const priority = (e: DirEntry) => {
    if (e.is_dir) return 0;
    if (e.name === "pruned_config.yaml" || e.name === "config.yaml") return 1;
    if (e.extension === "jsonl") return 2;
    return 3;
  };
  return [...list].sort((a, b) => {
    const pa = priority(a);
    const pb = priority(b);
    if (pa !== pb) return pa - pb;
    return a.name.localeCompare(b.name);
  });
}

interface OutputBrowserRunListProps {
  runs: OutputDir[];
  selectedRun: OutputDir | null;
  selectRun: (run: OutputDir) => void;
  compareSelection: Set<string>;
  toggleCompareRun: (runPath: string) => void;
  compareSelectedRuns: () => void;
  refresh: () => void;
  pickOutputDir: () => void;
  loading: boolean;
  projectRoot: string;
  activeRunLabel: string | null;
  profiles: SessionProfile[];
  profileName: string;
  setProfileName: (name: string) => void;
  saveProfile: (name: string) => void;
  loadProfile: (id: string) => void;
  deleteProfile: (id: string) => void;
}

export function OutputBrowserRunList({
  runs,
  selectedRun,
  selectRun,
  compareSelection,
  toggleCompareRun,
  compareSelectedRuns,
  refresh,
  pickOutputDir,
  loading,
  projectRoot,
  activeRunLabel,
  profiles,
  profileName,
  setProfileName,
  saveProfile,
  loadProfile,
  deleteProfile,
}: OutputBrowserRunListProps) {
  return (
    <div className="w-56 shrink-0 flex flex-col gap-2">
      <div className="flex items-center gap-2">
        <button
          onClick={refresh}
          disabled={loading}
          className="btn-ghost p-1.5 text-canvas-muted"
          title="Refresh"
        >
          <RefreshCw size={13} className={loading ? "animate-spin" : ""} />
        </button>
        <button
          onClick={pickOutputDir}
          className="btn-ghost p-1.5 text-canvas-muted"
          title="Open directory"
        >
          <FolderOpen size={13} />
        </button>
        <span className="text-xs text-canvas-muted">{runs.length} runs</span>
      </div>

      {compareSelection.size >= 2 && (
        <button
          onClick={compareSelectedRuns}
          className="btn-primary text-xs flex items-center gap-1.5 w-full justify-center"
        >
          <BarChart2 size={12} />
          Compare {compareSelection.size} Runs →
        </button>
      )}

      <div className="card flex-1 overflow-auto p-1 space-y-0.5">
        {runs.map((r) => (
          <LoadedRunRow
            key={r.path}
            path={r.path}
            projectRoot={projectRoot}
            label={r.name}
            activeRunLabel={activeRunLabel}
            selected={selectedRun?.path === r.path}
            pathHandoffs
            handoffKind="run"
            className="mx-0 px-1"
            leading={
              <>
                <input
                  type="checkbox"
                  checked={compareSelection.has(r.path)}
                  onChange={() => toggleCompareRun(r.path)}
                  className="shrink-0 accent-accent-primary"
                  title="Select for comparison"
                />
                <button
                  type="button"
                  onClick={() => void selectRun(r)}
                  className={`shrink-0 p-0.5 rounded hover:bg-canvas-hover transition-colors ${
                    selectedRun?.path === r.path
                      ? "text-accent-secondary"
                      : "text-accent-warning"
                  }`}
                  title="Open run directory"
                >
                  <Folder size={12} />
                </button>
              </>
            }
          />
        ))}
        {runs.length === 0 && !loading && (
          <p className="text-xs text-canvas-muted p-2">No run directories found.</p>
        )}
      </div>

      {/* Session profiles */}
      <div className="card p-2 space-y-2 shrink-0">
        <p className="text-xs font-semibold text-gray-300">Session Profiles</p>
        <div className="flex gap-1">
          <input
            type="text"
            className="input-base text-xs flex-1"
            value={profileName}
            onChange={(e) => setProfileName(e.target.value)}
            placeholder="Profile name…"
          />
          <button
            onClick={() => {
              saveProfile(profileName);
              setProfileName("");
            }}
            className="btn-ghost p-1.5"
            title="Save current launcher forms"
          >
            <Save size={12} />
          </button>
        </div>
        {profiles.length > 0 ? (
          <div className="max-h-28 overflow-y-auto space-y-0.5">
            {profiles.map((p) => (
              <div key={p.id} className="flex items-center gap-1 text-xs">
                <button
                  onClick={() => loadProfile(p.id)}
                  className="flex-1 text-left truncate text-gray-300 hover:text-accent-secondary py-0.5"
                >
                  {p.name}
                </button>
                <button
                  onClick={() => deleteProfile(p.id)}
                  className="text-canvas-muted hover:text-accent-danger p-0.5"
                >
                  <Trash2 size={10} />
                </button>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-[10px] text-canvas-muted">No saved profiles yet.</p>
        )}
      </div>
    </div>
  );
}

interface OutputBrowserDirTreeProps {
  selectedRun: OutputDir | null;
  entries: DirEntry[];
  expandedDirs: Set<string>;
  subEntries: Record<string, DirEntry[]>;
  openFile: (entry: DirEntry) => void;
  viewingPath: string | null;
  includeArrowSidecars: boolean;
  setIncludeArrowSidecars: (val: boolean) => void;
  exportRunAsBundle: () => void;
  bundleExporting: boolean;
  runJsonlPath: string | null;
  projectRoot: string;
  runMeta: Array<{ key: string; value: string }> | null;
  runCheckpoints: DirEntry[] | null;
  runKpi: Array<{ policy: string; overflows: number; kgkm: number; profit: number }> | null;
  activePolicy: string | null;
  setPolicy: (p: string | null) => void;
}

export function OutputBrowserDirTree({
  selectedRun,
  entries,
  expandedDirs,
  subEntries,
  openFile,
  viewingPath,
  includeArrowSidecars,
  setIncludeArrowSidecars,
  exportRunAsBundle,
  bundleExporting,
  runJsonlPath,
  projectRoot,
  runMeta,
  runCheckpoints,
  runKpi,
  activePolicy,
  setPolicy,
}: OutputBrowserDirTreeProps) {
  function renderEntries(list: DirEntry[], depth = 0): React.ReactNode {
    return list.map((e) => (
      <div key={e.path}>
        <button
          onClick={() => openFile(e)}
          className={`w-full flex items-center gap-2 py-1 px-2 rounded text-xs text-left hover:bg-canvas-hover transition-colors ${
            viewingPath === e.path ? "bg-accent-primary/15 text-accent-secondary" : "text-gray-300"
          }`}
          style={{ paddingLeft: `${8 + depth * 14}px` }}
        >
          {e.is_dir ? (
            expandedDirs.has(e.path)
              ? <ChevronDown size={11} className="shrink-0 text-canvas-muted" />
              : <ChevronRight size={11} className="shrink-0 text-canvas-muted" />
          ) : (
            <span className="w-[11px] shrink-0" />
          )}
          <FileIcon entry={e} />
          <span
            className={`truncate flex-1 ${
              e.name === "pruned_config.yaml" || e.name === "config.yaml"
                ? "text-accent-secondary font-medium"
                : e.extension === "jsonl"
                ? "text-accent-success"
                : isCheckpointEntry(e)
                ? "text-accent-primary font-medium"
                : ""
            }`}
          >
            {e.name}
          </span>
          {!e.is_dir && (
            <span className="text-canvas-muted shrink-0">{formatBytes(e.size_bytes)}</span>
          )}
        </button>
        {e.is_dir && expandedDirs.has(e.path) && subEntries[e.path] && (
          <div>{renderEntries(subEntries[e.path], depth + 1)}</div>
        )}
      </div>
    ));
  }

  if (!selectedRun) {
    return (
      <div className="w-56 shrink-0 flex flex-col gap-2">
        <div className="card flex-1 flex items-center justify-center text-canvas-muted text-xs">
          Select a run
        </div>
      </div>
    );
  }

  return (
    <div className="w-56 shrink-0 flex flex-col gap-2">
      <label className="flex items-center gap-2 text-[10px] text-canvas-muted px-1 shrink-0 cursor-pointer">
        <input
          type="checkbox"
          checked={includeArrowSidecars}
          onChange={(e) => setIncludeArrowSidecars(e.target.checked)}
          className="accent-accent-primary"
        />
        Include Arrow IPC sidecars for CSV + JSONL (§G.8)
      </label>
      <button
        onClick={exportRunAsBundle}
        disabled={bundleExporting}
        className="btn-ghost text-xs flex items-center gap-1.5 w-full justify-center shrink-0"
        title="Package run artefacts into a .wsroute zip bundle"
      >
        {bundleExporting ? (
          <RefreshCw size={12} className="animate-spin" />
        ) : (
          <Package size={12} />
        )}
        Export as .wsroute
      </button>
      <div className="card flex-1 overflow-auto p-1">
        <div className="px-2 py-1 shrink-0">
          <OpenPathToolbar
            path={runJsonlPath ?? selectedRun.path}
            projectRoot={projectRoot}
            kind={runJsonlPath ? "log" : "run"}
            storedLabel={selectedRun.name}
            chipClassName="font-medium max-w-none"
            labeled={!!runJsonlPath}
            labeledIconSize={12}
          />
        </div>
        {renderEntries(entries)}
      </div>
      {runMeta && runMeta.length > 0 && (
        <div className="card p-2 space-y-1 overflow-auto max-h-36">
          <p className="text-xs font-semibold text-gray-300 mb-1.5">Config</p>
          {runMeta.map((r) => (
            <div key={r.key} className="flex gap-2 text-xs leading-tight">
              <span className="text-canvas-muted font-mono truncate max-w-[50%]" title={r.key}>
                {r.key.split(".").pop()}
              </span>
              <span className="text-gray-300 truncate flex-1" title={r.value}>{r.value}</span>
            </div>
          ))}
        </div>
      )}
      {runCheckpoints && runCheckpoints.length > 0 && (
        <div className="card p-2 space-y-1 overflow-auto max-h-40">
          <p className="text-xs font-semibold text-gray-300 mb-1.5">
            Checkpoints ({runCheckpoints.length})
          </p>
          {runCheckpoints.map((ckpt) => (
            <div key={ckpt.path} className="flex items-center gap-2 text-[10px] leading-tight">
              <OpenPathToolbar
                path={ckpt.path}
                projectRoot={projectRoot}
                kind="checkpoint"
                label={ckpt.name}
                brushLabel={parentRunBrushLabelFromCheckpointPath(ckpt.path, projectRoot)}
                chipClassName="flex-1 min-w-0 max-w-none text-[10px]"
                labeled
                labeledIconSize={11}
                className="flex-1 min-w-0"
              >
                <span className="text-canvas-muted shrink-0">{formatBytes(ckpt.size_bytes)}</span>
              </OpenPathToolbar>
            </div>
          ))}
        </div>
      )}
      {runKpi && runKpi.length > 0 && (
        <div className="card p-2 space-y-1 overflow-auto max-h-48">
          <p className="text-xs font-semibold text-gray-300 mb-1.5">KPI Summary</p>
          <div className="grid grid-cols-3 gap-x-2 text-[10px] text-canvas-muted font-medium mb-1">
            <span>Policy</span><span className="text-right">Overflows</span><span className="text-right">kg/km</span>
          </div>
          {runKpi.map((r) => (
            <div key={r.policy} className="grid grid-cols-3 gap-x-2 text-[10px] leading-tight">
              <span
                className={`truncate font-mono cursor-pointer hover:text-accent-primary ${
                  activePolicy === r.policy ? "text-accent-secondary" : "text-gray-300"
                }`}
                title={`${r.policy} — click to brush policy`}
                onClick={() => setPolicy(activePolicy === r.policy ? null : r.policy)}
              >
                {r.policy}
              </span>
              <span className={`text-right font-mono ${r.overflows === 0 ? "text-accent-success" : r.overflows > 20 ? "text-accent-danger" : "text-accent-warning"}`}>{r.overflows.toFixed(1)}</span>
              <span className="text-right font-mono text-gray-400">{r.kgkm.toFixed(2)}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
