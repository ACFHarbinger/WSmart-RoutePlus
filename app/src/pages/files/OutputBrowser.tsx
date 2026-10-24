/**
 * Output Browser — browse the simulation output folders, outputs/simulations/ and the older assets/output/ (§G.14).
 *
 * Ports the PySide6 file-explorer tab. Provides:
 *   - Run list (from list_output_dirs)
 *   - File tree inside a selected run (from list_dir)
 *   - Run metadata card: auto-loads pruned_config.yaml and shows key fields
 *   - File viewer: CSV → DataExplorer-style table; YAML/text → raw view
 *   - "Open in Sim Summary" / "Open in Simulation Monitor" buttons for .jsonl logs
 *   - "Open in Data Explorer" button for .csv files
 *   - "Open in Config Editor" button for YAML / TOML / cfg / ini files
 */
import { useCallback, useEffect, useMemo, useState } from "react";
import { useLayoutStore } from "../../store/layout";
import { invoke } from "@tauri-apps/api/core";
import { open, save as saveDialog } from "@tauri-apps/plugin-dialog";
import { useAppStore } from "../../store/app";
import { useGlobalFiltersStore } from "../../store/filters";
import { recentKindFromPath } from "../../store/recentFiles";
import { useSessionProfilesStore } from "../../store/sessionProfiles";
import { toast } from "sonner";
import type { DirEntry, OutputDir, DayLogEntry, WsrouteBundleInfo, WsrouteExtractResult } from "../../types";
import { findRunJsonl } from "../../utils/runs/outputRunLogs";
import { useLogPathRunLabelBrush } from "../../hooks/brush/useLogPathRunLabelBrush";
import {
  filterCheckpointEntries,
  isCheckpointEntry,
} from "../../utils/training/checkpoints";
import { portfolioRunLabel } from "../../utils/duckdb/arrowPipeline";
import { useRecentHandoff } from "../../hooks/files/useRecentHandoff";
import { simOutputRoots } from "../../utils/runs/outputRoots";
import {
  OutputBrowserRunList,
  OutputBrowserDirTree,
  sortEntries,
  parseYamlFlat,
  META_KEYS,
  CSV_EXTENSIONS,
  TEXT_EXTENSIONS,
} from "../../components/files/OutputBrowserTree";
import { OutputBrowserDetail } from "../../components/files/OutputBrowserDetail";

export function OutputBrowser() {
  const {
    effectiveTheme: theme,
    setPendingBenchmarkLogs,
    pendingRunPath,
    setPendingRunPath,
  } = useAppStore();
  const { projectRoot, setMode, handoff } = useRecentHandoff();
  const {
    policy: activePolicy,
    runLabel: activeRunLabel,
    logScale,
    setPolicy,
  } = useGlobalFiltersStore();

  const [runs, setRuns] = useState<OutputDir[]>([]);
  const [selectedRun, setSelectedRun] = useState<OutputDir | null>(null);
  const [compareSelection, setCompareSelection] = useState<Set<string>>(new Set());
  const [profileName, setProfileName] = useState("");
  const { profiles, saveProfile, loadProfile, deleteProfile } = useSessionProfilesStore();
  const [entries, setEntries] = useState<DirEntry[]>([]);
  const [expandedDirs, setExpandedDirs] = useState<Set<string>>(new Set());
  const [subEntries, setSubEntries] = useState<Record<string, DirEntry[]>>({});
  const [fileContent, setFileContent] = useState<string | null>(null);
  const [wsrouteBundle, setWsrouteBundle] = useState<WsrouteBundleInfo | null>(null);
  const [csvRows, setCsvRows] = useState<Array<Record<string, string | number | null>> | null>(null);
  const [csvHeaders, setCsvHeaders] = useState<string[]>([]);
  const [viewingPath, setViewingPath] = useState<string | null>(null);
  const [viewingExt, setViewingExt] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [fileLoading, setFileLoading] = useState(false);
  const [bundleExporting, setBundleExporting] = useState(false);
  const [includeArrowSidecars, setIncludeArrowSidecars] = useState(true);
  const [bundleExtracting, setBundleExtracting] = useState(false);
  const [parquetExporting, setParquetExporting] = useState(false);
  // Run metadata from pruned_config.yaml
  const [runMeta, setRunMeta] = useState<Array<{ key: string; value: string }> | null>(null);
  // KPI summary parsed from the first .jsonl found in the run directory
  const [runKpi, setRunKpi] = useState<Array<{ policy: string; overflows: number; kgkm: number; profit: number }> | null>(null);
  const [runJsonlPath, setRunJsonlPath] = useState<string | null>(null);
  const [runCheckpoints, setRunCheckpoints] = useState<DirEntry[] | null>(null);
  const [viewingCheckpoint, setViewingCheckpoint] = useState<DirEntry | null>(null);

  const outputPaths = useMemo(() => (projectRoot ? simOutputRoots(projectRoot) : null), [projectRoot]);
  const brushLogPath = selectedRun ? (runJsonlPath ?? selectedRun.path) : null;
  const derivedRunLabel = useLogPathRunLabelBrush(brushLogPath);
  const sourceRunLabel = useMemo(
    () =>
      brushLogPath
        ? portfolioRunLabel(brushLogPath, selectedRun?.name, projectRoot)
        : null,
    [brushLogPath, selectedRun?.name, projectRoot]
  );

  const refresh = useCallback(async () => {
    if (!outputPaths) return;
    setLoading(true);
    try {
      const found = await Promise.all(
        outputPaths.map((outputPath) => invoke<OutputDir[]>("list_output_dirs", { outputPath }))
      );
      setRuns(found.flat());
    } catch (err) {
      toast.error("Failed to list output directories", { description: String(err) });
    } finally {
      setLoading(false);
    }
  }, [outputPaths]);

  useEffect(() => {
    if (outputPaths) refresh();
  }, [outputPaths, refresh]);

  const selectRun = useCallback(async (run: OutputDir) => {
    handoff(run.path, "run", { storedLabel: run.name, navigate: false });
    setSelectedRun(run);
    setFileContent(null);
    setCsvRows(null);
    setViewingPath(null);
    setViewingExt("");
    setViewingCheckpoint(null);
    setExpandedDirs(new Set());
    setSubEntries({});
    setRunMeta(null);
    setRunKpi(null);
    setRunJsonlPath(null);
    setRunCheckpoints(null);
    try {
      const e = sortEntries(await invoke<DirEntry[]>("list_dir", { path: run.path }));
      setEntries(e);

      const expanded = new Set<string>();
      const subs: Record<string, DirEntry[]> = {};

      // Auto-expand hydra/ subdirectory for structured tree view
      const hydraDir = e.find((f) => f.is_dir && f.name === "hydra");
      let hydraSub: DirEntry[] = [];
      if (hydraDir) {
        expanded.add(hydraDir.path);
        hydraSub = sortEntries(await invoke<DirEntry[]>("list_dir", { path: hydraDir.path }));
        subs[hydraDir.path] = hydraSub;
      }

      // Auto-expand checkpoints/ and surface eval-ready weights (§G.14 / §G.12)
      const ckptDir = e.find((f) => f.is_dir && f.name === "checkpoints");
      if (ckptDir) {
        try {
          const ckpts = filterCheckpointEntries(
            await invoke<DirEntry[]>("list_dir", { path: ckptDir.path })
          );
          if (ckpts.length > 0) {
            expanded.add(ckptDir.path);
            subs[ckptDir.path] = ckpts;
            setRunCheckpoints(ckpts);
          }
        } catch {
          setRunCheckpoints([]);
        }
      }

      setExpandedDirs(expanded);
      setSubEntries(subs);

      // Auto-load pruned_config.yaml for metadata panel (top-level or hydra/)
      let configEntry = e.find((f) => f.name === "pruned_config.yaml" || f.name === "config.yaml");
      if (!configEntry && hydraSub.length > 0) {
        configEntry = hydraSub.find(
          (f) => f.name === "pruned_config.yaml" || f.name === "config.yaml"
        );
      }
      if (configEntry) {
        try {
          const yaml = await invoke<string>("read_text_file", { path: configEntry.path });
          const all = parseYamlFlat(yaml);
          const meta = all.filter((r) => META_KEYS.some((k) => r.key.startsWith(k)));
          setRunMeta(meta.length > 0 ? meta : all.slice(0, 12));
        } catch { /* metadata optional */ }
      }

      // Auto-parse the first .jsonl for a KPI summary card
      const jsonlPath = await findRunJsonl(run.path);
      if (jsonlPath) {
        setRunJsonlPath(jsonlPath);
        try {
          const text = await invoke<string>("read_text_file", { path: jsonlPath });
          const acc: Record<string, { overflows: number[]; kgkm: number[]; profit: number[] }> = {};
          for (const line of text.split("\n")) {
            if (!line.trim()) continue;
            try {
              const entry = JSON.parse(line) as DayLogEntry;
              if (!acc[entry.policy]) acc[entry.policy] = { overflows: [], kgkm: [], profit: [] };
              acc[entry.policy].overflows.push(entry.data.overflows ?? 0);
              acc[entry.policy].kgkm.push(entry.data["kg/km"] ?? 0);
              acc[entry.policy].profit.push(entry.data.profit ?? 0);
            } catch { /* skip malformed lines */ }
          }
          const avg = (arr: number[]) => arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
          const kpi = Object.entries(acc).map(([policy, v]) => ({
            policy,
            overflows: avg(v.overflows),
            kgkm: avg(v.kgkm),
            profit: avg(v.profit),
          })).sort((a, b) => a.overflows - b.overflows);
          if (kpi.length > 0) setRunKpi(kpi);
        } catch { /* KPI optional */ }
      }
    } catch (err) {
      toast.error("Failed to list run directory", { description: String(err) });
    }
  }, [handoff]);

  useEffect(() => {
    if (!pendingRunPath) return;
    const run = runs.find((r) => r.path === pendingRunPath);
    if (run) {
      void selectRun(run);
      setPendingRunPath(null);
      return;
    }
    if (!loading && outputPaths) {
      void refresh();
    }
  }, [pendingRunPath, runs, loading, outputPaths, setPendingRunPath, selectRun, refresh]);

  const toggleDir = useCallback(async (entry: DirEntry) => {
    setExpandedDirs((prev) => {
      const next = new Set(prev);
      if (next.has(entry.path)) next.delete(entry.path);
      else next.add(entry.path);
      return next;
    });
    if (!subEntries[entry.path]) {
      try {
        const sub = sortEntries(await invoke<DirEntry[]>("list_dir", { path: entry.path }));
        setSubEntries((prev) => ({ ...prev, [entry.path]: sub }));
      } catch {}
    }
  }, [subEntries]);

  const draggingBundle = useLayoutStore((s) => s.fileDropDragging);

  const openFile = useCallback(async (entry: DirEntry) => {
    if (entry.is_dir) {
      toggleDir(entry);
      return;
    }
    setViewingPath(entry.path);
    setViewingExt(entry.extension);
    setFileContent(null);
    setCsvRows(null);
    setWsrouteBundle(null);
    setViewingCheckpoint(isCheckpointEntry(entry) ? entry : null);
    setFileLoading(true);

    try {
      const recentKind = recentKindFromPath(entry.path);
      if (recentKind) {
        handoff(entry.path, recentKind, { navigate: false });
      }

      if (isCheckpointEntry(entry)) {
        return;
      }
      if (entry.extension === "wsroute") {
        const info = await invoke<WsrouteBundleInfo>("inspect_wsroute_bundle", {
          path: entry.path,
        });
        setWsrouteBundle(info);
      } else if (CSV_EXTENSIONS.has(entry.extension)) {
        const csvFile = await invoke<{ headers: string[]; rows: Array<Record<string, string | number | null>> }>(
          "load_csv_file",
          { path: entry.path }
        );
        setCsvHeaders(csvFile.headers);
        setCsvRows(csvFile.rows);
      } else if (TEXT_EXTENSIONS.has(entry.extension)) {
        const text = await invoke<string>("read_text_file", { path: entry.path });
        setFileContent(text);
      }
    } catch (err) {
      toast.error("Failed to open file", { description: String(err) });
    } finally {
      setFileLoading(false);
    }
  }, [toggleDir, handoff]);

  const toggleCompareRun = useCallback((runPath: string) => {
    setCompareSelection((prev) => {
      const next = new Set(prev);
      if (next.has(runPath)) next.delete(runPath);
      else next.add(runPath);
      return next;
    });
  }, []);

  const compareSelectedRuns = useCallback(async () => {
    if (compareSelection.size < 2) return;
    const refs: Array<{ path: string; label: string }> = [];
    for (const runPath of compareSelection) {
      const run = runs.find((r) => r.path === runPath);
      const jsonl = await findRunJsonl(runPath);
      if (jsonl && run) {
        const label = portfolioRunLabel(jsonl, run.name, projectRoot);
        refs.push({ path: jsonl, label });
        handoff(jsonl, "log", { storedLabel: run.name, navigate: false });
      }
    }
    if (refs.length < 2) {
      toast.error("Need at least 2 runs with .jsonl logs to compare");
      return;
    }
    setPendingBenchmarkLogs(refs);
    setMode("benchmark");
    toast.success(`Comparing ${refs.length} runs in Benchmark Analysis`);
  }, [compareSelection, runs, projectRoot, handoff, setPendingBenchmarkLogs, setMode]);

  const exportRunAsBundle = useCallback(async () => {
    if (!selectedRun) return;
    const path = (await saveDialog({
      filters: [{ name: "WSmart-Route Bundle", extensions: ["wsroute"] }],
      defaultPath: `${selectedRun.name}.wsroute`,
    })) as string | null;
    if (!path) return;

    const outputPath = path.endsWith(".wsroute") ? path : `${path}.wsroute`;
    setBundleExporting(true);
    try {
      const info = await invoke<WsrouteBundleInfo>("create_wsroute_bundle", {
        sourceDir: selectedRun.path,
        outputPath,
        includeArrow: includeArrowSidecars,
      });
      const arrowNote =
        info.arrow_sidecars != null && info.arrow_sidecars > 0
          ? ` · ${info.arrow_sidecars} Arrow sidecars`
          : "";
      toast.success(`Exported ${info.files.length} files${arrowNote}`, {
        description: outputPath.split("/").pop(),
      });
    } catch (err) {
      toast.error("Failed to export bundle", { description: String(err) });
    } finally {
      setBundleExporting(false);
    }
  }, [selectedRun, includeArrowSidecars]);

  const extractBundleAndOpen = useCallback(async () => {
    if (!viewingPath || viewingExt !== "wsroute") return;
    const destDir = (await open({
      directory: true,
      title: "Extract bundle to…",
    })) as string | null;
    if (!destDir) return;

    setBundleExtracting(true);
    try {
      const result = await invoke<WsrouteExtractResult>("extract_wsroute_bundle", {
        path: viewingPath,
        destDir,
      });
      toast.success(`Extracted ${result.extracted_files.length} files`);
      if (result.log_path) {
        handoff(result.log_path, "log");
      } else {
        toast.info("No .jsonl log found in bundle — browse extracted files manually");
      }
    } catch (err) {
      toast.error("Failed to extract bundle", { description: String(err) });
    } finally {
      setBundleExtracting(false);
    }
  }, [viewingPath, viewingExt, handoff]);

  const pickOutputDir = useCallback(async () => {
    const path = (await open({ directory: true })) as string | null;
    if (!path) return;
    try {
      const e = await invoke<DirEntry[]>("list_dir", { path });
      const name = path.split(/[/\\]/).pop() ?? path;
      const fakeRun: OutputDir = {
        name,
        path,
        created_at: "",
        size_bytes: 0,
      };
      handoff(path, "run", { storedLabel: name, navigate: false });
      setSelectedRun(fakeRun);
      setEntries(e);
      setFileContent(null);
      setCsvRows(null);
      setRunMeta(null);
      setViewingPath(null);
      setViewingExt("");
      setViewingCheckpoint(null);
    } catch (err) {
      toast.error("Failed to open directory", { description: String(err) });
    }
  }, [handoff]);

  if (!projectRoot) {
    return (
      <div className="flex items-center justify-center h-64 text-canvas-muted text-sm">
        Set project root in settings to browse output directories.
      </div>
    );
  }

  return (
    <div className="flex gap-4 h-[calc(100vh-8rem)] relative">
      <OutputBrowserRunList
        runs={runs}
        selectedRun={selectedRun}
        selectRun={selectRun}
        compareSelection={compareSelection}
        toggleCompareRun={toggleCompareRun}
        compareSelectedRuns={compareSelectedRuns}
        refresh={refresh}
        pickOutputDir={pickOutputDir}
        loading={loading}
        projectRoot={projectRoot}
        activeRunLabel={activeRunLabel}
        profiles={profiles}
        profileName={profileName}
        setProfileName={setProfileName}
        saveProfile={saveProfile}
        loadProfile={loadProfile}
        deleteProfile={deleteProfile}
      />

      <OutputBrowserDirTree
        selectedRun={selectedRun}
        entries={entries}
        expandedDirs={expandedDirs}
        subEntries={subEntries}
        openFile={openFile}
        viewingPath={viewingPath}
        includeArrowSidecars={includeArrowSidecars}
        setIncludeArrowSidecars={setIncludeArrowSidecars}
        exportRunAsBundle={exportRunAsBundle}
        bundleExporting={bundleExporting}
        runJsonlPath={runJsonlPath}
        projectRoot={projectRoot}
        runMeta={runMeta}
        runCheckpoints={runCheckpoints}
        runKpi={runKpi}
        activePolicy={activePolicy}
        setPolicy={setPolicy}
      />

      <OutputBrowserDetail
        selectedRun={selectedRun}
        theme={theme}
        logScale={logScale}
        activePolicy={activePolicy}
        activeRunLabel={activeRunLabel}
        sourceRunLabel={sourceRunLabel}
        derivedRunLabel={derivedRunLabel}
        draggingBundle={draggingBundle}
        viewingPath={viewingPath}
        viewingExt={viewingExt}
        fileLoading={fileLoading}
        fileContent={fileContent}
        csvRows={csvRows}
        csvHeaders={csvHeaders}
        wsrouteBundle={wsrouteBundle}
        viewingCheckpoint={viewingCheckpoint}
        projectRoot={projectRoot}
        extractBundleAndOpen={extractBundleAndOpen}
        bundleExtracting={bundleExtracting}
        parquetExporting={parquetExporting}
        setParquetExporting={setParquetExporting}
      />
    </div>
  );
}
