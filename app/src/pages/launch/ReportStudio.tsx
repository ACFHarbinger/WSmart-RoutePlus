/**
 * Report Studio — analysis report & presentation deck generator (§H interim).
 *
 * Folds the archived `logic/gen` batch pipeline into the Studio as a launcher:
 * three tabs assemble the full CLI for `gen_dataset_analysis.py`,
 * `gen_simulation_analysis.py` (report or raw-output→CSV parse mode) and
 * `gen_presentation.py` (PPTX deck + speaker script + XLSX), spawn them via
 * the shared process infrastructure, and surface the generated artefact paths.
 *
 * The native §H document-authoring subsystem (ECharts rendering, Rust OOXML
 * exporters, live preview/editing) replaces this page phase by phase.
 */
import { useCallback, useMemo, useState } from "react";
import { Eye, Play, Terminal, FolderOpen, FileText, Presentation } from "lucide-react";
import { DeckPreview } from "../../components/gen/DeckPreview";
import { ReportPreview } from "../../components/gen/ReportPreview";
import type { HtmlSlide } from "../../gen/deck/htmlDeck";
import { open } from "@tauri-apps/plugin-dialog";
import { ProcessLogTail } from "../../components/monitor/process/ProcessLogTail";
import { StatusPill } from "../../components/ui/StatusPill";
import { OpenPathToolbar } from "../../components/common/OpenPathToolbar";
import { useAppStore } from "../../store/app";
import { useReportGenStore, type ReportGenTab } from "../../store/launchers";
import { useProcessStore } from "../../store/process";
import { useSpawnProcess } from "../../hooks/process/useSpawnProcess";
import type { ProcessEntry } from "../../types";

// Archived batch pipeline location (§H — retired from logic/gen)
export const GEN_SCRIPTS_DIR = "logic/gen";

const TABS: { id: ReportGenTab; label: string }[] = [
  { id: "dataset", label: "Dataset Analysis" },
  { id: "simulation", label: "Simulation Analysis" },
  { id: "presentation", label: "Presentation Deck" },
];

function isReportGenProcess(id: string): boolean {
  return id.startsWith("reportgen_");
}

function findRecentReportGenProcessId(
  processes: Record<string, ProcessEntry>
): string | null {
  const candidates = Object.entries(processes)
    .filter(([id]) => isReportGenProcess(id))
    .sort((a, b) => b[1].startTime - a[1].startTime);
  return candidates[0]?.[0] ?? null;
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-1">
      <label className="text-xs text-canvas-muted">{label}</label>
      {children}
    </div>
  );
}

function Check({
  label, checked, onChange,
}: { label: string; checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <label className="flex items-center gap-2 cursor-pointer text-sm text-gray-300">
      <input
        type="checkbox"
        className="accent-accent-primary"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
      />
      {label}
    </label>
  );
}

function Select<T extends string>({
  label, value, onChange, options,
}: {
  label: string;
  value: T;
  onChange: (v: T) => void;
  options: readonly T[];
}) {
  return (
    <Field label={label}>
      <select
        className="select-base w-36"
        value={value}
        onChange={(e) => onChange(e.target.value as T)}
      >
        {options.map((o) => (
          <option key={o} value={o}>{o}</option>
        ))}
      </select>
    </Field>
  );
}

export function ReportStudio() {
  const projectRoot = useAppStore((st) => st.projectRoot);
  const { spawn, launching } = useSpawnProcess();
  const s = useReportGenStore();
  const { tab, engine, patch } = s;

  const [liveProcessId, setLiveProcessId] = useState<string | null>(null);
  const processes = useProcessStore((p) => p.processes);
  const displayProcessId = useMemo(
    () => liveProcessId ?? findRecentReportGenProcessId(processes),
    [liveProcessId, processes]
  );
  const displayProc = displayProcessId ? processes[displayProcessId] : null;
  const isDone = displayProc != null && displayProc.status !== "running";

  // Deck preview (§H.7)
  const [previewSlides, setPreviewSlides] = useState<HtmlSlide[]>([]);
  const [previewOpen, setPreviewOpen] = useState(false);
  const [previewBuilding, setPreviewBuilding] = useState(false);

  const buildPreview = useCallback(async () => {
    if (!projectRoot) return;
    setPreviewOpen(true);
    setPreviewBuilding(true);
    try {
      const { buildHtmlDeckSlides } = await import("../../gen/deck/htmlDeck");
      const slides = await buildHtmlDeckSlides({
        projectRoot,
        figuresDir: s.presFiguresDir.trim() || "public/figures/simulation/30d",
        out: "",
        author: s.presAuthor.trim() || undefined,
        coauthors: s.presCoauthors.trim() ? s.presCoauthors.split(";").map((x) => x.trim()) : undefined,
        groups: s.presGroups.trim() ? s.presGroups.split(";").map((x) => x.trim()) : undefined,
        resultsTable: s.presResultsTable,
        resultsTableSplit: s.presResultsSplit,
        imageMode: s.presImageMode,
      });
      setPreviewSlides(slides);
    } catch (err) {
      setPreviewSlides([]);
      setNativeLog((log) => [...log, `Preview failed: ${err instanceof Error ? err.message : String(err)}`]);
    } finally {
      setPreviewBuilding(false);
    }
  }, [projectRoot, s]);

  // Native (§H in-app) engine state
  const [nativeLog, setNativeLog] = useState<string[]>([]);
  const [nativeStatus, setNativeStatus] = useState<"idle" | "running" | "completed" | "failed">("idle");
  const [nativeOutputs, setNativeOutputs] = useState<string[]>([]);

  const runNative = useCallback(async () => {
    if (!projectRoot) return;
    setNativeStatus("running");
    setNativeOutputs([]);
    const log: string[] = [];
    const progress = (msg: string) => {
      log.push(msg);
      setNativeLog([...log]);
    };
    try {
      const outputs: string[] = [];
      if (tab === "dataset") {
        const { generateDatasetReport } = await import("../../gen/report/datasetReport");
        const res = await generateDatasetReport(
          {
            projectRoot,
            theme: s.theme === "default" ? undefined : s.theme,
            npzCsv: s.dsNpzCsv.trim() || undefined,
            tdCsv: s.dsTdCsv.trim() || undefined,
            npzDir: s.dsNpzDir.trim() || undefined,
            outMd: s.dsOutMd.trim() || undefined,
            figuresDir: s.dsFiguresDir.trim() || undefined,
            force: s.dsForce,
            figuresOnly: s.dsFiguresOnly,
          },
          progress
        );
        if (res.outMd) outputs.push(res.outMd);
        if (res.outMd && s.dsHtml) {
          const { generateHtmlReport } = await import("../../gen/report/htmlReport");
          outputs.push(await generateHtmlReport(projectRoot, res.outMd, progress));
        }
        outputs.push(res.figuresDir);
      } else if (tab === "simulation") {
        const mod = await import("../../gen/report/simulationReport");
        if (s.simMode === "parse") {
          await mod.parseOutputToCsv(projectRoot, s.parseOutputDir.trim(), s.parseOutCsv.trim(), progress);
          outputs.push(s.parseOutCsv.trim());
        } else {
          const horizons = s.simHorizons
            .split("\n")
            .map((l) => l.trim())
            .filter(Boolean)
            .map((line) => {
              const [days, csv] = line.split("=");
              return { days: parseInt(days, 10), csv: csv ?? "" };
            })
            .filter((h) => Number.isFinite(h.days));
          const csvList = (v: string) =>
            v.trim() ? v.split(",").map((x) => x.trim()).filter(Boolean) : null;
          const res = await mod.generateSimulationReport(
            {
              projectRoot,
              theme: s.theme === "default" ? undefined : s.theme,
              paretoPoints: s.simParetoPoints === "default" ? undefined : s.simParetoPoints,
              horizons: horizons.length ? horizons : undefined,
              scenarios: s.simScenarios.trim()
                ? s.simScenarios.split(";").map((part) => {
                    const [city, N, dist] = part.split(":").map((x) => x.trim());
                    return { city, N: parseInt(N, 10), dist };
                  })
                : null,
              strategies: csvList(s.simStrategies),
              constructors: csvList(s.simConstructors),
              improvers: csvList(s.simImprovers),
              acceptance: csvList(s.simAcceptance),
              outMd: s.simOutMd.trim() || undefined,
              figuresDir: s.simFiguresDir.trim() || undefined,
              force: s.simForce,
              figuresOnly: s.simFiguresOnly,
              heatmapLabels: s.simHeatmapLabels,
            },
            progress
          );
          if (res.outMd) outputs.push(res.outMd);
          if (res.outMd && s.simHtml) {
            const { generateHtmlReport } = await import("../../gen/report/htmlReport");
            outputs.push(await generateHtmlReport(projectRoot, res.outMd, progress));
          }
          outputs.push(...res.figuresDirs);
        }
      } else {
        const { generatePresentation } = await import("../../gen/deck/deckBuilder");
        const res = await generatePresentation(
          {
            projectRoot,
            figuresDir: s.presFiguresDir.trim() || "public/figures/simulation/30d",
            out: s.presOut.trim() || "assets/windows/wsmart_route_results.pptx",
            author: s.presAuthor.trim() || undefined,
            coauthors: s.presCoauthors.trim() ? s.presCoauthors.split(";").map((x) => x.trim()) : undefined,
            groups: s.presGroups.trim() ? s.presGroups.split(";").map((x) => x.trim()) : undefined,
            resultsTable: s.presResultsTable,
            resultsTableSplit: s.presResultsSplit,
            speakerScript: s.presSpeakerScript,
            speakerScriptOut: s.presSpeakerOut.trim() || undefined,
            imageMode: s.presImageMode,
            excel: s.presExcel,
            html: s.presHtml,
            pdf: s.presPdf,
          },
          progress
        );
        outputs.push(...res.outputs);
      }
      setNativeOutputs(outputs);
      setNativeStatus("completed");
      progress("Done.");
    } catch (err) {
      progress(`ERROR: ${err instanceof Error ? err.message : String(err)}`);
      setNativeStatus("failed");
    }
  }, [projectRoot, tab, s]);

  const pickDir = async (field: string) => {
    const path = (await open({ directory: true })) as string | null;
    if (path) patch({ [field]: path } as never);
  };
  const pickFile = async (field: string) => {
    const path = (await open({})) as string | null;
    if (path) patch({ [field]: path } as never);
  };

  const cliArgs = useMemo(() => {
    const extra = s.extraArgs.trim().split(/\s+/).filter(Boolean);
    if (tab === "dataset") {
      const args = [`${GEN_SCRIPTS_DIR}/gen_dataset_analysis.py`];
      if (s.theme !== "default") args.push("--theme", s.theme);
      if (s.dsNpzCsv.trim()) args.push("--npz-csv", s.dsNpzCsv.trim());
      if (s.dsTdCsv.trim()) args.push("--td-csv", s.dsTdCsv.trim());
      if (s.dsNpzDir.trim()) args.push("--npz-dir", s.dsNpzDir.trim());
      if (s.dsOutMd.trim()) args.push("--out-md", s.dsOutMd.trim());
      if (s.dsFiguresDir.trim()) args.push("--figures-dir", s.dsFiguresDir.trim());
      if (s.dsForce) args.push("--force");
      if (s.dsFiguresOnly) args.push("--figures-only");
      return [...args, ...extra];
    }
    if (tab === "simulation") {
      const args = [`${GEN_SCRIPTS_DIR}/gen_simulation_analysis.py`];
      if (s.simMode === "parse") {
        args.push("--parse-output");
        if (s.parseOutputDir.trim()) args.push("--output-dir", s.parseOutputDir.trim());
        if (s.parseOutCsv.trim()) args.push("--out-csv", s.parseOutCsv.trim());
        return [...args, ...extra];
      }
      if (s.theme !== "default") args.push("--theme", s.theme);
      if (s.simFontsize > 0) args.push("--fontsize", String(s.simFontsize));
      if (s.simParetoPoints !== "default") args.push("--pareto-points", s.simParetoPoints);
      for (const line of s.simHorizons.split("\n").map((l) => l.trim()).filter(Boolean)) {
        args.push("--horizon", line);
      }
      if (s.simScenarios.trim()) args.push("--scenarios", s.simScenarios.trim());
      if (s.simStrategies.trim()) args.push("--strategies", s.simStrategies.trim());
      if (s.simConstructors.trim()) args.push("--constructors", s.simConstructors.trim());
      if (s.simImprovers.trim()) args.push("--improvers", s.simImprovers.trim());
      if (s.simAcceptance.trim()) args.push("--acceptance", s.simAcceptance.trim());
      if (s.simOutMd.trim()) args.push("--out-md", s.simOutMd.trim());
      if (s.simFiguresDir.trim()) args.push("--figures-dir", s.simFiguresDir.trim());
      if (s.simMapMode !== "street") args.push("--map-mode", s.simMapMode);
      if (s.simHeatmapLabels !== "both") args.push("--scenario-heatmap-labels", s.simHeatmapLabels);
      if (s.simForce) args.push("--force");
      if (s.simFiguresOnly) args.push("--figures-only");
      return [...args, ...extra];
    }
    const args = [`${GEN_SCRIPTS_DIR}/gen_presentation.py`];
    if (s.presFiguresDir.trim()) args.push("--figures-dir", s.presFiguresDir.trim());
    if (s.presOut.trim()) args.push("--out", s.presOut.trim());
    if (s.presAuthor.trim()) args.push("--author", s.presAuthor.trim());
    if (s.presCoauthors.trim()) args.push("--coauthors", s.presCoauthors.trim());
    if (s.presGroups.trim()) args.push("--groups", s.presGroups.trim());
    args.push("--results-table", s.presResultsTable);
    if (s.presResultsSplit !== "none") args.push("--results-table-split", s.presResultsSplit);
    if (s.presSpeakerScript) {
      args.push("--speaker-script");
      if (s.presSpeakerOut.trim()) args.push("--speaker-script-out", s.presSpeakerOut.trim());
    }
    if (s.presImageMode !== "native") args.push("--image-mode", s.presImageMode);
    if (s.presExcel) args.push("--excel");
    return [...args, ...extra];
  }, [tab, s]);

  const commandPreview = `python ${cliArgs
    .map((a) => (/\s/.test(a) ? `"${a}"` : a))
    .join(" \\\n  ")}`;

  const launch = useCallback(async () => {
    if (!projectRoot) return;
    const procId = `reportgen_${tab}_${Date.now()}`;
    setLiveProcessId(procId);
    await spawn({ id: procId, pythonArgs: cliArgs, workingDir: projectRoot });
  }, [projectRoot, tab, cliArgs, spawn]);

  // Artefact paths to surface after a successful run
  const outputPaths = useMemo(() => {
    if (tab === "dataset") {
      return [s.dsOutMd.trim() || "public/dataset_analysis.md"];
    }
    if (tab === "simulation") {
      if (s.simMode === "parse") return [s.parseOutCsv.trim()].filter(Boolean);
      return [s.simOutMd.trim() || "public/simulation_analysis.md"];
    }
    const out = s.presOut.trim() || "assets/windows/wsmart_route_results.pptx";
    const paths = [out];
    if (s.presSpeakerScript) {
      paths.push(s.presSpeakerOut.trim() || out.replace(/\.pptx$/i, ".docx"));
    }
    if (s.presExcel) paths.push(out.replace(/\.pptx$/i, ".xlsx"));
    return paths;
  }, [tab, s]);

  return (
    <div className="space-y-4 max-w-2xl">
      {/* Tab selector */}
      <div className="card space-y-3">
        <div className="flex items-center gap-2">
          <Presentation size={16} className="text-accent-primary" />
          <h2 className="text-sm font-semibold text-gray-200">Report Studio</h2>
        </div>
        <p className="text-[11px] text-canvas-muted">
          Generates the dataset / simulation analysis reports (markdown + figures) and the
          results presentation deck (PPTX + speaker DOCX + XLSX).{" "}
          <strong>Native</strong> renders everything in-app (ECharts figures, MathJax
          equations, pptxgenjs deck — §H); <strong>Legacy</strong> spawns the archived{" "}
          <code>logic/gen</code> Python scripts.
        </p>
        <div className="flex gap-2 flex-wrap items-center">
          {TABS.map((t) => (
            <button
              key={t.id}
              onClick={() => patch({ tab: t.id })}
              className={tab === t.id ? "btn-primary text-sm py-1.5 px-4" : "btn-ghost text-sm py-1.5 px-4"}
            >
              {t.label}
            </button>
          ))}
          <span className="flex-1" />
          <div className="flex gap-1 rounded-lg bg-canvas-bg p-0.5">
            {(["native", "legacy"] as const).map((e) => (
              <button
                key={e}
                onClick={() => patch({ engine: e })}
                className={
                  engine === e
                    ? "btn-primary text-xs py-1 px-3"
                    : "btn-ghost text-xs py-1 px-3 text-canvas-muted"
                }
              >
                {e === "native" ? "Native" : "Legacy"}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Shared theme */}
      {(tab === "dataset" || (tab === "simulation" && s.simMode === "report")) && (
        <div className="card space-y-3">
          <h2 className="text-sm font-semibold text-gray-200">Chart Theme</h2>
          <div className="flex flex-wrap gap-4 items-end">
            <Select
              label="Theme"
              value={s.theme}
              onChange={(v) => patch({ theme: v })}
              options={["default", "dark", "light"] as const}
            />
            {tab === "simulation" && (
              <>
                <Field label="Base Fontsize (0 = default)">
                  <input
                    type="number"
                    className="input-base font-mono text-sm w-24"
                    value={s.simFontsize}
                    min={0}
                    step={0.5}
                    onChange={(e) => patch({ simFontsize: Number(e.target.value) })}
                  />
                </Field>
                <Select
                  label="Pareto Points"
                  value={s.simParetoPoints}
                  onChange={(v) => patch({ simParetoPoints: v })}
                  options={["default", "all", "front"] as const}
                />
              </>
            )}
          </div>
        </div>
      )}

      {tab === "dataset" && (
        <div className="card space-y-4">
          <h2 className="text-sm font-semibold text-gray-200">Dataset Analysis Report</h2>
          <p className="text-[11px] text-canvas-muted">
            NPZ/TD statistics report with extended stats, violin/box/KDE figures. Empty
            fields use the defaults from <code>dataset_analysis_config.json</code>.
          </p>
          <div className="grid grid-cols-2 gap-3">
            <Field label="NPZ Stats CSV">
              <div className="flex gap-2">
                <input type="text" className="input-base font-mono text-xs flex-1" value={s.dsNpzCsv} onChange={(e) => patch({ dsNpzCsv: e.target.value })} placeholder="config default" />
                <button onClick={() => pickFile("dsNpzCsv")} className="btn-ghost text-xs"><FolderOpen size={12} /></button>
              </div>
            </Field>
            <Field label="TD Stats CSV">
              <div className="flex gap-2">
                <input type="text" className="input-base font-mono text-xs flex-1" value={s.dsTdCsv} onChange={(e) => patch({ dsTdCsv: e.target.value })} placeholder="config default" />
                <button onClick={() => pickFile("dsTdCsv")} className="btn-ghost text-xs"><FolderOpen size={12} /></button>
              </div>
            </Field>
            <Field label="Raw NPZ Directory">
              <div className="flex gap-2">
                <input type="text" className="input-base font-mono text-xs flex-1" value={s.dsNpzDir} onChange={(e) => patch({ dsNpzDir: e.target.value })} placeholder="config default" />
                <button onClick={() => pickDir("dsNpzDir")} className="btn-ghost text-xs"><FolderOpen size={12} /></button>
              </div>
            </Field>
            <Field label="Output Markdown">
              <input type="text" className="input-base font-mono text-xs" value={s.dsOutMd} onChange={(e) => patch({ dsOutMd: e.target.value })} placeholder="public/dataset_analysis.md" />
            </Field>
            <Field label="Figures Directory">
              <input type="text" className="input-base font-mono text-xs" value={s.dsFiguresDir} onChange={(e) => patch({ dsFiguresDir: e.target.value })} placeholder="config default" />
            </Field>
          </div>
          <div className="flex flex-wrap gap-4">
            <Check label="Force overwrite" checked={s.dsForce} onChange={(v) => patch({ dsForce: v })} />
            <Check label="Figures only (skip markdown)" checked={s.dsFiguresOnly} onChange={(v) => patch({ dsFiguresOnly: v })} />
            {engine === "native" && (
              <Check label="HTML report" checked={s.dsHtml} onChange={(v) => patch({ dsHtml: v })} />
            )}
          </div>
        </div>
      )}

      {tab === "simulation" && (
        <div className="card space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-semibold text-gray-200">Simulation Analysis</h2>
            <div className="flex gap-2">
              {(["report", "parse"] as const).map((m) => (
                <button
                  key={m}
                  onClick={() => patch({ simMode: m })}
                  className={s.simMode === m ? "btn-primary text-xs py-1 px-3" : "btn-ghost text-xs py-1 px-3"}
                >
                  {m === "report" ? "Report" : "Parse Output → CSV"}
                </button>
              ))}
            </div>
          </div>
          {s.simMode === "parse" ? (
            <>
              <p className="text-[11px] text-canvas-muted">
                Parses a raw <code>assets/output/&lt;horizon&gt;days/</code> tree into the
                summary CSV schema (policy metadata decoded from filenames).
              </p>
              <Field label="Raw Output Tree">
                <div className="flex gap-2">
                  <input type="text" className="input-base font-mono text-xs flex-1" value={s.parseOutputDir} onChange={(e) => patch({ parseOutputDir: e.target.value })} />
                  <button onClick={() => pickDir("parseOutputDir")} className="btn-ghost text-xs"><FolderOpen size={12} /></button>
                </div>
              </Field>
              <Field label="Destination CSV">
                <input type="text" className="input-base font-mono text-xs" value={s.parseOutCsv} onChange={(e) => patch({ parseOutCsv: e.target.value })} />
              </Field>
            </>
          ) : (
            <>
              <p className="text-[11px] text-canvas-muted">
                Multi-horizon report: Pareto fronts, KPI bars, heatmaps, bubble/radar charts,
                bin maps, hierarchical results table. Scenarios/policies are auto-detected;
                filters below narrow them. Empty fields use{" "}
                <code>simulation_analysis_config.json</code> defaults.
              </p>
              <Field label="Horizons (one DAYS=CSV per line, empty = config)">
                <textarea
                  className="input-base w-full font-mono text-xs h-14 resize-y"
                  value={s.simHorizons}
                  onChange={(e) => patch({ simHorizons: e.target.value })}
                  placeholder={"30=public/global/simulation/simulation_summary.csv\n90=public/global/simulation/simulation_summary_90d.csv"}
                  spellCheck={false}
                />
              </Field>
              <div className="grid grid-cols-2 gap-3">
                <Field label="Scenarios (City:N:Dist;…)">
                  <input type="text" className="input-base font-mono text-xs" value={s.simScenarios} onChange={(e) => patch({ simScenarios: e.target.value })} placeholder="Rio Maior:100:Gamma-3" />
                </Field>
                <Field label="Strategies (comma-sep)">
                  <input type="text" className="input-base font-mono text-xs" value={s.simStrategies} onChange={(e) => patch({ simStrategies: e.target.value })} placeholder="all" />
                </Field>
                <Field label="Constructors (comma-sep)">
                  <input type="text" className="input-base font-mono text-xs" value={s.simConstructors} onChange={(e) => patch({ simConstructors: e.target.value })} placeholder="all" />
                </Field>
                <Field label="Improvers (comma-sep)">
                  <input type="text" className="input-base font-mono text-xs" value={s.simImprovers} onChange={(e) => patch({ simImprovers: e.target.value })} placeholder="all" />
                </Field>
                <Field label="Acceptance (comma-sep)">
                  <input type="text" className="input-base font-mono text-xs" value={s.simAcceptance} onChange={(e) => patch({ simAcceptance: e.target.value })} placeholder="all" />
                </Field>
                <Field label="Output Markdown">
                  <input type="text" className="input-base font-mono text-xs" value={s.simOutMd} onChange={(e) => patch({ simOutMd: e.target.value })} placeholder="public/simulation_analysis.md" />
                </Field>
                <Field label="Figures Directory">
                  <input type="text" className="input-base font-mono text-xs" value={s.simFiguresDir} onChange={(e) => patch({ simFiguresDir: e.target.value })} placeholder="config default" />
                </Field>
              </div>
              <div className="flex flex-wrap gap-4 items-end">
                <Select label="Bin Map Mode" value={s.simMapMode} onChange={(v) => patch({ simMapMode: v })} options={["street", "scatter"] as const} />
                <Select label="Heatmap Labels" value={s.simHeatmapLabels} onChange={(v) => patch({ simHeatmapLabels: v })} options={["both", "show", "hide"] as const} />
              </div>
              <div className="flex flex-wrap gap-4">
                <Check label="Force overwrite" checked={s.simForce} onChange={(v) => patch({ simForce: v })} />
                <Check label="Figures only (skip markdown)" checked={s.simFiguresOnly} onChange={(v) => patch({ simFiguresOnly: v })} />
                {engine === "native" && (
                  <Check label="HTML report" checked={s.simHtml} onChange={(v) => patch({ simHtml: v })} />
                )}
              </div>
            </>
          )}
        </div>
      )}

      {tab === "presentation" && (
        <div className="card space-y-4">
          <h2 className="text-sm font-semibold text-gray-200">Results Presentation Deck</h2>
          <p className="text-[11px] text-canvas-muted">
            21-slide PPTX with native editable OMML equations, diagrams, result figures and
            the hierarchical results table; optional speaker-script DOCX and XLSX workbook.
          </p>
          <div className="grid grid-cols-2 gap-3">
            <Field label="Figures Directory">
              <div className="flex gap-2">
                <input type="text" className="input-base font-mono text-xs flex-1" value={s.presFiguresDir} onChange={(e) => patch({ presFiguresDir: e.target.value })} />
                <button onClick={() => pickDir("presFiguresDir")} className="btn-ghost text-xs"><FolderOpen size={12} /></button>
              </div>
            </Field>
            <Field label="Output PPTX">
              <input type="text" className="input-base font-mono text-xs" value={s.presOut} onChange={(e) => patch({ presOut: e.target.value })} />
            </Field>
            <Field label="Author">
              <input type="text" className="input-base text-xs" value={s.presAuthor} onChange={(e) => patch({ presAuthor: e.target.value })} placeholder="content default" />
            </Field>
            <Field label="Co-authors (semicolon-sep)">
              <input type="text" className="input-base text-xs" value={s.presCoauthors} onChange={(e) => patch({ presCoauthors: e.target.value })} />
            </Field>
            <Field label="Research Groups (semicolon-sep)">
              <input type="text" className="input-base text-xs" value={s.presGroups} onChange={(e) => patch({ presGroups: e.target.value })} />
            </Field>
          </div>
          <div className="flex flex-wrap gap-4 items-end">
            <Select label="Results Table" value={s.presResultsTable} onChange={(v) => patch({ presResultsTable: v })} options={["30d", "90d", "all", "none"] as const} />
            <Select label="Table Split" value={s.presResultsSplit} onChange={(v) => patch({ presResultsSplit: v })} options={["none", "strategy", "constructor", "improver"] as const} />
            <Select label="Diagram Images" value={s.presImageMode} onChange={(v) => patch({ presImageMode: v })} options={["native", "fetch"] as const} />
          </div>
          <div className="flex flex-wrap gap-4">
            <Check label="Speaker script (.docx)" checked={s.presSpeakerScript} onChange={(v) => patch({ presSpeakerScript: v })} />
            <Check label="Excel results workbook" checked={s.presExcel} onChange={(v) => patch({ presExcel: v })} />
            {engine === "native" && (
              <>
                <Check label="HTML slideshow" checked={s.presHtml} onChange={(v) => patch({ presHtml: v })} />
                <Check label="PDF deck" checked={s.presPdf} onChange={(v) => patch({ presPdf: v })} />
              </>
            )}
          </div>
          {s.presSpeakerScript && (
            <Field label="Speaker Script Output (empty = beside PPTX)">
              <input type="text" className="input-base font-mono text-xs" value={s.presSpeakerOut} onChange={(e) => patch({ presSpeakerOut: e.target.value })} />
            </Field>
          )}
        </div>
      )}

      {/* Extra CLI args (legacy engine only) */}
      {engine === "legacy" && (
        <div className="card space-y-2">
          <Field label="Extra CLI Arguments (advanced, space-separated)">
            <input
              type="text"
              className="input-base font-mono text-xs"
              value={s.extraArgs}
              onChange={(e) => patch({ extraArgs: e.target.value })}
              placeholder="--private-dir public/private/simulation"
              spellCheck={false}
            />
          </Field>
        </div>
      )}

      {/* Command preview (legacy engine only) */}
      {engine === "legacy" && (
        <div className="card space-y-2">
          <div className="flex items-center gap-2 text-xs text-canvas-muted">
            <Terminal size={12} />
            Command preview
          </div>
          <pre className="font-mono text-xs text-accent-secondary whitespace-pre-wrap bg-canvas-bg rounded-lg p-3">
            {commandPreview}
          </pre>
        </div>
      )}

      {/* Launch */}
      <div className="flex items-center gap-3">
        {!projectRoot && (
          <p className="text-xs text-accent-warning">Configure Project Root in Settings first.</p>
        )}
        <button
          onClick={engine === "native" ? runNative : launch}
          disabled={launching || nativeStatus === "running" || !projectRoot}
          className="btn-primary flex items-center gap-2"
        >
          <Play size={14} />
          {launching || nativeStatus === "running"
            ? "Generating…"
            : engine === "native"
              ? "Generate (native)"
              : "Generate (script)"}
        </button>
        {engine === "native" && tab === "presentation" && (
          <button
            onClick={buildPreview}
            disabled={previewBuilding || !projectRoot}
            className="btn-ghost flex items-center gap-2"
          >
            <Eye size={14} />
            {previewBuilding ? "Building preview…" : "Preview deck"}
          </button>
        )}
      </div>

      {/* Deck preview canvas (§H.7) */}
      {engine === "native" && tab === "presentation" && previewOpen && (
        <DeckPreview slides={previewSlides} onRebuild={buildPreview} building={previewBuilding} />
      )}

      {/* Native engine output */}
      {engine === "native" && nativeStatus !== "idle" && (
        <div className="card space-y-2">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-semibold text-gray-200 flex items-center gap-2">
              <FileText size={14} />
              Native Generation
            </h2>
            <StatusPill status={nativeStatus === "running" ? "running" : nativeStatus} />
          </div>
          <ProcessLogTail
            logLines={nativeLog}
            maxLines={30}
            waiting={nativeStatus === "running"}
          />
          {nativeStatus === "completed" && nativeOutputs.length > 0 && (
            <div className="space-y-1 pt-1">
              <p className="text-[10px] uppercase tracking-widest text-canvas-muted">Artefacts</p>
              {nativeOutputs.map((p) => (
                <OpenPathToolbar
                  key={p}
                  path={p}
                  projectRoot={projectRoot}
                  handoff={false}
                  chipClassName="max-w-full"
                />
              ))}
            </div>
          )}
          {nativeStatus === "completed" && projectRoot && nativeOutputs.some((p) => p.endsWith(".md")) && (
            <ReportPreview
              projectRoot={projectRoot}
              mdRel={nativeOutputs.find((p) => p.endsWith(".md"))!}
              onLog={(m) => setNativeLog((l) => [...l, m])}
            />
          )}
        </div>
      )}

      {/* Legacy process output */}
      {engine === "legacy" && displayProc && (
        <div className="card space-y-2">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-semibold text-gray-200 flex items-center gap-2">
              <FileText size={14} />
              Generation Output
            </h2>
            <StatusPill status={displayProc.status} />
          </div>
          <ProcessLogTail
            logLines={displayProc.logLines}
            maxLines={30}
            waiting={displayProc.status === "running"}
          />
          {isDone && displayProc.status === "completed" && (
            <div className="space-y-1 pt-1">
              <p className="text-[10px] uppercase tracking-widest text-canvas-muted">Artefacts</p>
              {outputPaths.map((p) => (
                <OpenPathToolbar
                  key={p}
                  path={p}
                  projectRoot={projectRoot}
                  handoff={false}
                  chipClassName="max-w-full"
                />
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
