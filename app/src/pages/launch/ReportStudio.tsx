/**
 * Report Studio — analysis report & presentation deck generator (§H interim).
 */
import { useCallback, useEffect, useMemo, useState } from "react";
import { Eye, Play, Terminal, Presentation, CheckCircle2 } from "lucide-react";
import { DeckPreview } from "../../components/gen/DeckPreview";
import type { HtmlSlide } from "../../gen/deck/htmlDeck";
import { open } from "@tauri-apps/plugin-dialog";
import {
  Field,
  ReportStudioTabForms,
} from "../../components/launch/ReportStudioForms";
import {
  buildReportStudioCliArgs,
  GEN_SCRIPTS_DIR,
  ReportStudioOutputSection,
  resolveReportStudioOutputPaths,
} from "../../components/launch/ReportStudioHelpers";
import { useAppStore } from "../../store/app";
import { useReportGenStore, type ReportGenTab } from "../../store/launchers";
import { useLoadedRunStore } from "../../store/loadedRun";
import { useProcessStore } from "../../store/process";
import { useSpawnProcess } from "../../hooks/process/useSpawnProcess";
import type { ProcessEntry } from "../../types";

export { GEN_SCRIPTS_DIR };

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

export function ReportStudio() {
  const projectRoot = useAppStore((st) => st.projectRoot);
  const { spawn, launching } = useSpawnProcess();
  const s = useReportGenStore();
  const { tab, engine, patch } = s;
  const loadedPath = useLoadedRunStore((st) => st.path);
  const loadedLabel = useLoadedRunStore((st) => st.label);
  const loadedArea = useLoadedRunStore((st) => st.area);
  const loadedNumLoc = useLoadedRunStore((st) => st.numLoc);
  const loadedDistribution = useLoadedRunStore((st) => st.distribution);

  // Auto-populate scenario filter / output directory from useLoadedRunStore when present
  useEffect(() => {
    if (!loadedPath) return;
    if (
      loadedArea &&
      loadedNumLoc &&
      loadedDistribution &&
      !s.simScenarios.trim()
    ) {
      patch({
        simScenarios: `${loadedArea}:${loadedNumLoc}:${loadedDistribution}`,
      });
    }
  }, [loadedPath, loadedArea, loadedNumLoc, loadedDistribution, s.simScenarios, patch]);

  const [liveProcessId, setLiveProcessId] = useState<string | null>(null);
  const processes = useProcessStore((p) => p.processes);
  const displayProcessId = useMemo(
    () => liveProcessId ?? findRecentReportGenProcessId(processes),
    [liveProcessId, processes]
  );
  const displayProc = displayProcessId ? processes[displayProcessId] : null;
  const isDone = displayProc != null && displayProc.status !== "running";

  const [previewSlides, setPreviewSlides] = useState<HtmlSlide[]>([]);
  const [previewOpen, setPreviewOpen] = useState(false);
  const [previewBuilding, setPreviewBuilding] = useState(false);
  const [nativeLog, setNativeLog] = useState<string[]>([]);
  const [nativeStatus, setNativeStatus] = useState<
    "idle" | "running" | "completed" | "failed"
  >("idle");
  const [nativeOutputs, setNativeOutputs] = useState<string[]>([]);

  const buildPreview = useCallback(async () => {
    if (!projectRoot) return;
    setPreviewOpen(true);
    setPreviewBuilding(true);
    try {
      const { buildHtmlDeckSlides } = await import("../../gen/deck/htmlDeck");
      const slides = await buildHtmlDeckSlides({
        projectRoot,
        figuresDir:
          s.presFiguresDir.trim() || "public/figures/simulation/30d",
        out: "",
        author: s.presAuthor.trim() || undefined,
        coauthors: s.presCoauthors.trim()
          ? s.presCoauthors.split(";").map((x) => x.trim())
          : undefined,
        groups: s.presGroups.trim()
          ? s.presGroups.split(";").map((x) => x.trim())
          : undefined,
        resultsTable: s.presResultsTable,
        resultsTableSplit: s.presResultsSplit,
        imageMode: s.presImageMode,
      });
      setPreviewSlides(slides);
    } catch (err) {
      setPreviewSlides([]);
      setNativeLog((log) => [
        ...log,
        `Preview failed: ${err instanceof Error ? err.message : String(err)}`,
      ]);
    } finally {
      setPreviewBuilding(false);
    }
  }, [projectRoot, s]);

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
        const { generateDatasetReport } = await import(
          "../../gen/report/datasetReport"
        );
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
          const { generateHtmlReport } = await import(
            "../../gen/report/htmlReport"
          );
          outputs.push(
            await generateHtmlReport(projectRoot, res.outMd, progress)
          );
        }
        outputs.push(res.figuresDir);
      } else if (tab === "simulation") {
        const mod = await import("../../gen/report/simulationReport");
        if (s.simMode === "parse") {
          await mod.parseOutputToCsv(
            projectRoot,
            s.parseOutputDir.trim(),
            s.parseOutCsv.trim(),
            progress
          );
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
            v.trim()
              ? v
                  .split(",")
                  .map((x) => x.trim())
                  .filter(Boolean)
              : null;
          const res = await mod.generateSimulationReport(
            {
              projectRoot,
              theme: s.theme === "default" ? undefined : s.theme,
              paretoPoints:
                s.simParetoPoints === "default" ? undefined : s.simParetoPoints,
              horizons: horizons.length ? horizons : undefined,
              scenarios: s.simScenarios.trim()
                ? s.simScenarios.split(";").map((part) => {
                    const [city, N, dist] = part
                      .split(":")
                      .map((x) => x.trim());
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
            const { generateHtmlReport } = await import(
              "../../gen/report/htmlReport"
            );
            outputs.push(
              await generateHtmlReport(projectRoot, res.outMd, progress)
            );
          }
          outputs.push(...res.figuresDirs);
        }
      } else {
        const { generatePresentation } = await import(
          "../../gen/deck/deckBuilder"
        );
        const res = await generatePresentation(
          {
            projectRoot,
            figuresDir:
              s.presFiguresDir.trim() || "public/figures/simulation/30d",
            out:
              s.presOut.trim() || "assets/windows/wsmart_route_results.pptx",
            author: s.presAuthor.trim() || undefined,
            coauthors: s.presCoauthors.trim()
              ? s.presCoauthors.split(";").map((x) => x.trim())
              : undefined,
            groups: s.presGroups.trim()
              ? s.presGroups.split(";").map((x) => x.trim())
              : undefined,
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

  const cliArgs = useMemo(() => buildReportStudioCliArgs(tab, s), [tab, s]);

  const commandPreview = `python ${cliArgs
    .map((a) => (/\s/.test(a) ? `"${a}"` : a))
    .join(" \\\n  ")}`;

  const launch = useCallback(async () => {
    if (!projectRoot) return;
    const procId = `reportgen_${tab}_${Date.now()}`;
    setLiveProcessId(procId);
    await spawn({ id: procId, pythonArgs: cliArgs, workingDir: projectRoot });
  }, [projectRoot, tab, cliArgs, spawn]);

  const outputPaths = useMemo(
    () => resolveReportStudioOutputPaths(tab, s),
    [tab, s]
  );

  return (
    <div className="space-y-4 max-w-2xl">
      <div className="card space-y-3">
        <div className="flex items-center justify-between flex-wrap gap-2">
          <div className="flex items-center gap-2">
            <Presentation size={16} className="text-accent-primary" />
            <h2 className="text-sm font-semibold text-gray-200">Report Studio</h2>
          </div>
          {loadedLabel && (
            <span className="inline-flex items-center gap-1.5 text-[11px] font-mono px-2.5 py-0.5 rounded-full bg-emerald-500/15 text-emerald-300 border border-emerald-500/30">
              <CheckCircle2 size={11} />
              Bound to {loadedLabel}
            </span>
          )}
        </div>
        <p className="text-[11px] text-canvas-muted">
          Generates the dataset / simulation analysis reports (markdown +
          figures) and the results presentation deck (PPTX + speaker DOCX +
          XLSX). <strong>Native</strong> renders everything in-app (§H);{" "}
          <strong>Legacy</strong> spawns the archived <code>logic/gen</code>{" "}
          Python scripts.
        </p>
        <div className="flex gap-2 flex-wrap items-center">
          {TABS.map((t) => (
            <button
              key={t.id}
              onClick={() => patch({ tab: t.id })}
              className={
                tab === t.id
                  ? "btn-primary text-sm py-1.5 px-4"
                  : "btn-ghost text-sm py-1.5 px-4"
              }
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

      <ReportStudioTabForms
        s={s}
        pickFile={(f) => void pickFile(f)}
        pickDir={(f) => void pickDir(f)}
      />

      {engine === "legacy" && (
        <>
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

          <div className="card space-y-2">
            <div className="flex items-center gap-2 text-xs text-canvas-muted">
              <Terminal size={12} />
              Command preview
            </div>
            <pre className="font-mono text-xs text-accent-secondary whitespace-pre-wrap bg-canvas-bg rounded-lg p-3">
              {commandPreview}
            </pre>
          </div>
        </>
      )}

      <div className="flex items-center gap-3">
        {!projectRoot && (
          <p className="text-xs text-accent-warning">
            Configure Project Root in Settings first.
          </p>
        )}
        <button
          onClick={() => void (engine === "native" ? runNative() : launch())}
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
            onClick={() => void buildPreview()}
            disabled={previewBuilding || !projectRoot}
            className="btn-ghost flex items-center gap-2"
          >
            <Eye size={14} />
            {previewBuilding ? "Building preview…" : "Preview deck"}
          </button>
        )}
      </div>

      {engine === "native" && tab === "presentation" && previewOpen && (
        <DeckPreview
          slides={previewSlides}
          onRebuild={() => void buildPreview()}
          building={previewBuilding}
        />
      )}

      <ReportStudioOutputSection
        engine={engine}
        nativeStatus={nativeStatus}
        nativeLog={nativeLog}
        setNativeLog={setNativeLog}
        nativeOutputs={nativeOutputs}
        projectRoot={projectRoot}
        displayProc={displayProc}
        isDone={isDone}
        outputPaths={outputPaths}
      />
    </div>
  );
}
