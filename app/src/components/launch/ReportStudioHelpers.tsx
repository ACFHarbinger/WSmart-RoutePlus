import { FileText } from "lucide-react";
import { ProcessLogTail } from "../monitor/process/ProcessLogTail";
import { StatusPill } from "../ui/StatusPill";
import { OpenPathToolbar } from "../common/OpenPathToolbar";
import { ReportPreview } from "../gen/ReportPreview";
import type { ReportGenState } from "../../store/launchers";
import type { ProcessEntry } from "../../types";

export const GEN_SCRIPTS_DIR = "logic/gen";

type ReportGenStoreState = ReportGenState;

export function buildReportStudioCliArgs(
  tab: ReportGenStoreState["tab"],
  s: ReportGenStoreState
): string[] {
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
}

export function resolveReportStudioOutputPaths(
  tab: ReportGenStoreState["tab"],
  s: ReportGenStoreState
): string[] {
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
}

export function ReportStudioOutputSection({
  engine,
  nativeStatus,
  nativeLog,
  setNativeLog,
  nativeOutputs,
  projectRoot,
  displayProc,
  isDone,
  outputPaths,
}: {
  engine: "native" | "legacy";
  nativeStatus: "idle" | "running" | "completed" | "failed";
  nativeLog: string[];
  setNativeLog: React.Dispatch<React.SetStateAction<string[]>>;
  nativeOutputs: string[];
  projectRoot: string | null;
  displayProc: ProcessEntry | null;
  isDone: boolean;
  outputPaths: string[];
}) {
  return (
    <>
      {engine === "native" && nativeStatus !== "idle" && (
        <div className="card space-y-2">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-semibold text-gray-200 flex items-center gap-2">
              <FileText size={14} />
              Native Generation
            </h2>
            <StatusPill
              status={nativeStatus === "running" ? "running" : nativeStatus}
            />
          </div>
          <ProcessLogTail
            logLines={nativeLog}
            maxLines={30}
            waiting={nativeStatus === "running"}
          />
          {nativeStatus === "completed" && nativeOutputs.length > 0 && (
            <div className="space-y-1 pt-1">
              <p className="text-[10px] uppercase tracking-widest text-canvas-muted">
                Artefacts
              </p>
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
          {nativeStatus === "completed" &&
            projectRoot &&
            nativeOutputs.some((p) => p.endsWith(".md")) && (
              <ReportPreview
                projectRoot={projectRoot}
                mdRel={nativeOutputs.find((p) => p.endsWith(".md"))!}
                onLog={(m) => setNativeLog((l) => [...l, m])}
              />
            )}
        </div>
      )}

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
              <p className="text-[10px] uppercase tracking-widest text-canvas-muted">
                Artefacts
              </p>
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
    </>
  );
}
