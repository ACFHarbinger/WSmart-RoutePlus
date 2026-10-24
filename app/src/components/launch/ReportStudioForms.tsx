import { FolderOpen } from "lucide-react";
import type { ReportGenState } from "../../store/launchers";

type ReportGenStoreState = ReportGenState;

export function Field({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div className="flex flex-col gap-1">
      <label className="text-xs text-canvas-muted">{label}</label>
      {children}
    </div>
  );
}

export function Check({
  label,
  checked,
  onChange,
}: {
  label: string;
  checked: boolean;
  onChange: (v: boolean) => void;
}) {
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

export function Select<T extends string>({
  label,
  value,
  onChange,
  options,
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
          <option key={o} value={o}>
            {o}
          </option>
        ))}
      </select>
    </Field>
  );
}

function PathInput({
  label,
  value,
  onChange,
  onPick,
  placeholder,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  onPick?: () => void;
  placeholder?: string;
}) {
  return (
    <Field label={label}>
      <div className="flex gap-2">
        <input
          type="text"
          className="input-base font-mono text-xs flex-1"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder={placeholder}
        />
        {onPick && (
          <button onClick={onPick} className="btn-ghost text-xs">
            <FolderOpen size={12} />
          </button>
        )}
      </div>
    </Field>
  );
}

export function ReportStudioTabForms({
  s,
  pickFile,
  pickDir,
}: {
  s: ReportGenStoreState;
  pickFile: (field: string) => void;
  pickDir: (field: string) => void;
}) {
  const { tab, engine, patch } = s;

  return (
    <>
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
                    onChange={(e) =>
                      patch({ simFontsize: Number(e.target.value) })
                    }
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
          <h2 className="text-sm font-semibold text-gray-200">
            Dataset Analysis Report
          </h2>
          <p className="text-[11px] text-canvas-muted">
            NPZ/TD statistics report with extended stats, violin/box/KDE figures.
            Empty fields use the defaults from{" "}
            <code>dataset_analysis_config.json</code>.
          </p>
          <div className="grid grid-cols-2 gap-3">
            <PathInput
              label="NPZ Stats CSV"
              value={s.dsNpzCsv}
              onChange={(v) => patch({ dsNpzCsv: v })}
              onPick={() => pickFile("dsNpzCsv")}
              placeholder="config default"
            />
            <PathInput
              label="TD Stats CSV"
              value={s.dsTdCsv}
              onChange={(v) => patch({ dsTdCsv: v })}
              onPick={() => pickFile("dsTdCsv")}
              placeholder="config default"
            />
            <PathInput
              label="Raw NPZ Directory"
              value={s.dsNpzDir}
              onChange={(v) => patch({ dsNpzDir: v })}
              onPick={() => pickDir("dsNpzDir")}
              placeholder="config default"
            />
            <PathInput
              label="Output Markdown"
              value={s.dsOutMd}
              onChange={(v) => patch({ dsOutMd: v })}
              placeholder="public/dataset_analysis.md"
            />
            <PathInput
              label="Figures Directory"
              value={s.dsFiguresDir}
              onChange={(v) => patch({ dsFiguresDir: v })}
              placeholder="config default"
            />
          </div>
          <div className="flex flex-wrap gap-4">
            <Check
              label="Force overwrite"
              checked={s.dsForce}
              onChange={(v) => patch({ dsForce: v })}
            />
            <Check
              label="Figures only (skip markdown)"
              checked={s.dsFiguresOnly}
              onChange={(v) => patch({ dsFiguresOnly: v })}
            />
            {engine === "native" && (
              <Check
                label="HTML report"
                checked={s.dsHtml}
                onChange={(v) => patch({ dsHtml: v })}
              />
            )}
          </div>
        </div>
      )}

      {tab === "simulation" && (
        <div className="card space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-semibold text-gray-200">
              Simulation Analysis
            </h2>
            <div className="flex gap-2">
              {(["report", "parse"] as const).map((m) => (
                <button
                  key={m}
                  onClick={() => patch({ simMode: m })}
                  className={
                    s.simMode === m
                      ? "btn-primary text-xs py-1 px-3"
                      : "btn-ghost text-xs py-1 px-3"
                  }
                >
                  {m === "report" ? "Report" : "Parse Output → CSV"}
                </button>
              ))}
            </div>
          </div>
          {s.simMode === "parse" ? (
            <>
              <p className="text-[11px] text-canvas-muted">
                Parses a raw{" "}
                <code>outputs/simulations/&lt;horizon&gt;days/</code> tree into
                the summary CSV schema (policy metadata decoded from filenames).
              </p>
              <PathInput
                label="Raw Output Tree"
                value={s.parseOutputDir}
                onChange={(v) => patch({ parseOutputDir: v })}
                onPick={() => pickDir("parseOutputDir")}
              />
              <PathInput
                label="Destination CSV"
                value={s.parseOutCsv}
                onChange={(v) => patch({ parseOutCsv: v })}
              />
            </>
          ) : (
            <>
              <p className="text-[11px] text-canvas-muted">
                Multi-horizon report: Pareto fronts, KPI bars, heatmaps,
                bubble/radar charts, bin maps, hierarchical results table.
                Scenarios/policies are auto-detected; filters below narrow them.
                Empty fields use <code>simulation_analysis_config.json</code>{" "}
                defaults.
              </p>
              <Field label="Horizons (one DAYS=CSV per line, empty = config)">
                <textarea
                  className="input-base w-full font-mono text-xs h-14 resize-y"
                  value={s.simHorizons}
                  onChange={(e) => patch({ simHorizons: e.target.value })}
                  placeholder={
                    "30=public/global/simulation/simulation_summary.csv\n90=public/global/simulation/simulation_summary_90d.csv"
                  }
                  spellCheck={false}
                />
              </Field>
              <div className="grid grid-cols-2 gap-3">
                <PathInput
                  label="Scenarios (City:N:Dist;…)"
                  value={s.simScenarios}
                  onChange={(v) => patch({ simScenarios: v })}
                  placeholder="Rio Maior:100:Gamma-3"
                />
                <PathInput
                  label="Strategies (comma-sep)"
                  value={s.simStrategies}
                  onChange={(v) => patch({ simStrategies: v })}
                  placeholder="all"
                />
                <PathInput
                  label="Constructors (comma-sep)"
                  value={s.simConstructors}
                  onChange={(v) => patch({ simConstructors: v })}
                  placeholder="all"
                />
                <PathInput
                  label="Improvers (comma-sep)"
                  value={s.simImprovers}
                  onChange={(v) => patch({ simImprovers: v })}
                  placeholder="all"
                />
                <PathInput
                  label="Acceptance (comma-sep)"
                  value={s.simAcceptance}
                  onChange={(v) => patch({ simAcceptance: v })}
                  placeholder="all"
                />
                <PathInput
                  label="Output Markdown"
                  value={s.simOutMd}
                  onChange={(v) => patch({ simOutMd: v })}
                  placeholder="public/simulation_analysis.md"
                />
                <PathInput
                  label="Figures Directory"
                  value={s.simFiguresDir}
                  onChange={(v) => patch({ simFiguresDir: v })}
                  placeholder="config default"
                />
              </div>
              <div className="flex flex-wrap gap-4 items-end">
                <Select
                  label="Bin Map Mode"
                  value={s.simMapMode}
                  onChange={(v) => patch({ simMapMode: v })}
                  options={["street", "scatter"] as const}
                />
                <Select
                  label="Heatmap Labels"
                  value={s.simHeatmapLabels}
                  onChange={(v) => patch({ simHeatmapLabels: v })}
                  options={["both", "show", "hide"] as const}
                />
              </div>
              <div className="flex flex-wrap gap-4">
                <Check
                  label="Force overwrite"
                  checked={s.simForce}
                  onChange={(v) => patch({ simForce: v })}
                />
                <Check
                  label="Figures only (skip markdown)"
                  checked={s.simFiguresOnly}
                  onChange={(v) => patch({ simFiguresOnly: v })}
                />
                {engine === "native" && (
                  <Check
                    label="HTML report"
                    checked={s.simHtml}
                    onChange={(v) => patch({ simHtml: v })}
                  />
                )}
              </div>
            </>
          )}
        </div>
      )}

      {tab === "presentation" && (
        <div className="card space-y-4">
          <h2 className="text-sm font-semibold text-gray-200">
            Results Presentation Deck
          </h2>
          <p className="text-[11px] text-canvas-muted">
            21-slide PPTX with native editable OMML equations, diagrams, result
            figures and the hierarchical results table; optional speaker-script
            DOCX and XLSX workbook.
          </p>
          <div className="grid grid-cols-2 gap-3">
            <PathInput
              label="Figures Directory"
              value={s.presFiguresDir}
              onChange={(v) => patch({ presFiguresDir: v })}
              onPick={() => pickDir("presFiguresDir")}
            />
            <PathInput
              label="Output PPTX"
              value={s.presOut}
              onChange={(v) => patch({ presOut: v })}
            />
            <PathInput
              label="Author"
              value={s.presAuthor}
              onChange={(v) => patch({ presAuthor: v })}
              placeholder="content default"
            />
            <PathInput
              label="Co-authors (semicolon-sep)"
              value={s.presCoauthors}
              onChange={(v) => patch({ presCoauthors: v })}
            />
            <PathInput
              label="Research Groups (semicolon-sep)"
              value={s.presGroups}
              onChange={(v) => patch({ presGroups: v })}
            />
          </div>
          <div className="flex flex-wrap gap-4 items-end">
            <Select
              label="Results Table"
              value={s.presResultsTable}
              onChange={(v) => patch({ presResultsTable: v })}
              options={["30d", "90d", "all", "none"] as const}
            />
            <Select
              label="Table Split"
              value={s.presResultsSplit}
              onChange={(v) => patch({ presResultsSplit: v })}
              options={["none", "strategy", "constructor", "improver"] as const}
            />
            <Select
              label="Diagram Images"
              value={s.presImageMode}
              onChange={(v) => patch({ presImageMode: v })}
              options={["native", "fetch"] as const}
            />
          </div>
          <div className="flex flex-wrap gap-4">
            <Check
              label="Speaker script (.docx)"
              checked={s.presSpeakerScript}
              onChange={(v) => patch({ presSpeakerScript: v })}
            />
            <Check
              label="Excel results workbook"
              checked={s.presExcel}
              onChange={(v) => patch({ presExcel: v })}
            />
            {engine === "native" && (
              <>
                <Check
                  label="HTML slideshow"
                  checked={s.presHtml}
                  onChange={(v) => patch({ presHtml: v })}
                />
                <Check
                  label="PDF deck"
                  checked={s.presPdf}
                  onChange={(v) => patch({ presPdf: v })}
                />
              </>
            )}
          </div>
          {s.presSpeakerScript && (
            <PathInput
              label="Speaker Script Output (empty = beside PPTX)"
              value={s.presSpeakerOut}
              onChange={(v) => patch({ presSpeakerOut: v })}
            />
          )}
        </div>
      )}
    </>
  );
}
