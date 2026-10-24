/**
 * System Tools — file-system maintenance and test-suite runner.
 *
 * Ports the retired PySide6 GUI's "File System Tools" (update / delete /
 * cryptography) and "Program Test Suite" tabs to the Studio. Each form
 * assembles a `main.py file_system <sub>` or `main.py test_suite` command
 * and spawns it through the shared process infrastructure (§G.15).
 */
import { useCallback, useMemo, useState } from "react";
import { Play, Terminal, FolderOpen, AlertTriangle } from "lucide-react";
import { open } from "@tauri-apps/plugin-dialog";
import { ProcessLogTail } from "../../components/monitor/process/ProcessLogTail";
import { StatusPill } from "../../components/ui/StatusPill";
import { SubtabBar } from "../../components/ui/telemetry/SubtabBar";
import { useAppStore } from "../../store/app";
import { useSystemToolsStore, type SystemToolsTab } from "../../store/launchers";
import { useProcessStore } from "../../store/process";
import { useSpawnProcess } from "../../hooks/process/useSpawnProcess";
import type { ProcessEntry } from "../../types";

const TABS: { id: SystemToolsTab; label: string }[] = [
  { id: "update", label: "FS Update" },
  { id: "delete", label: "FS Delete" },
  { id: "crypto", label: "Cryptography" },
  { id: "tests", label: "Test Suite" },
];

const UPDATE_OPERATIONS = ["", "=", "+", "-", "*", "/", "**", "//", "%"] as const;
const STATS_FUNCTIONS = ["", "mean", "median", "std", "min", "max", "sum"] as const;

// Mirrors logic.src.constants TEST_MODULES (minus the removed gui module)
const TEST_MODULES = [
  "parser", "train", "mrl", "hp_optim", "gen_data", "eval", "test_sim",
  "file_system", "actions", "edge_cases", "layers", "scheduler",
  "optimizer", "integration",
] as const;

const DELETE_TARGETS: { key: string; label: string; inverted: boolean }[] = [
  // inverted: CLI flag is store_false (deleted by default; flag opts out)
  { key: "delWandb", label: "WandB logs (wandb/)", inverted: true },
  { key: "delLog", label: "Train logs (logs/)", inverted: true },
  { key: "delOutput", label: "Model outputs (model_weights/)", inverted: true },
  { key: "delData", label: "Datasets (datasets/)", inverted: false },
  { key: "delEval", label: "Evaluation results (results/)", inverted: false },
  { key: "delTest", label: "Simulator test outputs (assets/output/)", inverted: false },
  { key: "delTestCheckpoint", label: "Simulator checkpoints (assets/temp/)", inverted: false },
  { key: "delCache", label: "Cache directories", inverted: false },
];

function isSystemToolsProcess(id: string): boolean {
  return id.startsWith("fstool_") || id.startsWith("testsuite_");
}

function findRecentSystemToolsProcessId(
  processes: Record<string, ProcessEntry>
): string | null {
  const candidates = Object.entries(processes)
    .filter(([id]) => isSystemToolsProcess(id))
    .sort((a, b) => b[1].startTime - a[1].startTime);
  return candidates[0]?.[0] ?? null;
}

function Field({
  label, children,
}: { label: string; children: React.ReactNode }) {
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

export function SystemTools() {
  const projectRoot = useAppStore((s) => s.projectRoot);
  const { spawn, launching } = useSpawnProcess();
  const s = useSystemToolsStore();
  const { tab, patch } = s;

  const [liveProcessId, setLiveProcessId] = useState<string | null>(null);
  const processes = useProcessStore((p) => p.processes);
  const displayProcessId = useMemo(
    () => liveProcessId ?? findRecentSystemToolsProcessId(processes),
    [liveProcessId, processes]
  );
  const displayProc = displayProcessId ? processes[displayProcessId] : null;

  const pickPath = async (field: "targetEntry" | "cryptoInputPath" | "cryptoOutputPath", directory = false) => {
    const path = (await open({ directory })) as string | null;
    if (path) patch({ [field]: path } as never);
  };

  const cliArgs = useMemo(() => {
    if (tab === "update") {
      const args = ["file_system", "update"];
      if (s.targetEntry.trim()) args.push("--target_entry", s.targetEntry.trim());
      if (s.outputKey.trim()) args.push("--output_key", s.outputKey.trim());
      if (s.filenamePattern.trim()) args.push("--filename_pattern", s.filenamePattern.trim());
      if (s.updateOperation) args.push("--update_operation", s.updateOperation, "--update_value", String(s.updateValue));
      if (s.statsFunction) args.push("--stats_function", s.statsFunction);
      if (s.outputFilename.trim()) args.push("--output_filename", s.outputFilename.trim());
      const keys = s.inputKeys.trim().split(/\s+/).filter(Boolean);
      if (keys.length) args.push("--input_keys", ...keys);
      if (s.updatePreview) args.push("--update_preview");
      return args;
    }
    if (tab === "delete") {
      const args = ["file_system", "delete"];
      for (const t of DELETE_TARGETS) {
        const checked = s[t.key as keyof typeof s] as boolean;
        // inverted flags opt OUT of deletion; regular flags opt IN
        if (t.inverted && !checked) args.push(`--${flagName(t.key)}`);
        if (!t.inverted && checked) args.push(`--${flagName(t.key)}`);
      }
      if (s.deletePreview) args.push("--delete_preview");
      return args;
    }
    if (tab === "crypto") {
      const args = ["file_system", "cryptography"];
      if (s.envFile.trim()) args.push("--env_file", s.envFile.trim());
      if (s.symkeyName.trim()) args.push("--symkey_name", s.symkeyName.trim());
      if (s.cryptoInputPath.trim()) args.push("--input_path", s.cryptoInputPath.trim());
      if (s.cryptoOutputPath.trim()) args.push("--output_path", s.cryptoOutputPath.trim());
      args.push("--salt_size", String(s.saltSize));
      args.push("--key_length", String(s.keyLength));
      args.push("--hash_iterations", String(s.hashIterations));
      return args;
    }
    const args = ["test_suite"];
    if (s.testModules.length) args.push("-m", ...s.testModules);
    if (s.testKeyword.trim()) args.push("-k", s.testKeyword.trim());
    if (s.testMarkers.trim()) args.push("--markers", s.testMarkers.trim());
    if (s.testVerbose) args.push("-v");
    if (s.testCoverage) args.push("--coverage");
    if (s.testParallel) args.push("-n");
    if (s.testFailedFirst) args.push("--ff");
    if (s.testMaxfail > 0) args.push("--maxfail", String(s.testMaxfail));
    if (s.testDir.trim() && s.testDir.trim() !== "tests") args.push("--test-dir", s.testDir.trim());
    return args;
  }, [tab, s]);

  const commandPreview = `python main.py ${cliArgs.join(" ")}`;

  const launch = useCallback(async () => {
    if (!projectRoot) return;
    const procId = `${tab === "tests" ? "testsuite" : "fstool"}_${Date.now()}`;
    setLiveProcessId(procId);
    await spawn({
      id: procId,
      pythonArgs: ["main.py", ...cliArgs],
      workingDir: projectRoot,
    });
  }, [projectRoot, tab, cliArgs, spawn]);

  const deleteArmed =
    tab === "delete" &&
    !s.deletePreview;

  return (
    <div className="space-y-4 max-w-2xl">
      {/* Tab selector */}
      <div className="bg-noc-bgPanel border border-noc-border rounded-xl p-4 space-y-3">
        <h2 className="text-sm font-semibold text-noc-textPrimary">System Tools</h2>
        <SubtabBar
          tabs={TABS}
          activeId={tab}
          onChange={(id) => patch({ tab: id as SystemToolsTab })}
        />
      </div>

      {tab === "update" && (
        <div className="card space-y-4">
          <h2 className="text-sm font-semibold text-gray-200">File System Update</h2>
          <Field label="Target Entry (file or directory)">
            <div className="flex gap-2">
              <input
                type="text"
                className="input-base font-mono text-xs flex-1"
                value={s.targetEntry}
                onChange={(e) => patch({ targetEntry: e.target.value })}
                placeholder="path/to/file-or-dir"
              />
              <button onClick={() => pickPath("targetEntry")} className="btn-ghost text-xs flex items-center gap-1">
                <FolderOpen size={12} />
              </button>
            </div>
          </Field>
          <div className="flex flex-wrap gap-4 items-end">
            <Field label="Output Key">
              <input type="text" className="input-base font-mono text-sm w-36" value={s.outputKey} onChange={(e) => patch({ outputKey: e.target.value })} />
            </Field>
            <Field label="Filename Pattern">
              <input type="text" className="input-base font-mono text-sm w-36" value={s.filenamePattern} onChange={(e) => patch({ filenamePattern: e.target.value })} placeholder="*.json" />
            </Field>
            <Field label="Operation">
              <select className="select-base w-24" value={s.updateOperation} onChange={(e) => patch({ updateOperation: e.target.value })}>
                {UPDATE_OPERATIONS.map((o) => <option key={o} value={o}>{o || "(none)"}</option>)}
              </select>
            </Field>
            <Field label="Value">
              <input type="number" className="input-base font-mono text-sm w-24" value={s.updateValue} onChange={(e) => patch({ updateValue: Number(e.target.value) })} />
            </Field>
          </div>
          <div className="flex flex-wrap gap-4 items-end">
            <Field label="Stats Function">
              <select className="select-base w-28" value={s.statsFunction} onChange={(e) => patch({ statsFunction: e.target.value })}>
                {STATS_FUNCTIONS.map((f) => <option key={f} value={f}>{f || "(none)"}</option>)}
              </select>
            </Field>
            <Field label="Output Filename">
              <input type="text" className="input-base font-mono text-sm w-44" value={s.outputFilename} onChange={(e) => patch({ outputFilename: e.target.value })} />
            </Field>
            <Field label="Input Keys (space-separated)">
              <input type="text" className="input-base font-mono text-sm w-44" value={s.inputKeys} onChange={(e) => patch({ inputKeys: e.target.value })} />
            </Field>
          </div>
          <Check label="Preview changes only (no write)" checked={s.updatePreview} onChange={(v) => patch({ updatePreview: v })} />
        </div>
      )}

      {tab === "delete" && (
        <div className="card space-y-4">
          <h2 className="text-sm font-semibold text-gray-200">File System Delete</h2>
          <div className="grid grid-cols-2 gap-2">
            {DELETE_TARGETS.map((t) => (
              <Check
                key={t.key}
                label={t.label}
                checked={s[t.key as keyof typeof s] as boolean}
                onChange={(v) => patch({ [t.key]: v } as never)}
              />
            ))}
          </div>
          <Check label="Preview only (list matches, no deletion)" checked={s.deletePreview} onChange={(v) => patch({ deletePreview: v })} />
          {deleteArmed && (
            <p className="text-xs text-accent-warning flex items-center gap-1.5">
              <AlertTriangle size={12} />
              Preview is off — checked directories will be deleted permanently.
            </p>
          )}
        </div>
      )}

      {tab === "crypto" && (
        <div className="card space-y-4">
          <h2 className="text-sm font-semibold text-gray-200">Cryptography Tools</h2>
          <div className="flex flex-wrap gap-4 items-end">
            <Field label="Environment File">
              <input type="text" className="input-base font-mono text-sm w-44" value={s.envFile} onChange={(e) => patch({ envFile: e.target.value })} />
            </Field>
            <Field label="Symmetric Key Name">
              <input type="text" className="input-base font-mono text-sm w-44" value={s.symkeyName} onChange={(e) => patch({ symkeyName: e.target.value })} />
            </Field>
          </div>
          <Field label="Input Path">
            <div className="flex gap-2">
              <input type="text" className="input-base font-mono text-xs flex-1" value={s.cryptoInputPath} onChange={(e) => patch({ cryptoInputPath: e.target.value })} />
              <button onClick={() => pickPath("cryptoInputPath")} className="btn-ghost text-xs flex items-center gap-1">
                <FolderOpen size={12} />
              </button>
            </div>
          </Field>
          <Field label="Output Path">
            <input type="text" className="input-base font-mono text-xs w-full" value={s.cryptoOutputPath} onChange={(e) => patch({ cryptoOutputPath: e.target.value })} />
          </Field>
          <div className="flex flex-wrap gap-4 items-end">
            <Field label="Salt Size (bytes)">
              <input type="number" className="input-base font-mono text-sm w-24" value={s.saltSize} min={1} max={256} onChange={(e) => patch({ saltSize: Number(e.target.value) })} />
            </Field>
            <Field label="Key Length (bytes)">
              <input type="number" className="input-base font-mono text-sm w-24" value={s.keyLength} min={1} max={1024} onChange={(e) => patch({ keyLength: Number(e.target.value) })} />
            </Field>
            <Field label="Hash Iterations">
              <input type="number" className="input-base font-mono text-sm w-32" value={s.hashIterations} min={1000} step={10000} onChange={(e) => patch({ hashIterations: Number(e.target.value) })} />
            </Field>
          </div>
        </div>
      )}

      {tab === "tests" && (
        <div className="card space-y-4">
          <h2 className="text-sm font-semibold text-gray-200">Program Test Suite</h2>
          <Field label="Modules (empty = all)">
            <div className="grid grid-cols-3 gap-1.5">
              {TEST_MODULES.map((m) => (
                <Check
                  key={m}
                  label={m}
                  checked={s.testModules.includes(m)}
                  onChange={(v) =>
                    patch({
                      testModules: v
                        ? [...s.testModules, m]
                        : s.testModules.filter((x) => x !== m),
                    })
                  }
                />
              ))}
            </div>
          </Field>
          <div className="flex flex-wrap gap-4 items-end">
            <Field label="Keyword (-k)">
              <input type="text" className="input-base font-mono text-sm w-40" value={s.testKeyword} onChange={(e) => patch({ testKeyword: e.target.value })} />
            </Field>
            <Field label="Markers">
              <input type="text" className="input-base font-mono text-sm w-40" value={s.testMarkers} onChange={(e) => patch({ testMarkers: e.target.value })} />
            </Field>
            <Field label="Max Failures (0 = ∞)">
              <input type="number" className="input-base font-mono text-sm w-24" value={s.testMaxfail} min={0} onChange={(e) => patch({ testMaxfail: Number(e.target.value) })} />
            </Field>
            <Field label="Test Directory">
              <input type="text" className="input-base font-mono text-sm w-32" value={s.testDir} onChange={(e) => patch({ testDir: e.target.value })} />
            </Field>
          </div>
          <div className="flex flex-wrap gap-4">
            <Check label="Verbose" checked={s.testVerbose} onChange={(v) => patch({ testVerbose: v })} />
            <Check label="Coverage" checked={s.testCoverage} onChange={(v) => patch({ testCoverage: v })} />
            <Check label="Parallel" checked={s.testParallel} onChange={(v) => patch({ testParallel: v })} />
            <Check label="Failed First" checked={s.testFailedFirst} onChange={(v) => patch({ testFailedFirst: v })} />
          </div>
        </div>
      )}

      {/* Command preview */}
      <div className="card space-y-2">
        <div className="flex items-center gap-2 text-xs text-canvas-muted">
          <Terminal size={12} />
          Command preview
        </div>
        <pre className="font-mono text-xs text-accent-secondary whitespace-pre-wrap bg-canvas-bg rounded-lg p-3">
          {commandPreview}
        </pre>
      </div>

      {/* Launch */}
      <div className="flex items-center gap-3">
        {!projectRoot && (
          <p className="text-xs text-accent-warning">Configure Project Root in Settings first.</p>
        )}
        <button
          onClick={launch}
          disabled={launching || !projectRoot}
          className={deleteArmed ? "btn-primary bg-red-600 hover:bg-red-500 flex items-center gap-2" : "btn-primary flex items-center gap-2"}
        >
          <Play size={14} />
          {launching ? "Launching…" : tab === "tests" ? "Run Tests" : "Run Tool"}
        </button>
      </div>

      {/* Live output */}
      {displayProc && (
        <div className="card space-y-2">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-semibold text-gray-200">
              {displayProcessId?.startsWith("testsuite_") ? "Test Suite Output" : "Tool Output"}
            </h2>
            <StatusPill status={displayProc.status} />
          </div>
          <ProcessLogTail
            logLines={displayProc.logLines}
            maxLines={40}
            waiting={displayProc.status === "running"}
          />
        </div>
      )}
    </div>
  );
}

function flagName(storeKey: string): string {
  const map: Record<string, string> = {
    delWandb: "wandb",
    delLog: "log",
    delOutput: "output",
    delData: "data",
    delEval: "eval",
    delTest: "test",
    delTestCheckpoint: "test_checkpoint",
    delCache: "cache",
  };
  return map[storeKey] ?? storeKey;
}
