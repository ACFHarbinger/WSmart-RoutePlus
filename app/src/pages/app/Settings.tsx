/**
 * Settings — persistent app configuration (§G.19).
 *
 * Stores all values in the Zustand persist store (Tauri Store plugin under the hood)
 * so they survive app restarts.
 *
 * Fields:
 *   Project Root    — the repo root where main.py lives; used by all launchers
 *   Python Path     — override the auto-detected Python executable (blank = auto)
 *   Theme           — dark / light / system
 *   Reduce Motion   — accessibility / low-GPU mode
 *   Window          — background mode (BackgroundModeCard; applies at once)
 */
import { useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { open, save as saveDialog } from "@tauri-apps/plugin-dialog";
import {
  CheckCircle,
  Download,
  FolderOpen,
  Save,
  Upload,
  XCircle,
} from "lucide-react";
import { OpenPathToolbar } from "../../components/common/OpenPathToolbar";
import { BackgroundModeCard } from "../../components/settings/BackgroundModeCard";
import { useStartupTiming } from "../../hooks/app/useStartupTiming";
import { toast } from "sonner";
import { useAppStore } from "../../store/app";
import {
  SettingsAboutSection,
  SettingsPipelineSection,
} from "../../components/app/SettingsSections";

type ValidationState = "idle" | "checking" | "ok" | "error";

interface FieldValidation {
  state: ValidationState;
  message: string;
}

const IDLE: FieldValidation = { state: "idle", message: "" };

function ValidationBadge({ v }: { v: FieldValidation }) {
  if (v.state === "idle" || v.state === "checking") return null;
  const ok = v.state === "ok";
  return (
    <p
      className={`flex items-center gap-1.5 text-xs mt-1 ${
        ok ? "text-accent-success" : "text-accent-danger"
      }`}
    >
      {ok ? <CheckCircle size={11} /> : <XCircle size={11} />}
      {v.message}
    </p>
  );
}

export function Settings() {
  const {
    projectRoot,
    pythonPath,
    theme,
    effectiveTheme,
    reduceMotion,
    setProjectRoot,
    setPythonPath,
    setTheme,
    setReduceMotion,
  } = useAppStore();

  const [draftRoot, setDraftRoot] = useState(projectRoot);
  const [draftPython, setDraftPython] = useState(pythonPath);
  const [draftTheme, setDraftTheme] = useState(theme);
  const [draftReduceMotion, setDraftReduceMotion] = useState(reduceMotion);

  const [rootValidation, setRootValidation] = useState<FieldValidation>(IDLE);
  const [pythonValidation, setPythonValidation] = useState<FieldValidation>(IDLE);
  const [appVersion, setAppVersion] = useState("…");
  const { firstPaintMs, prefetchMs, duckdbMs, withinBudget } = useStartupTiming();
  const [chartRenderMs, setChartRenderMs] = useState<number | null>(null);
  const [importSettingsPath, setImportSettingsPath] = useState("");

  useEffect(() => {
    invoke<string>("get_app_version")
      .then(setAppVersion)
      .catch(() => setAppVersion("0.1.0"));
  }, []);

  useEffect(() => {
    setDraftTheme(theme);
    setDraftReduceMotion(reduceMotion);
  }, [theme, reduceMotion]);

  const isDirty =
    draftRoot.trim() !== projectRoot ||
    draftPython.trim() !== pythonPath ||
    draftTheme !== theme ||
    draftReduceMotion !== reduceMotion;

  const pickRoot = async () => {
    const path = (await open({ directory: true })) as string | null;
    if (path) {
      setDraftRoot(path);
      setRootValidation(IDLE);
    }
  };

  const validateRoot = async () => {
    const path = draftRoot.trim();
    if (!path) {
      setRootValidation({ state: "error", message: "Path is required" });
      return false;
    }
    setRootValidation({ state: "checking", message: "" });
    try {
      const msg = await invoke<string>("validate_project_root", { path });
      setRootValidation({ state: "ok", message: msg });
      return true;
    } catch (e) {
      setRootValidation({ state: "error", message: String(e) });
      return false;
    }
  };

  const validatePython = async () => {
    const py = draftPython.trim();
    if (!py) {
      setPythonValidation(IDLE);
      return true;
    }
    setPythonValidation({ state: "checking", message: "" });
    try {
      const version = await invoke<string>("probe_python", { pythonPath: py });
      setPythonValidation({ state: "ok", message: version });
      return true;
    } catch (e) {
      setPythonValidation({ state: "error", message: String(e) });
      return false;
    }
  };

  const save = async () => {
    const rootOk = await validateRoot();
    const pythonOk = await validatePython();
    if (!rootOk || !pythonOk) {
      toast.error("Fix validation errors before saving");
      return;
    }
    setProjectRoot(draftRoot.trim());
    setPythonPath(draftPython.trim());
    setTheme(draftTheme);
    setReduceMotion(draftReduceMotion);
    toast.success("Settings saved");
  };

  const discard = () => {
    setDraftRoot(projectRoot);
    setDraftPython(pythonPath);
    setDraftTheme(theme);
    setDraftReduceMotion(reduceMotion);
    setRootValidation(IDLE);
    setPythonValidation(IDLE);
  };

  const exportSettings = async () => {
    const path = (await saveDialog({
      filters: [{ name: "JSON", extensions: ["json"] }],
      defaultPath: "wsmart-studio-settings.json",
    })) as string | null;
    if (!path) return;
    const data = {
      projectRoot,
      pythonPath,
      theme,
      reduceMotion,
    };
    try {
      await invoke("write_text_file", { path, content: JSON.stringify(data, null, 2) });
      toast.success("Settings exported", { description: path.split("/").pop() });
    } catch (err) {
      toast.error("Export failed", { description: String(err) });
    }
  };

  const importSettings = async () => {
    const path = (await open({
      filters: [{ name: "JSON", extensions: ["json"] }],
    })) as string | null;
    if (!path) return;
    setImportSettingsPath(path);
    try {
      const text = await invoke<string>("read_text_file", { path });
      const data = JSON.parse(text) as Record<string, unknown>;
      if (typeof data.projectRoot === "string") setDraftRoot(data.projectRoot);
      if (typeof data.pythonPath === "string") setDraftPython(data.pythonPath);
      if (data.theme === "dark" || data.theme === "light" || data.theme === "system") {
        setDraftTheme(data.theme);
      }
      if (typeof data.reduceMotion === "boolean") {
        setDraftReduceMotion(data.reduceMotion);
      }
      setRootValidation(IDLE);
      setPythonValidation(IDLE);
      toast.success("Settings imported — review and save to apply");
    } catch (err) {
      toast.error("Import failed", { description: String(err) });
    }
  };

  return (
    <div className="max-w-xl space-y-6">
      {/* Project Root */}
      <div className="card space-y-3">
        <h2 className="text-sm font-semibold text-gray-200">Project Root</h2>
        <p className="text-xs text-canvas-muted">
          The directory that contains <code className="font-mono">main.py</code>.
          All launchers use this as their working directory.
        </p>
        <div className="flex gap-2">
          <input
            type="text"
            className="input-base flex-1 font-mono text-xs"
            value={draftRoot}
            onChange={(e) => {
              setDraftRoot(e.target.value);
              setRootValidation(IDLE);
            }}
            onBlur={validateRoot}
            placeholder="/path/to/WSmart-Route"
            spellCheck={false}
          />
          <button
            onClick={pickRoot}
            className="btn-ghost shrink-0 flex items-center gap-1.5 text-xs"
          >
            <FolderOpen size={13} />
            Browse
          </button>
        </div>
        {draftRoot.trim() ? (
          <OpenPathToolbar
            path={draftRoot.trim()}
            chipClassName="max-w-full"
            handoff={false}
          />
        ) : null}
        {!draftRoot && (
          <p className="text-xs text-accent-warning">
            Required — launchers will be disabled without this.
          </p>
        )}
        <ValidationBadge v={rootValidation} />
      </div>

      {/* Python Executable */}
      <div className="card space-y-3">
        <h2 className="text-sm font-semibold text-gray-200">Python Executable</h2>
        <p className="text-xs text-canvas-muted">
          Leave blank to auto-detect: Studio checks{" "}
          <code className="font-mono text-xs">&lt;project&gt;/.venv/bin/python</code>{" "}
          (uv venv) first, then <code className="font-mono text-xs">python3</code> on PATH.
        </p>
        <input
          type="text"
          className="input-base font-mono text-xs w-full"
          value={draftPython}
          onChange={(e) => {
            setDraftPython(e.target.value);
            setPythonValidation(IDLE);
          }}
          onBlur={validatePython}
          placeholder="(auto-detect)"
          spellCheck={false}
        />
        {draftPython.trim() ? (
          <OpenPathToolbar
            path={draftPython.trim()}
            projectRoot={draftRoot.trim() || null}
            chipClassName="max-w-full"
            handoff={false}
          />
        ) : null}
        <ValidationBadge v={pythonValidation} />
      </div>

      {/* Theme & Motion */}
      <div className="card space-y-4">
        <div>
          <h2 className="text-sm font-semibold text-gray-200">Appearance</h2>
          <p className="text-xs text-canvas-muted mt-0.5">
            Select standard desktop color palette or follow operating system preference.
          </p>
        </div>
        <div className="flex gap-3 flex-wrap">
          {(["dark", "light", "system"] as const).map((t) => (
            <label
              key={t}
              className={`flex items-center gap-2 py-2 px-4 rounded-lg border cursor-pointer transition-colors ${
                draftTheme === t
                  ? "border-accent-primary bg-accent-primary/15 text-accent-secondary"
                  : "border-canvas-border text-gray-400 hover:border-canvas-muted"
              }`}
            >
              <input
                type="radio"
                name="theme"
                value={t}
                checked={draftTheme === t}
                onChange={() => setDraftTheme(t)}
                className="sr-only"
              />
              <span className="text-sm capitalize">
                {t === "system" ? "System" : t}
              </span>
            </label>
          ))}
        </div>
        {draftTheme === "system" && (
          <p className="text-xs text-canvas-muted">
            Currently using <span className="capitalize text-gray-300">{effectiveTheme}</span> theme
            from your system appearance.
          </p>
        )}

        <div className="pt-3 border-t border-canvas-border">
          <label className="flex items-center justify-between cursor-pointer">
            <div>
              <span className="text-sm font-medium text-gray-200">
                Reduce Motion
              </span>
              <p className="text-xs text-canvas-muted mt-0.5">
                Disable chart transitions, pulsating rings, and HUD telemetry animations.
              </p>
            </div>
            <input
              type="checkbox"
              checked={draftReduceMotion}
              onChange={(e) => setDraftReduceMotion(e.target.checked)}
              className="accent-accent-primary h-4 w-4 rounded"
            />
          </label>
        </div>
      </div>

      <BackgroundModeCard />
      {/* Save / Discard */}
      <div className="flex items-center gap-3">
        <button
          onClick={save}
          disabled={!isDirty}
          className="btn-primary flex items-center gap-2"
        >
          <Save size={14} />
          Save Settings
        </button>
        {isDirty && (
          <button onClick={discard} className="btn-ghost text-sm">
            Discard
          </button>
        )}
      </div>

      {/* Import / Export */}
      <div className="card space-y-3">
        <h2 className="text-sm font-semibold text-gray-200">Backup & Restore</h2>
        <p className="text-xs text-canvas-muted">
          Export your settings to JSON for backup or team onboarding. Import populates the
          draft fields — review and click Save Settings to apply.
        </p>
        <div className="flex gap-3">
          <button
            onClick={exportSettings}
            className="btn-ghost flex items-center gap-2 text-sm"
          >
            <Download size={13} />
            Export Settings
          </button>
          <button
            onClick={importSettings}
            className="btn-ghost flex items-center gap-2 text-sm"
          >
            <Upload size={13} />
            Import Settings
          </button>
        </div>
        {importSettingsPath.trim() ? (
          <OpenPathToolbar
            path={importSettingsPath.trim()}
            projectRoot={draftRoot.trim() || null}
            chipClassName="max-w-full"
            handoff={false}
          />
        ) : null}
      </div>

      {/* Phase 0 Data Pipeline */}
      <SettingsPipelineSection
        draftRoot={draftRoot}
        duckdbMs={duckdbMs}
        setChartRenderMs={setChartRenderMs}
      />

      {/* About */}
      <SettingsAboutSection
        appVersion={appVersion}
        firstPaintMs={firstPaintMs}
        prefetchMs={prefetchMs}
        duckdbMs={duckdbMs}
        withinBudget={withinBudget}
        chartRenderMs={chartRenderMs}
      />
    </div>
  );
}
