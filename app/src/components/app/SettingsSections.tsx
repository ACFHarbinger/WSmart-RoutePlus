import { useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import { Compass, Download, RefreshCw } from "lucide-react";
import { toast } from "sonner";
import { OpenPathToolbar } from "../common/OpenPathToolbar";
import { useDuckDbStore } from "../../store/duckdb";
import { useLayoutStore } from "../../store/layout";
import {
  ARROW_PIPELINE_BUDGET_MS,
  formatPipelineTimingBadge,
  runCsvArrowPipeline,
  runSimulationArrowPipeline,
} from "../../utils/duckdb/arrowPipeline";
import {
  getChartRenderBudgetMs,
  runChartRenderBenchmark,
} from "../../utils/charts/chartRenderBenchmark";

export function SettingsPipelineSection({
  draftRoot,
  duckdbMs,
  setChartRenderMs,
}: {
  draftRoot: string;
  duckdbMs: number | null;
  setChartRenderMs: (ms: number | null) => void;
}) {
  const { ready: duckdbReady, lastPipeline, setLastPipeline, setLoading, loading } =
    useDuckDbStore();
  const [benchRunning, setBenchRunning] = useState(false);
  const [chartBenchRunning, setChartBenchRunning] = useState(false);
  const [arrowBenchPath, setArrowBenchPath] = useState("");

  return (
    <div className="card space-y-3">
      <h2 className="text-sm font-semibold text-gray-200">Data Pipeline (Phase 0)</h2>
      <p className="text-xs text-canvas-muted">
        CSV → Rust Arrow IPC → DuckDB-Wasm (or pre-built .arrow sidecar from a .wsroute
        bundle). Target end-to-end latency &lt; {ARROW_PIPELINE_BUDGET_MS} ms.
      </p>
      <p className="text-xs">
        DuckDB worker:{" "}
        <span className={duckdbReady ? "text-accent-success" : "text-canvas-muted"}>
          {duckdbReady ? "ready" : "initialising…"}
        </span>
        {duckdbMs !== null && (
          <>
            {" "}
            · worker ready in <span className="font-mono">{duckdbMs} ms</span>
          </>
        )}
      </p>
      {lastPipeline && (
        <div className="text-xs text-canvas-muted space-y-0.5 font-mono">
          <p>
            <span
              className={
                lastPipeline.withinBudget ? "text-accent-success" : "text-accent-warning"
              }
            >
              {formatPipelineTimingBadge(lastPipeline)}
            </span>
            {lastPipeline.columnCount > 0 && <> · {lastPipeline.columnCount} cols</>}
          </p>
          <p>
            rust {lastPipeline.rustMs} ms · read {lastPipeline.readMs} ms · duckdb{" "}
            {lastPipeline.duckdbMs} ms
          </p>
        </div>
      )}
      {arrowBenchPath.trim() ? (
        <OpenPathToolbar
          path={arrowBenchPath.trim()}
          projectRoot={draftRoot.trim() || null}
          chipClassName="max-w-full"
        />
      ) : null}
      <button
        disabled={!duckdbReady || benchRunning || loading}
        onClick={async () => {
          const path = (await open({
            filters: [{ name: "CSV / JSONL", extensions: ["csv", "jsonl"] }],
          })) as string | null;
          if (!path) return;
          setArrowBenchPath(path);
          setBenchRunning(true);
          setLoading(true);
          try {
            const timing = path.toLowerCase().endsWith(".jsonl")
              ? await runSimulationArrowPipeline(path, "bench_sim", draftRoot.trim() || null)
              : await runCsvArrowPipeline(path, "bench_csv", draftRoot.trim() || null);
            setLastPipeline(timing);
            toast.success("Arrow pipeline complete", {
              description: `${timing.totalMs} ms (${timing.rowCount} rows)`,
            });
          } catch (err) {
            toast.error("Arrow pipeline failed", { description: String(err) });
          } finally {
            setBenchRunning(false);
            setLoading(false);
          }
        }}
        className="btn-ghost text-xs flex items-center gap-1.5"
      >
        <RefreshCw size={12} className={benchRunning ? "animate-spin" : ""} />
        Run Arrow Pipeline Benchmark
      </button>
      <button
        disabled={chartBenchRunning}
        onClick={async () => {
          setChartBenchRunning(true);
          try {
            const ms = await runChartRenderBenchmark();
            setChartRenderMs(ms);
            const budget = getChartRenderBudgetMs();
            toast.success("Chart render benchmark complete", {
              description: `${ms} ms (budget ${budget} ms)`,
            });
          } catch (err) {
            toast.error("Chart render benchmark failed", { description: String(err) });
          } finally {
            setChartBenchRunning(false);
          }
        }}
        className="btn-ghost text-xs flex items-center gap-1.5"
      >
        <RefreshCw size={12} className={chartBenchRunning ? "animate-spin" : ""} />
        Run Chart Render Benchmark
      </button>
    </div>
  );
}

export function SettingsAboutSection({
  appVersion,
  firstPaintMs,
  prefetchMs,
  duckdbMs,
  withinBudget,
  chartRenderMs,
}: {
  appVersion: string;
  firstPaintMs: number | null;
  prefetchMs: number | null;
  duckdbMs: number | null;
  withinBudget: boolean | null;
  chartRenderMs: number | null;
}) {
  const { setGuidedTourOpen, setGuidedTourStep } = useLayoutStore();
  const [checkingUpdate, setCheckingUpdate] = useState(false);
  const [installingUpdate, setInstallingUpdate] = useState(false);
  const [pendingInstall, setPendingInstall] = useState(false);

  return (
    <div className="card space-y-2 text-xs text-canvas-muted">
      <p className="font-semibold text-gray-300 text-sm">About WSmart-Route Studio</p>
      <p>
        Version: <span className="font-mono">{appVersion}</span>
      </p>
      {firstPaintMs !== null && (
        <p>
          Startup to first paint: <span className="font-mono">{firstPaintMs} ms</span>
        </p>
      )}
      {prefetchMs !== null && (
        <p>
          Route prefetch complete: <span className="font-mono">{prefetchMs} ms</span>
          {" · "}
          <span className={withinBudget ? "text-accent-success" : "text-accent-warning"}>
            {withinBudget ? "within 2s budget" : "over 2s budget"}
          </span>
        </p>
      )}
      {duckdbMs !== null && (
        <p>
          DuckDB-Wasm worker ready: <span className="font-mono">{duckdbMs} ms</span>
        </p>
      )}
      {chartRenderMs !== null && (
        <p>
          Chart render benchmark: <span className="font-mono">{chartRenderMs} ms</span>
          {" · "}
          <span
            className={
              chartRenderMs <= getChartRenderBudgetMs()
                ? "text-accent-success"
                : "text-accent-warning"
            }
          >
            {chartRenderMs <= getChartRenderBudgetMs()
              ? `within ${getChartRenderBudgetMs()} ms budget`
              : `over ${getChartRenderBudgetMs()} ms budget`}
          </span>
        </p>
      )}
      <p>Runtime: Tauri 2.0 · React 19 · Rust</p>
      <div className="flex flex-wrap gap-2 mt-1">
        <button
          disabled={checkingUpdate || installingUpdate}
          onClick={async () => {
            setCheckingUpdate(true);
            setPendingInstall(false);
            try {
              const result = await invoke<{
                available: boolean;
                current_version: string;
                latest_version: string | null;
                message: string;
                can_install: boolean;
                notes: string | null;
              }>("check_for_updates");
              setPendingInstall(result.available && result.can_install);
              if (result.available) {
                toast.success(result.message, {
                  description: [
                    `${result.current_version} → ${result.latest_version}`,
                    result.notes,
                  ]
                    .filter(Boolean)
                    .join(" · "),
                });
              } else {
                toast.info(result.message);
              }
            } catch (err) {
              toast.error("Update check failed", { description: String(err) });
            } finally {
              setCheckingUpdate(false);
            }
          }}
          className="btn-ghost text-xs flex items-center gap-1.5"
        >
          <RefreshCw size={12} className={checkingUpdate ? "animate-spin" : ""} />
          Check for Updates
        </button>
        {pendingInstall && (
          <button
            disabled={installingUpdate || checkingUpdate}
            onClick={async () => {
              setInstallingUpdate(true);
              try {
                await invoke("install_app_update");
              } catch (err) {
                toast.error("Update install failed", { description: String(err) });
                setInstallingUpdate(false);
              }
            }}
            className="btn-primary text-xs flex items-center gap-1.5"
          >
            <Download size={12} className={installingUpdate ? "animate-pulse" : ""} />
            Download & Install
          </button>
        )}
      </div>
      <button
        onClick={() => {
          setGuidedTourStep(0);
          setGuidedTourOpen(true);
        }}
        className="btn-ghost text-xs flex items-center gap-1.5 mt-1"
      >
        <Compass size={12} />
        Take Guided Tour
      </button>
      <p className="text-[10px]">
        Set <code className="font-mono">WSMART_UPDATE_URL</code> to a static JSON manifest
        (version-only check) or a Tauri updater endpoint. For signed installs also set{" "}
        <code className="font-mono">WSMART_UPDATER_PUBKEY</code>. See{" "}
        <code className="font-mono">app/updater.example.json</code>.
      </p>
      <p>
        ROADMAP: <code className="font-mono">docs/moon/ROADMAP.md §G</code>
      </p>
    </div>
  );
}
