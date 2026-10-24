/**
 * ML Introspection — TensorDict/NPZ inspector, attention heatmap, loss contour (§G.5).
 */
import { useCallback, useEffect, useMemo, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import { FolderOpen, RefreshCw } from "lucide-react";
import { toast } from "sonner";
import { OpenPathToolbar } from "../../common/OpenPathToolbar";
import { useLogPathRunLabelBrush } from "../../../hooks/brush/useLogPathRunLabelBrush";
import { useAppStore } from "../../../store/app";
import { runTensorArrowPipeline, type ArrowPipelineTiming } from "../../../utils/duckdb/arrowPipeline";
import { distributionDisplayName, inferDistributionLabel } from "../../../utils/benchmark/distributionCompare";
import { loadGraphCoordinates } from "../../../utils/graph/graphCoords";
import { resolveBpcMarker, type LandscapeMarker } from "../../../utils/graph/lossLandscape";
import { classifyAttentionRole, defaultIndices, detectHeadAxis, groupAttentionKeys, leadingIndexCount, suggestAttentionKeys, type AttentionRole } from "../../../utils/charts/tensorHeatmap";
import { RuntimeAttentionPanel } from "../topology/RuntimeAttentionPanel";
import { parseAttentionVizLine } from "../../../utils/graph/attentionViz";
import type { AttentionVizEntry, NpzArchiveInfo, NpzVectorData, TensorSlicePreview } from "../../../types";
import type { LossLandscapeView } from "./LossLandscape3D";
import { MLArchiveTableSection, MLAttentionControlsToolbar, type AttentionView, type CompareMode, type IntrospectionTab } from "./MLAttentionControls";
import { MLAttentionVisualSection, MLLossLandscapeSection } from "./MLAttentionAndLossViews";
import { useMLIntrospectionOptions } from "./useMLIntrospectionOptions";

export function MLIntrospectionPanel({
  logScale = false,
}: {
  logScale?: boolean;
}) {
  const { projectRoot, pythonPath, effectiveTheme: theme } = useAppStore();
  const [tab, setTab] = useState<IntrospectionTab>("tensor");
  const [archivePath, setArchivePath] = useState<string | null>(null);
  useLogPathRunLabelBrush(archivePath);
  const [archive, setArchive] = useState<NpzArchiveInfo | null>(null);
  const [selectedKey, setSelectedKey] = useState<string>("");
  const [indices, setIndices] = useState<number[]>([]);
  const [decodeStep, setDecodeStep] = useState(0);
  const [compareStep, setCompareStep] = useState(0);
  const [preview, setPreview] = useState<TensorSlicePreview | null>(null);
  const [comparePreview, setComparePreview] = useState<TensorSlicePreview | null>(null);
  const [distArchivePath, setDistArchivePath] = useState<string | null>(null);
  const [distPreview, setDistPreview] = useState<TensorSlicePreview | null>(null);
  const [lossPreview, setLossPreview] = useState<TensorSlicePreview | null>(null);
  const [lossMarkers, setLossMarkers] = useState<LandscapeMarker[]>([]);
  const [loading, setLoading] = useState(false);
  const [topK, setTopK] = useState(32);
  const [sparseK, setSparseK] = useState(0);
  const [compareMode, setCompareMode] = useState<CompareMode>("single");
  const [headIndex, setHeadIndex] = useState(0);
  const [attentionRole, setAttentionRole] = useState<AttentionRole | "all">("all");
  const [attentionView, setAttentionView] = useState<AttentionView>("heatmap");
  const [graphPreset, setGraphPreset] = useState("rm-100");
  const [graphCoords, setGraphCoords] = useState<Array<{ lat: number; lng: number }> | null>(null);
  const [tensorPipeline, setTensorPipeline] = useState<ArrowPipelineTiming | null>(null);
  const [pipelineLoading, setPipelineLoading] = useState(false);
  const [clusterK, setClusterK] = useState(0);
  const [lossView, setLossView] = useState<LossLandscapeView>("surface");
  const [runtimeAttentionEntries, setRuntimeAttentionEntries] = useState<AttentionVizEntry[]>([]);

  const pickArchive = useCallback(async () => {
    const selected = await open({
      multiple: false,
      filters: [{ name: "Tensor archives", extensions: ["npz", "npy", "td"] }],
    });
    if (!selected || typeof selected !== "string") return;
    setArchivePath(selected);
  }, []);

  const pickDistArchive = useCallback(async () => {
    const selected = await open({
      multiple: false,
      filters: [{ name: "Tensor archives", extensions: ["npz", "npy", "td"] }],
    });
    if (!selected || typeof selected !== "string") return;
    setDistArchivePath(selected);
  }, []);

  const loadAttentionLog = useCallback(async () => {
    const selected = await open({
      multiple: false,
      filters: [{ name: "Attention viz log", extensions: ["jsonl"] }],
    });
    if (!selected || typeof selected !== "string") return;
    try {
      const text = await invoke<string>("read_text_file", { path: selected });
      const entries = text
        .split("\n")
        .map((line) => parseAttentionVizLine(line))
        .filter((entry): entry is AttentionVizEntry => entry !== null);
      setRuntimeAttentionEntries(entries);
      toast.success("Attention log loaded", { description: `${entries.length} capture(s)` });
    } catch (err) {
      toast.error("Load failed", { description: String(err) });
    }
  }, []);

  const primaryDistLabel = useMemo(
    () =>
      archivePath
        ? distributionDisplayName(
            inferDistributionLabel(
              archivePath,
              archive?.arrays.map((a) => a.key)
            )
          )
        : "Primary",
    [archivePath, archive]
  );

  const distCompareLabel = useMemo(
    () =>
      distArchivePath
        ? distributionDisplayName(inferDistributionLabel(distArchivePath))
        : "Compare",
    [distArchivePath]
  );

  const inspectArchive = useCallback(
    async (path: string) => {
      setLoading(true);
      try {
        const info = await invoke<NpzArchiveInfo>("inspect_npz_archive", {
          path,
          projectRoot: projectRoot || null,
          pythonExecutable: pythonPath || null,
        });
        setArchive(info);
        const attnKeys = suggestAttentionKeys(info.arrays);
        const lossKey = info.arrays.find((a) =>
          /loss_grid|loss_surface|landscape/i.test(a.key)
        );
        const defaultKey =
          attnKeys[0] ??
          lossKey?.key ??
          info.arrays.find((a) => a.shape.length >= 2)?.key ??
          "";
        setSelectedKey(defaultKey);
        if (defaultKey) {
          const shape = info.arrays.find((a) => a.key === defaultKey)?.shape ?? [];
          setIndices(defaultIndices(shape));
          setHeadIndex(0);
        }
        toast.success("Archive inspected", {
          description: `${info.arrays.length} array(s)`,
        });
      } catch (err) {
        setArchive(null);
        toast.error("Inspect failed", { description: String(err) });
      } finally {
        setLoading(false);
      }
    },
    [projectRoot, pythonPath]
  );

  useEffect(() => {
    if (archivePath) void inspectArchive(archivePath);
  }, [archivePath, inspectArchive]);

  const selectedArray = archive?.arrays.find((a) => a.key === selectedKey);
  const leading = selectedArray ? leadingIndexCount(selectedArray.shape) : 0;
  const headAxis = selectedArray ? detectHeadAxis(selectedArray.shape, selectedKey) : null;
  const headCount = headAxis != null ? selectedArray?.shape[headAxis] ?? 1 : 0;
  const attentionGroups = useMemo(
    () => (archive ? groupAttentionKeys(archive.arrays) : null),
    [archive]
  );
  const activeAttentionRole = classifyAttentionRole(selectedKey);
  const roleFilteredKeys = useMemo(() => {
    if (!archive || tab !== "attention") return [];
    if (attentionRole === "all") {
      const suggested = suggestAttentionKeys(archive.arrays);
      return suggested.length
        ? suggested
        : archive.arrays.filter((a) => a.shape.length >= 2).map((a) => a.key);
    }
    const group = attentionGroups?.[attentionRole] ?? [];
    return group.length
      ? group
      : archive.arrays.filter((a) => a.shape.length >= 2).map((a) => a.key);
  }, [archive, tab, attentionRole, attentionGroups]);

  const effectiveIndices = useMemo(() => {
    const base = [...indices];
    if (headAxis != null) base[headAxis] = headIndex;
    if (leading >= 1 && tab === "attention") {
      const stepDim = leading - 1;
      if (stepDim >= 0) base[stepDim] = decodeStep;
    }
    return base;
  }, [indices, leading, decodeStep, tab, headAxis, headIndex]);

  const compareIndices = useMemo(() => {
    const base = [...effectiveIndices];
    if (leading >= 1 && tab === "attention" && compareMode !== "single") {
      const stepDim = leading - 1;
      if (stepDim >= 0) base[stepDim] = compareStep;
    }
    return base;
  }, [effectiveIndices, leading, compareStep, tab, compareMode]);

  const loadSlice = useCallback(
    async (path: string, idx: number[], key: string) => {
      if (!path || !key) return null;
      return invoke<TensorSlicePreview>("load_tensor_slice", {
        path,
        key: path.endsWith(".npy") ? null : key,
        indices: idx,
        maxDim: tab === "loss" ? 96 : topK,
        projectRoot: projectRoot || null,
        pythonExecutable: pythonPath || null,
      });
    },
    [tab, topK, projectRoot, pythonPath]
  );

  const loadLossMarkers = useCallback(async (path: string, rows: number, cols: number) => {
    if (!path.endsWith(".npz")) {
      setLossMarkers([]);
      return;
    }
    try {
      const vectors = await invoke<NpzVectorData[]>("load_npz_vectors", {
        path,
        keys: ["theta1", "theta2", "bpc_theta1", "bpc_theta2", "bpc_loss"],
      });
      const bpc = resolveBpcMarker(vectors, rows, cols);
      setLossMarkers(bpc ? [bpc] : []);
    } catch {
      setLossMarkers([]);
    }
  }, []);

  const loadPreview = useCallback(async () => {
    if (!archivePath || !selectedKey) return;
    setLoading(true);
    try {
      const slice = await loadSlice(archivePath, effectiveIndices, selectedKey);
      if (!slice) return;
      if (tab === "loss") {
        setLossPreview(slice);
        const rows = slice.values.length;
        const cols = slice.values[0]?.length ?? 0;
        await loadLossMarkers(archivePath, rows, cols);
      } else {
        setPreview(slice);
        if (tab === "attention" && compareMode === "distribution" && distArchivePath) {
          const dist = await loadSlice(distArchivePath, effectiveIndices, selectedKey);
          setDistPreview(dist);
          setComparePreview(null);
        } else if (
          tab === "attention" &&
          compareMode !== "single" &&
          compareMode !== "distribution"
        ) {
          const cmp = await loadSlice(archivePath, compareIndices, selectedKey);
          setComparePreview(cmp);
          setDistPreview(null);
        } else {
          setComparePreview(null);
          setDistPreview(null);
        }
      }
    } catch (err) {
      toast.error("Slice load failed", { description: String(err) });
    } finally {
      setLoading(false);
    }
  }, [
    archivePath,
    selectedKey,
    effectiveIndices,
    compareIndices,
    tab,
    compareMode,
    distArchivePath,
    loadSlice,
    loadLossMarkers,
  ]);

  useEffect(() => {
    if (archivePath && selectedKey) void loadPreview();
  }, [
    archivePath,
    selectedKey,
    effectiveIndices,
    compareIndices,
    tab,
    topK,
    compareMode,
    distArchivePath,
    loadPreview,
  ]);

  useEffect(() => {
    if (!archive || tab !== "loss") return;
    const lossKey = archive.arrays.find((a) =>
      /loss_grid|loss_surface|landscape/i.test(a.key)
    )?.key;
    if (lossKey && lossKey !== selectedKey) setSelectedKey(lossKey);
  }, [archive, tab, selectedKey]);

  useEffect(() => {
    if (
      !projectRoot ||
      tab !== "attention" ||
      (attentionView !== "graph" && attentionView !== "sigma")
    )
      return;
    let cancelled = false;
    void loadGraphCoordinates(projectRoot, graphPreset)
      .then((coords) => {
        if (!cancelled) setGraphCoords(coords);
      })
      .catch(() => {
        if (!cancelled) setGraphCoords(null);
      });
    return () => {
      cancelled = true;
    };
  }, [projectRoot, tab, attentionView, graphPreset]);

  const ingestTensorSlice = useCallback(async () => {
    if (!archivePath || !selectedKey) return;
    setPipelineLoading(true);
    try {
      const timing = await runTensorArrowPipeline(archivePath, {
        key: archivePath.endsWith(".npy") ? null : selectedKey,
        indices: effectiveIndices,
        maxDim: topK,
        tableName: "studio_tensor",
        projectRoot,
        pythonExecutable: pythonPath,
      });
      setTensorPipeline(timing);
      toast.success("Tensor slice ingested", {
        description: `${timing.rowCount} rows in ${timing.totalMs} ms`,
      });
    } catch (err) {
      toast.error("DuckDB ingest failed", { description: String(err) });
    } finally {
      setPipelineLoading(false);
    }
  }, [archivePath, selectedKey, effectiveIndices, topK, projectRoot, pythonPath]);

  const {
    processedValues, heatmapOption, compareHeatmapOption, primaryHeatmapOption,
    attentionGraphOption, lossOption, lossMinima,
  } = useMLIntrospectionOptions({
    preview, comparePreview, distPreview, lossPreview, lossMarkers, tab, clusterK,
    sparseK, compareMode, effectiveIndices, theme, primaryDistLabel, distCompareLabel,
    selectedKey, compareStep, activeAttentionRole, logScale, graphCoords, attentionView, decodeStep,
  });

  const maxDecodeStep = selectedArray && leading >= 1 ? selectedArray.shape[leading - 1] - 1 : 0;

  if (!projectRoot) {
    return <div className="card text-xs text-canvas-muted">Set project root to inspect TensorDict / NPZ archives.</div>;
  }

  return (
    <div className="card space-y-3">
      <div className="flex items-center justify-between flex-wrap gap-2">
        <div>
          <h2 className="text-sm font-semibold text-gray-200">ML Introspection (§G.5)</h2>
          <p className="text-[10px] text-canvas-muted">
            NPZ/NPY/TensorDict (.td) inspector · runtime attention ring-buffer · 3D loss topography
          </p>
        </div>
        <div className="flex items-center gap-1 bg-canvas-elevated rounded-lg p-0.5">
          {(["tensor", "attention", "loss"] as const).map((t) => (
            <button
              key={t}
              onClick={() => setTab(t)}
              className={`text-xs px-2.5 py-1 rounded-md transition-colors capitalize ${
                tab === t ? "bg-accent-primary text-white" : "text-canvas-muted hover:text-gray-200"
              }`}
            >
              {t === "tensor" ? "Archive" : t === "attention" ? "Attention" : "Loss"}
            </button>
          ))}
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <button onClick={() => void pickArchive()} className="btn-primary text-xs flex items-center gap-1">
          <FolderOpen size={12} /> Open .npz / .npy / .td
        </button>
        {archivePath && (
          <button
            onClick={() => archivePath && void inspectArchive(archivePath)}
            disabled={loading}
            className="btn-ghost text-xs flex items-center gap-1"
          >
            <RefreshCw size={12} className={loading ? "animate-spin" : ""} /> Re-inspect
          </button>
        )}
        {archivePath && (
          <OpenPathToolbar path={archivePath} projectRoot={projectRoot} chipClassName="max-w-md" handoff={false} />
        )}
      </div>

      {archive && tab === "tensor" && (
        <MLArchiveTableSection
          archive={archive} selectedKey={selectedKey} setSelectedKey={setSelectedKey}
          setIndices={setIndices} setDecodeStep={setDecodeStep} setHeadIndex={setHeadIndex}
          ingestTensorSlice={ingestTensorSlice} pipelineLoading={pipelineLoading} tensorPipeline={tensorPipeline}
        />
      )}

      {tab === "attention" && (
        <div className="space-y-3">
          <div className="flex flex-wrap items-center gap-2">
            <button onClick={() => void loadAttentionLog()} className="btn-ghost text-xs flex items-center gap-1">
              <FolderOpen size={12} /> Load attention_viz.jsonl
            </button>
            {runtimeAttentionEntries.length > 0 && (
              <span className="text-[10px] text-canvas-muted">
                {runtimeAttentionEntries.length} runtime capture{runtimeAttentionEntries.length !== 1 ? "s" : ""}
              </span>
            )}
          </div>
          <RuntimeAttentionPanel entries={runtimeAttentionEntries} theme={theme} logScale={logScale} />
        </div>
      )}

      {(tab === "attention" || tab === "loss") && archive && (
        <MLAttentionControlsToolbar
          tab={tab} archive={archive} selectedKey={selectedKey} setSelectedKey={setSelectedKey}
          setIndices={setIndices} setDecodeStep={setDecodeStep} setHeadIndex={setHeadIndex}
          roleFilteredKeys={roleFilteredKeys} attentionRole={attentionRole} setAttentionRole={setAttentionRole}
          attentionGroups={attentionGroups} attentionView={attentionView} setAttentionView={setAttentionView}
          graphPreset={graphPreset} setGraphPreset={setGraphPreset} headCount={headCount} headIndex={headIndex}
          leading={leading} indices={indices} headAxis={headAxis} selectedArrayShape={selectedArray?.shape}
          maxDecodeStep={maxDecodeStep} decodeStep={decodeStep} compareMode={compareMode} setCompareMode={setCompareMode}
          pickDistArchive={pickDistArchive} distArchivePath={distArchivePath} archivePath={archivePath}
          primaryDistLabel={primaryDistLabel} distCompareLabel={distCompareLabel} compareStep={compareStep}
          setCompareStep={setCompareStep} sparseK={sparseK} setSparseK={setSparseK} clusterK={clusterK}
          setClusterK={setClusterK} topK={topK} setTopK={setTopK} preview={preview}
        />
      )}

      {tab === "attention" && (
        <MLAttentionVisualSection
          attentionView={attentionView} attentionGraphOption={attentionGraphOption} selectedKey={selectedKey}
          decodeStep={decodeStep} graphPreset={graphPreset} logScale={logScale} processedValues={processedValues}
          graphCoords={graphCoords} sparseK={sparseK} theme={theme} heatmapOption={heatmapOption}
          compareMode={compareMode} compareHeatmapOption={compareHeatmapOption}
          primaryHeatmapOption={primaryHeatmapOption} distCompareLabel={distCompareLabel}
          compareStep={compareStep} activeAttentionRole={activeAttentionRole}
        />
      )}

      {tab === "loss" && (
        <MLLossLandscapeSection
          lossPreview={lossPreview} lossView={lossView} setLossView={setLossView}
          lossMarkers={lossMarkers} lossOption={lossOption} lossMinima={lossMinima} logScale={logScale}
        />
      )}
    </div>
  );
}
