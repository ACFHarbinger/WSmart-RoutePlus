import { FolderOpen } from "lucide-react";
import { GRAPH_PRESETS } from "../../../utils/graph/graphCoords";
import {
  defaultIndices,
  type AttentionRole,
} from "../../../utils/charts/tensorHeatmap";
import type { ArrowPipelineTiming } from "../../../utils/duckdb/arrowPipeline";
import type { NpzArchiveInfo, TensorSlicePreview } from "../../../types";

export type IntrospectionTab = "tensor" | "attention" | "loss";
export type CompareMode = "single" | "side-by-side" | "overlay" | "distribution";
export type AttentionView = "heatmap" | "graph" | "sigma";

export function formatBytes(b: number): string {
  if (b < 1024) return `${b} B`;
  if (b < 1024 ** 2) return `${(b / 1024).toFixed(1)} KB`;
  return `${(b / 1024 ** 2).toFixed(1)} MB`;
}

export function MLArchiveTableSection({
  archive,
  selectedKey,
  setSelectedKey,
  setIndices,
  setDecodeStep,
  setHeadIndex,
  ingestTensorSlice,
  pipelineLoading,
  tensorPipeline,
}: {
  archive: NpzArchiveInfo;
  selectedKey: string;
  setSelectedKey: (k: string) => void;
  setIndices: (idx: number[]) => void;
  setDecodeStep: (step: number) => void;
  setHeadIndex: (head: number) => void;
  ingestTensorSlice: () => void;
  pipelineLoading: boolean;
  tensorPipeline: ArrowPipelineTiming | null;
}) {
  return (
    <div className="overflow-auto">
      <table className="w-full text-xs">
        <thead>
          <tr className="border-b border-canvas-border">
            <th className="text-left py-2 px-2 text-canvas-muted">Key</th>
            <th className="text-left py-2 px-2 text-canvas-muted">Shape</th>
            <th className="text-left py-2 px-2 text-canvas-muted">Dtype</th>
            <th className="text-right py-2 px-2 text-canvas-muted">Size</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-canvas-border">
          {archive.arrays.map((a) => (
            <tr
              key={a.key}
              className={`hover:bg-canvas-hover cursor-pointer ${
                selectedKey === a.key ? "bg-canvas-hover/60" : ""
              }`}
              onClick={() => {
                setSelectedKey(a.key);
                setIndices(defaultIndices(a.shape));
                setDecodeStep(0);
                setHeadIndex(0);
              }}
            >
              <td className="py-2 px-2 font-mono text-gray-300">{a.key}</td>
              <td className="py-2 px-2 font-mono text-canvas-muted">
                [{a.shape.join(", ")}]
              </td>
              <td className="py-2 px-2 text-canvas-muted">{a.dtype}</td>
              <td className="py-2 px-2 text-right text-canvas-muted tabular-nums">
                {formatBytes(a.size_bytes)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="flex flex-wrap items-center gap-2 mt-3">
        <button
          onClick={() => void ingestTensorSlice()}
          disabled={!selectedKey || pipelineLoading}
          className="btn-primary text-xs"
        >
          {pipelineLoading ? "Ingesting…" : "Ingest slice → DuckDB"}
        </button>
        {tensorPipeline && (
          <span className="text-[10px] text-canvas-muted font-mono tabular-nums">
            {tensorPipeline.tableName}: {tensorPipeline.rowCount} rows ·{" "}
            {tensorPipeline.totalMs} ms
            {tensorPipeline.withinBudget ? " ✓" : " (budget)"}
          </span>
        )}
      </div>
      {tensorPipeline && (
        <p className="text-[10px] text-canvas-muted mt-1 font-mono">
          SELECT row, col, value FROM &quot;{tensorPipeline.tableName}&quot; ORDER BY
          value DESC LIMIT 20;
        </p>
      )}
      <p className="text-[10px] text-canvas-muted mt-2">
        Total payload {formatBytes(archive.total_bytes)}
        {archive.used_memmap
          ? " · large .npy/.npz eligible for mmap / decompress slice"
          : ""}
      </p>
    </div>
  );
}

export function MLAttentionControlsToolbar({
  tab,
  archive,
  selectedKey,
  setSelectedKey,
  setIndices,
  setDecodeStep,
  setHeadIndex,
  roleFilteredKeys,
  attentionRole,
  setAttentionRole,
  attentionGroups,
  attentionView,
  setAttentionView,
  graphPreset,
  setGraphPreset,
  headCount,
  headIndex,
  leading,
  indices,
  headAxis,
  selectedArrayShape,
  maxDecodeStep,
  decodeStep,
  compareMode,
  setCompareMode,
  pickDistArchive,
  distArchivePath,
  archivePath,
  primaryDistLabel,
  distCompareLabel,
  compareStep,
  setCompareStep,
  sparseK,
  setSparseK,
  clusterK,
  setClusterK,
  topK,
  setTopK,
  preview,
}: {
  tab: IntrospectionTab;
  archive: NpzArchiveInfo;
  selectedKey: string;
  setSelectedKey: (k: string) => void;
  setIndices: (fn: number[] | ((prev: number[]) => number[])) => void;
  setDecodeStep: (step: number) => void;
  setHeadIndex: (head: number) => void;
  roleFilteredKeys: string[];
  attentionRole: AttentionRole | "all";
  setAttentionRole: (role: AttentionRole | "all") => void;
  attentionGroups: Record<AttentionRole, string[]> | null;
  attentionView: AttentionView;
  setAttentionView: (v: AttentionView) => void;
  graphPreset: string;
  setGraphPreset: (p: string) => void;
  headCount: number;
  headIndex: number;
  leading: number;
  indices: number[];
  headAxis: number | null;
  selectedArrayShape?: number[];
  maxDecodeStep: number;
  decodeStep: number;
  compareMode: CompareMode;
  setCompareMode: (m: CompareMode) => void;
  pickDistArchive: () => void;
  distArchivePath: string | null;
  archivePath: string | null;
  primaryDistLabel: string;
  distCompareLabel: string;
  compareStep: number;
  setCompareStep: (s: number) => void;
  sparseK: number;
  setSparseK: (k: number) => void;
  clusterK: number;
  setClusterK: (k: number) => void;
  topK: number;
  setTopK: (k: number) => void;
  preview: TensorSlicePreview | null;
}) {
  return (
    <div className="flex flex-wrap items-center gap-3 text-xs">
      <label className="flex items-center gap-1.5 text-canvas-muted">
        Array
        <select
          className="select-base text-xs py-0.5 max-w-[200px]"
          value={selectedKey}
          onChange={(e) => {
            const key = e.target.value;
            setSelectedKey(key);
            const shape = archive.arrays.find((a) => a.key === key)?.shape ?? [];
            setIndices(defaultIndices(shape));
            setDecodeStep(0);
            setHeadIndex(0);
          }}
        >
          {(tab === "attention"
            ? roleFilteredKeys
            : archive.arrays
                .filter(
                  (a) =>
                    /loss_grid|loss_surface|landscape/i.test(a.key) ||
                    a.shape.length === 2
                )
                .map((a) => a.key)
          ).map((k) => (
            <option key={k} value={k}>
              {k}
            </option>
          ))}
        </select>
      </label>

      {tab === "attention" && (
        <>
          <label className="flex items-center gap-1 text-canvas-muted">
            Q/K/V
            <select
              className="select-base text-xs py-0.5"
              value={attentionRole}
              onChange={(e) => {
                const role = e.target.value as AttentionRole | "all";
                setAttentionRole(role);
                if (role !== "all" && attentionGroups?.[role]?.[0]) {
                  const key = attentionGroups[role][0];
                  setSelectedKey(key);
                  const shape = archive.arrays.find((a) => a.key === key)?.shape ?? [];
                  setIndices(defaultIndices(shape));
                  setDecodeStep(0);
                  setHeadIndex(0);
                }
              }}
            >
              <option value="all">All</option>
              <option value="query">
                Query ({attentionGroups?.query.length ?? 0})
              </option>
              <option value="key">Key ({attentionGroups?.key.length ?? 0})</option>
              <option value="value">
                Value ({attentionGroups?.value.length ?? 0})
              </option>
              <option value="weights">
                Weights ({attentionGroups?.weights.length ?? 0})
              </option>
            </select>
          </label>
          <label className="flex items-center gap-1 text-canvas-muted">
            View
            <select
              className="select-base text-xs py-0.5"
              value={attentionView}
              onChange={(e) => setAttentionView(e.target.value as AttentionView)}
            >
              <option value="heatmap">Heatmap</option>
              <option value="graph">Graph on coords (ECharts)</option>
              <option value="sigma">Graph on coords (Sigma.js)</option>
            </select>
          </label>
          {(attentionView === "graph" || attentionView === "sigma") && (
            <label className="flex items-center gap-1 text-canvas-muted">
              Graph
              <select
                className="select-base text-xs py-0.5"
                value={graphPreset}
                onChange={(e) => setGraphPreset(e.target.value)}
              >
                {GRAPH_PRESETS.map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.label}
                  </option>
                ))}
              </select>
            </label>
          )}
        </>
      )}

      {tab === "attention" && headCount > 1 && (
        <label className="flex items-center gap-1 text-canvas-muted">
          Head
          <select
            className="select-base text-xs py-0.5 w-14"
            value={headIndex}
            onChange={(e) => setHeadIndex(Number(e.target.value))}
          >
            {Array.from({ length: headCount }, (_, i) => (
              <option key={i} value={i}>
                {i}
              </option>
            ))}
          </select>
        </label>
      )}

      {tab === "attention" &&
        leading > 1 &&
        indices.slice(0, leading - 1).map((_, dim) =>
          dim === headAxis ? null : (
            <label key={dim} className="flex items-center gap-1 text-canvas-muted">
              d{dim}
              <input
                type="number"
                min={0}
                max={(selectedArrayShape?.[dim] ?? 1) - 1}
                value={indices[dim] ?? 0}
                onChange={(e) => {
                  const v = Number(e.target.value);
                  setIndices((prev) => {
                    const next = [...prev];
                    next[dim] = v;
                    return next;
                  });
                }}
                className="input-base w-14 text-xs py-0.5"
              />
            </label>
          )
        )}

      {tab === "attention" && leading >= 1 && (
        <div className="flex items-center gap-2 flex-1 min-w-[180px]">
          <span className="text-canvas-muted">Decode step</span>
          <input
            type="range"
            min={0}
            max={Math.max(0, maxDecodeStep)}
            value={decodeStep}
            onChange={(e) => setDecodeStep(Number(e.target.value))}
            className="flex-1 accent-accent-primary"
          />
          <span className="font-mono text-gray-300 w-8 tabular-nums">
            {decodeStep}
          </span>
        </div>
      )}

      {tab === "attention" && (
        <>
          <label className="flex items-center gap-1 text-canvas-muted">
            Compare
            <select
              className="select-base text-xs py-0.5"
              value={compareMode}
              onChange={(e) => setCompareMode(e.target.value as CompareMode)}
            >
              <option value="single">Single</option>
              <option value="side-by-side">Side-by-side</option>
              <option value="overlay">Overlay Δ</option>
              <option value="distribution">Empirical vs Gamma-3</option>
            </select>
          </label>
          {compareMode === "distribution" && (
            <>
              <button
                onClick={() => void pickDistArchive()}
                className="btn-ghost text-xs flex items-center gap-1"
              >
                <FolderOpen size={12} />
                {distArchivePath ? "Change compare archive" : "Open compare .npz"}
              </button>
              {archivePath && (
                <span className="text-canvas-muted">
                  {primaryDistLabel} vs {distCompareLabel}
                </span>
              )}
            </>
          )}
          {compareMode !== "single" && compareMode !== "distribution" && (
            <div className="flex items-center gap-2 min-w-[140px]">
              <span className="text-canvas-muted">vs step</span>
              <input
                type="range"
                min={0}
                max={Math.max(0, maxDecodeStep)}
                value={compareStep}
                onChange={(e) => setCompareStep(Number(e.target.value))}
                className="flex-1 accent-accent-warning"
              />
              <span className="font-mono text-gray-300 w-8 tabular-nums">
                {compareStep}
              </span>
            </div>
          )}
          <label className="flex items-center gap-1 text-canvas-muted">
            Sparse top-k
            <select
              className="select-base text-xs py-0.5 w-16"
              value={sparseK}
              onChange={(e) => setSparseK(Number(e.target.value))}
            >
              <option value={0}>Off</option>
              {[4, 8, 16, 32].map((k) => (
                <option key={k} value={k}>
                  {k}
                </option>
              ))}
            </select>
          </label>
          <label className="flex items-center gap-1 text-canvas-muted">
            K-means
            <select
              className="select-base text-xs py-0.5 w-16"
              value={clusterK}
              onChange={(e) => setClusterK(Number(e.target.value))}
            >
              <option value={0}>Off</option>
              {[2, 3, 4, 6, 8].map((k) => (
                <option key={k} value={k}>
                  {k}
                </option>
              ))}
            </select>
          </label>
          <label className="flex items-center gap-1 text-canvas-muted">
            Matrix cap
            <select
              className="select-base text-xs py-0.5 w-16"
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value))}
            >
              {[24, 32, 48, 64].map((k) => (
                <option key={k} value={k}>
                  {k}
                </option>
              ))}
            </select>
          </label>
        </>
      )}

      {preview && tab === "attention" && (
        <span className="text-canvas-muted tabular-nums">
          {preview.rust_ms} ms · downsampled
          {preview.used_memmap
            ? " · mmap slice"
            : preview.used_decompress_slice
            ? " · decompress slice"
            : ""}
        </span>
      )}
    </div>
  );
}
