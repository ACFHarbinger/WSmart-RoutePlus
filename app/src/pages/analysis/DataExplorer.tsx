/**
 * Data Explorer — browse and inspect simulation output CSV files.
 * Ports Streamlit `data_explorer` mode.
 */
import { useCallback, useEffect, useMemo, useState } from "react";
import { ChevronDown, ChevronUp, FolderOpen, Download } from "lucide-react";
import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import { toast } from "sonner";
import { OpenPathToolbar } from "../../components/common/OpenPathToolbar";
import { PolicyTelemetryTrendsPanel } from "../../components/analysis/telemetry/PolicyTelemetryTrendsPanel";
import { SqlQueryPanel } from "../../components/analysis/explorer/SqlQueryPanel";
import { GlobalFilterBar } from "../../components/layout/GlobalFilterBar";
import { useLogPathRunLabelBrush } from "../../hooks/brush/useLogPathRunLabelBrush";
import { useTableRunLabelBrush } from "../../hooks/brush/useTableRunLabelBrush";
import { useAppStore } from "../../store/app";
import { useGlobalFiltersStore } from "../../store/filters";
import { useRecentHandoff } from "../../hooks/files/useRecentHandoff";
import { downloadCsv, downloadParquetFromCsv } from "../../utils/charts/tableExport";
import { formatPipelineTimingBadge, portfolioRunLabel, runCsvArrowPipeline } from "../../utils/duckdb/arrowPipeline";
import { groupRunLabelsByCity, resolveBrushedRunLabels } from "../../utils/benchmark/cityComparison";
import { useDuckDbStore } from "../../store/duckdb";

interface CsvRow {
  [key: string]: string | number | null;
}

interface CsvFile {
  path: string;
  headers: string[];
  rows: CsvRow[];
}

function headerCol(headers: string[], pattern: RegExp): string | undefined {
  return headers.find((h) => pattern.test(h));
}

function distinctColumnValues(rows: CsvRow[], col: string): string[] {
  return [...new Set(rows.map((r) => String(r[col] ?? "")).filter(Boolean))].sort();
}

export function DataExplorer() {
  const { pendingCsvPath, setPendingCsvPath, effectiveTheme: theme } = useAppStore();
  const { projectRoot, handoff } = useRecentHandoff();
  const {
    policy: activePolicy,
    runLabel: activeRunLabel,
    brushedCity,
    logScale,
    setPolicy,
    setRunLabel,
    setBrushedCity,
  } = useGlobalFiltersStore();
  const { ready: duckdbReady, lastPipeline, setLastPipeline, setLoading, loading } =
    useDuckDbStore();
  const [file, setFile] = useState<CsvFile | null>(null);
  const [exporting, setExporting] = useState(false);
  const [page, setPage] = useState(0);
  const [sortCol, setSortCol] = useState<string | null>(null);
  const [sortDir, setSortDir] = useState<"asc" | "desc">("asc");
  const [filterText, setFilterText] = useState("");
  const PAGE_SIZE = 50;

  const loadCsv = useCallback(
    async (path: string) => {
      const loaded = await invoke<CsvFile>("load_csv_file", { path });
      setFile(loaded);
      setPage(0);
      setSortCol(null);
      setSortDir("asc");
      setFilterText("");
      handoff(path, "csv", { navigate: false });

      if (duckdbReady) {
        setLoading(true);
        try {
          const timing = await runCsvArrowPipeline(path, "explorer_csv", projectRoot);
          setLastPipeline(timing);
        } catch (err) {
          console.warn("DuckDB ingest failed:", err);
        } finally {
          setLoading(false);
        }
      }
    },
    [handoff, duckdbReady, projectRoot, setLastPipeline, setLoading]
  );

  const openCsv = useCallback(async () => {
    const path = (await open({
      filters: [{ name: "CSV Files", extensions: ["csv"] }],
    })) as string | null;
    if (!path) return;
    await loadCsv(path);
  }, [loadCsv]);

  useEffect(() => {
    if (!pendingCsvPath) return;
    void loadCsv(pendingCsvPath);
    setPendingCsvPath(null);
  }, [pendingCsvPath, setPendingCsvPath, loadCsv]);

  const policyCol = useMemo(
    () => (file ? headerCol(file.headers, /^policy$/i) : undefined),
    [file]
  );
  const runLabelCol = useMemo(
    () => (file ? headerCol(file.headers, /^run_label$/i) : undefined),
    [file]
  );
  const cityScaleCol = useMemo(
    () => (file ? headerCol(file.headers, /^city_scale$/i) : undefined),
    [file]
  );

  const csvPolicies = useMemo(
    () => (file && policyCol ? distinctColumnValues(file.rows, policyCol) : []),
    [file, policyCol]
  );
  const csvRunLabels = useMemo(
    () => (file && runLabelCol ? distinctColumnValues(file.rows, runLabelCol) : []),
    [file, runLabelCol]
  );
  useLogPathRunLabelBrush(file?.path ?? null);
  const sourceRunLabel = useMemo(
    () => (file ? portfolioRunLabel(file.path, undefined, projectRoot) : null),
    [file, projectRoot]
  );
  const derivedTableRunLabel = useTableRunLabelBrush(
    lastPipeline?.tableName === "explorer_csv" ? "explorer_csv" : null,
    csvRunLabels,
    Boolean(runLabelCol)
  );
  const csvCities = useMemo(() => {
    if (!file) return [];
    if (cityScaleCol) return distinctColumnValues(file.rows, cityScaleCol);
    if (runLabelCol) return groupRunLabelsByCity(csvRunLabels).map(([city]) => city);
    return [];
  }, [file, cityScaleCol, runLabelCol, csvRunLabels]);

  const hasBrushColumns = Boolean(policyCol || runLabelCol || cityScaleCol);

  const highlightPolicies = useMemo(
    () => (activePolicy ? [activePolicy] : null),
    [activePolicy]
  );
  const highlightRunLabels = useMemo(
    () =>
      resolveBrushedRunLabels(csvRunLabels, activeRunLabel, brushedCity) ??
      (activeRunLabel ? [activeRunLabel] : null),
    [csvRunLabels, activeRunLabel, brushedCity]
  );

  const rowMatchesHighlight = useCallback(
    (row: CsvRow) => {
      const policyMatch =
        !highlightPolicies?.length ||
        !policyCol ||
        highlightPolicies.includes(String(row[policyCol] ?? ""));
      const runMatch =
        !highlightRunLabels?.length ||
        !runLabelCol ||
        highlightRunLabels.includes(String(row[runLabelCol] ?? ""));
      const cityMatch =
        !brushedCity ||
        (cityScaleCol
          ? String(row[cityScaleCol] ?? "") === brushedCity
          : !runLabelCol ||
            (resolveBrushedRunLabels(csvRunLabels, null, brushedCity)?.includes(
              String(row[runLabelCol] ?? "")
            ) ??
              false));
      return policyMatch && runMatch && cityMatch;
    },
    [
      highlightPolicies,
      highlightRunLabels,
      brushedCity,
      policyCol,
      runLabelCol,
      cityScaleCol,
      csvRunLabels,
    ]
  );

  const applyCrossFilter = useCallback(
    (col: string, value: string) => {
      if (/^policy$/i.test(col)) {
        setPolicy(value);
        toast.success("Cross-filter applied", { description: `Policy: ${value}` });
        return;
      }
      if (/^run_label$/i.test(col)) {
        setBrushedCity(null);
        setRunLabel(value);
        toast.success("Cross-filter applied", { description: `Run: ${value}` });
        return;
      }
      if (/^city_scale$/i.test(col)) {
        setRunLabel(null);
        setBrushedCity(value);
        toast.success("Cross-filter applied", { description: `City: ${value}` });
      }
    },
    [setPolicy, setRunLabel, setBrushedCity]
  );

  const filteredRows = useMemo(() => {
    if (!file) return [];
    const q = filterText.trim().toLowerCase();
    if (!q) return file.rows;
    return file.rows.filter((row) =>
      file.headers.some((h) => String(row[h] ?? "").toLowerCase().includes(q))
    );
  }, [file, filterText]);

  const sortedRows = useMemo(() => {
    if (!file) return [];
    if (!sortCol) return filteredRows;
    const rows = [...filteredRows];
    rows.sort((a, b) => {
      const av = a[sortCol];
      const bv = b[sortCol];
      const an = typeof av === "number" ? av : Number(av);
      const bn = typeof bv === "number" ? bv : Number(bv);
      const bothNumeric = !Number.isNaN(an) && !Number.isNaN(bn) && av !== "" && bv !== "";
      const cmp = bothNumeric
        ? an - bn
        : String(av ?? "").localeCompare(String(bv ?? ""));
      return sortDir === "asc" ? cmp : -cmp;
    });
    return rows;
  }, [file, filteredRows, sortCol, sortDir]);

  const hasActiveBrush = Boolean(activePolicy || activeRunLabel || brushedCity);

  const exportRows = useMemo(() => {
    if (!hasBrushColumns || !hasActiveBrush) return sortedRows;
    return sortedRows.filter(rowMatchesHighlight);
  }, [sortedRows, hasBrushColumns, hasActiveBrush, rowMatchesHighlight]);

  const pageRows = sortedRows.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);
  const totalPages = file ? Math.ceil(sortedRows.length / PAGE_SIZE) : 0;

  const toggleSort = (col: string) => {
    if (sortCol === col) setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    else {
      setSortCol(col);
      setSortDir("asc");
    }
    setPage(0);
  };

  return (
    <div className="space-y-4">
      {hasBrushColumns && (
        <GlobalFilterBar
          policies={csvPolicies}
          runLabels={
            csvRunLabels.length > 0
              ? csvRunLabels
              : sourceRunLabel
                ? [sourceRunLabel]
                : derivedTableRunLabel
                  ? [derivedTableRunLabel]
                  : []
          }
          cities={csvCities.length > 1 ? csvCities : []}
          showLogScale
        />
      )}

      <div className="flex items-center gap-3">
        <button onClick={openCsv} className="btn-primary flex items-center gap-2">
          <FolderOpen size={14} />
          Open CSV
        </button>
        {file && (
          <OpenPathToolbar
            path={file.path}
            projectRoot={projectRoot}
            kind="csv"
            trailing={
              <span className="shrink-0">
                · {file.rows.length.toLocaleString()} rows
                {hasActiveBrush && hasBrushColumns && (
                  <> · {exportRows.length.toLocaleString()} brushed</>
                )}
                {loading && " · DuckDB ingesting…"}
                {!loading && lastPipeline?.tableName === "explorer_csv" && (
                  <> · {formatPipelineTimingBadge(lastPipeline)}</>
                )}
              </span>
            }
          >
            <button
              onClick={() =>
                downloadCsv(
                  file.path.split("/").pop() ?? "data.csv",
                  file.headers,
                  exportRows.map((row) => file.headers.map((h) => row[h] ?? ""))
                )
              }
              className="btn-ghost text-xs flex items-center gap-1.5"
            >
              <Download size={12} />
              Export CSV
            </button>
            {projectRoot && (
              <button
                disabled={exporting}
                onClick={async () => {
                  setExporting(true);
                  try {
                    const out = await downloadParquetFromCsv(
                      projectRoot,
                      file.path,
                      file.path.replace(/\.csv$/i, ".parquet")
                    );
                    if (out) toast.success("Parquet export complete", { description: out.split("/").pop() });
                  } catch (err) {
                    toast.error("Parquet export failed", { description: String(err) });
                  } finally {
                    setExporting(false);
                  }
                }}
                className="btn-ghost text-xs flex items-center gap-1.5"
              >
                <Download size={12} />
                Export Parquet
              </button>
            )}
          </OpenPathToolbar>
        )}
      </div>

      {!file && (
        <div className="flex items-center justify-center h-48 text-canvas-muted text-sm">
          Open a CSV file from the output directory to explore.
        </div>
      )}

      {file && (
        <PolicyTelemetryTrendsPanel
          theme={theme}
          logScale={logScale}
          initialPolicy={activePolicy}
          initialRunLabel={
            activeRunLabel ??
            (csvRunLabels.length === 1 ? csvRunLabels[0]! : null) ??
            sourceRunLabel ??
            derivedTableRunLabel
          }
        />
      )}

      {file && duckdbReady && lastPipeline?.tableName && (
        <SqlQueryPanel
          tableName={lastPipeline.tableName}
          theme={theme}
          highlightPolicies={highlightPolicies}
          highlightRunLabels={
            highlightRunLabels ??
            (derivedTableRunLabel
              ? [derivedTableRunLabel]
              : sourceRunLabel
                ? [sourceRunLabel]
                : null)
          }
          brushSqlSync={hasBrushColumns || Boolean(derivedTableRunLabel || sourceRunLabel)}
          autoRunOnBrushSync={hasBrushColumns || Boolean(derivedTableRunLabel || sourceRunLabel)}
          portfolioMode={Boolean(runLabelCol) || Boolean(derivedTableRunLabel || sourceRunLabel)}
          portfolioRunLabels={
            csvRunLabels.length > 0
              ? csvRunLabels
              : derivedTableRunLabel
                ? [derivedTableRunLabel]
                : sourceRunLabel
                  ? [sourceRunLabel]
                  : []
          }
        />
      )}

      {file && (
        <>
          <div className="flex items-center gap-3">
            <input
              type="search"
              className="input-base text-xs flex-1 max-w-sm"
              placeholder="Filter rows (matches any column)…"
              value={filterText}
              onChange={(e) => {
                setFilterText(e.target.value);
                setPage(0);
              }}
            />
            {(filterText || (hasActiveBrush && hasBrushColumns)) && (
              <span className="text-xs text-canvas-muted">
                {sortedRows.length.toLocaleString()} / {file.rows.length.toLocaleString()} rows
                {hasActiveBrush && hasBrushColumns && (
                  <> · {exportRows.length.toLocaleString()} exportable</>
                )}
              </span>
            )}
          </div>
          <div className="overflow-auto rounded-xl border border-canvas-border">
            <table className="w-full text-xs">
              <thead className="bg-canvas-elevated sticky top-0">
                <tr>
                  {file.headers.map((h) => (
                    <th
                      key={h}
                      onClick={() => toggleSort(h)}
                      className="px-3 py-2 text-left text-canvas-muted font-medium whitespace-nowrap cursor-pointer hover:text-gray-200 select-none"
                    >
                      <span className="inline-flex items-center gap-1">
                        {h}
                        {sortCol === h &&
                          (sortDir === "asc" ? <ChevronUp size={10} /> : <ChevronDown size={10} />)}
                      </span>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-canvas-border">
                {pageRows.map((row, i) => {
                  const highlighted = hasBrushColumns ? rowMatchesHighlight(row) : true;
                  const isActive =
                    (policyCol &&
                      activePolicy &&
                      String(row[policyCol] ?? "") === activePolicy) ||
                    (runLabelCol &&
                      activeRunLabel &&
                      String(row[runLabelCol] ?? "") === activeRunLabel) ||
                    (cityScaleCol &&
                      brushedCity &&
                      String(row[cityScaleCol] ?? "") === brushedCity);
                  return (
                    <tr
                      key={i}
                      className={`hover:bg-canvas-hover ${
                        isActive
                          ? "bg-accent-primary/15 ring-1 ring-inset ring-accent-primary/40"
                          : highlighted
                            ? ""
                            : hasBrushColumns
                              ? "opacity-35"
                              : ""
                      }`}
                    >
                      {file.headers.map((h) => {
                        const isBrushCol =
                          hasBrushColumns &&
                          (h === policyCol || h === runLabelCol || h === cityScaleCol);
                        return (
                          <td
                            key={h}
                            onClick={
                              isBrushCol
                                ? () => applyCrossFilter(h, String(row[h] ?? ""))
                                : undefined
                            }
                            className={`px-3 py-1.5 text-gray-300 whitespace-nowrap font-mono ${
                              isBrushCol ? "cursor-pointer hover:text-accent-secondary" : ""
                            }`}
                          >
                            {row[h] ?? "—"}
                          </td>
                        );
                      })}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          {totalPages > 1 && (
            <div className="flex items-center gap-3 text-xs text-canvas-muted">
              <button className="btn-ghost py-1 px-2" disabled={page === 0} onClick={() => setPage((p) => p - 1)}>
                ← Prev
              </button>
              <span className="tabular-nums">Page {page + 1} / {totalPages}</span>
              <button className="btn-ghost py-1 px-2" disabled={page >= totalPages - 1} onClick={() => setPage((p) => p + 1)}>
                Next →
              </button>
            </div>
          )}
        </>
      )}
    </div>
  );
}
