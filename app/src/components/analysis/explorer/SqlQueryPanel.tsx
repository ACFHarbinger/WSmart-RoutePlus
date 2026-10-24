/**
 * DuckDB-Wasm SQL query editor panel (§G.6).
 */
import { lazy, Suspense, useCallback, useEffect, useMemo, useState } from "react";
import { ChevronDown, ChevronUp, Download, Play } from "lucide-react";
import { toast } from "sonner";
import { useGlobalFiltersStore } from "../../../store/filters";
import { resolveBrushedRunLabels } from "../../../utils/benchmark/cityComparison";
import { queryDuckDb } from "../../../utils/duckdb/duckdbClient";
import {
  brushedPortfolioSql,
  sqlTemplates,
  type PortfolioBrushFilter,
} from "../../../utils/duckdb/duckdbTemplates";
import {
  buildAutoChartOption,
  suggestChartAlternatives,
  type AutoChartSpec,
  type AutoChartType,
} from "../../../utils/charts/queryAutoChart";
import { downloadCsv } from "../../../utils/charts/tableExport";
import { SqlQueryResultsView } from "./SqlQueryResultsView";

const MonacoEditor = lazy(() => import("@monaco-editor/react"));

interface Props {
  tableName: string;
  theme: "dark" | "light";
  onDaySelect?: (day: number) => void;
  onProfitRange?: (min: number, max: number) => void;
  highlightPolicies?: string[] | null;
  highlightRunLabels?: string[] | null;
  brushSqlSync?: boolean;
  autoRunOnBrushSync?: boolean;
  defaultOpen?: boolean;
  portfolioMode?: boolean;
  algorithmMode?: boolean;
  portfolioRunLabels?: string[];
}

export function SqlQueryPanel({
  tableName,
  theme,
  onDaySelect,
  onProfitRange,
  highlightPolicies: highlightPoliciesProp = null,
  highlightRunLabels: highlightRunLabelsProp = null,
  brushSqlSync = false,
  autoRunOnBrushSync = false,
  defaultOpen = false,
  portfolioMode = false,
  algorithmMode = false,
  portfolioRunLabels = [],
}: Props) {
  const activePolicy = useGlobalFiltersStore((s) => s.policy);
  const activeRunLabel = useGlobalFiltersStore((s) => s.runLabel);
  const brushedCity = useGlobalFiltersStore((s) => s.brushedCity);
  const setPolicy = useGlobalFiltersStore((s) => s.setPolicy);
  const setRunLabel = useGlobalFiltersStore((s) => s.setRunLabel);
  const setBrushedCity = useGlobalFiltersStore((s) => s.setBrushedCity);
  const [open, setOpen] = useState(defaultOpen);
  const [sql, setSql] = useState(`SELECT * FROM "${tableName}" LIMIT 100`);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [rows, setRows] = useState<Record<string, unknown>[]>([]);
  const [sortCol, setSortCol] = useState<string | null>(null);
  const [sortDir, setSortDir] = useState<"asc" | "desc">("asc");
  const [filterText, setFilterText] = useState("");
  const [chartTypeOverride, setChartTypeOverride] = useState<AutoChartType | null>(null);
  const logScale = useGlobalFiltersStore((s) => s.logScale);

  const templates = useMemo(
    () => sqlTemplates(tableName, { portfolio: portfolioMode, algorithm: algorithmMode }),
    [tableName, portfolioMode, algorithmMode]
  );

  const brushFilter = useMemo((): PortfolioBrushFilter => {
    const filter: PortfolioBrushFilter = {};
    if (highlightPoliciesProp?.length) {
      filter.policies = highlightPoliciesProp;
    } else if (activePolicy) {
      filter.policies = [activePolicy];
    }
    if (highlightRunLabelsProp?.length) {
      filter.runLabels = highlightRunLabelsProp;
    } else if (brushedCity) {
      const expanded = resolveBrushedRunLabels(
        portfolioRunLabels,
        null,
        brushedCity
      );
      if (expanded?.length) {
        filter.runLabels = expanded;
      } else {
        filter.cityScale = brushedCity;
      }
    } else if (activeRunLabel) {
      filter.runLabels = [activeRunLabel];
    }
    return filter;
  }, [
    highlightPoliciesProp,
    highlightRunLabelsProp,
    activePolicy,
    activeRunLabel,
    brushedCity,
    portfolioRunLabels,
  ]);

  const hasBrushFilter = Boolean(
    brushFilter.policies?.length || brushFilter.runLabels?.length || brushFilter.cityScale
  );

  useEffect(() => {
    if (!brushSqlSync) return;
    setSql(brushedPortfolioSql(tableName, brushFilter));
    if (hasBrushFilter) setOpen(true);
  }, [brushSqlSync, brushFilter, hasBrushFilter, tableName]);

  useEffect(() => {
    if (!brushSqlSync || !autoRunOnBrushSync || !hasBrushFilter) return;
    const nextSql = brushedPortfolioSql(tableName, brushFilter);
    void (async () => {
      setRunning(true);
      setError(null);
      try {
        const result = await queryDuckDb<Record<string, unknown>>(nextSql);
        setRows(result);
        setSortCol(null);
        setFilterText("");
      } catch (err) {
        setError(String(err));
        setRows([]);
      } finally {
        setRunning(false);
      }
    })();
  }, [brushSqlSync, autoRunOnBrushSync, brushFilter, hasBrushFilter, tableName]);

  const columns = useMemo(() => {
    if (!rows.length) return [];
    return Object.keys(rows[0]);
  }, [rows]);

  const chartAlternatives = useMemo(
    () => (columns.length ? suggestChartAlternatives(columns, rows) : []),
    [columns, rows]
  );

  const chartSpec = useMemo((): AutoChartSpec | null => {
    if (!chartAlternatives.length) return null;
    if (chartTypeOverride) {
      return chartAlternatives.find((s) => s.type === chartTypeOverride) ?? chartAlternatives[0];
    }
    return chartAlternatives[0];
  }, [chartAlternatives, chartTypeOverride]);

  useEffect(() => {
    setChartTypeOverride(null);
  }, [rows]);

  const chartOption = useMemo(
    () =>
      chartSpec
        ? buildAutoChartOption(chartSpec, rows, { logScale })
        : null,
    [chartSpec, rows, logScale]
  );

  const sortedRows = useMemo(() => {
    if (!sortCol) return rows;
    return [...rows].sort((a, b) => {
      const av = a[sortCol];
      const bv = b[sortCol];
      const an = typeof av === "number" ? av : Number(av);
      const bn = typeof bv === "number" ? bv : Number(bv);
      const bothNumeric = !Number.isNaN(an) && !Number.isNaN(bn);
      const cmp = bothNumeric
        ? an - bn
        : String(av ?? "").localeCompare(String(bv ?? ""));
      return sortDir === "asc" ? cmp : -cmp;
    });
  }, [rows, sortCol, sortDir]);

  const filteredRows = useMemo(() => {
    const q = filterText.trim().toLowerCase();
    if (!q) return sortedRows;
    return sortedRows.filter((row) =>
      columns.some((c) => String(row[c] ?? "").toLowerCase().includes(q))
    );
  }, [sortedRows, filterText, columns]);

  const runQuery = useCallback(async () => {
    setRunning(true);
    setError(null);
    try {
      const result = await queryDuckDb<Record<string, unknown>>(sql);
      setRows(result);
      setSortCol(null);
      setFilterText("");
    } catch (err) {
      setError(String(err));
      setRows([]);
    } finally {
      setRunning(false);
    }
  }, [sql]);

  const exportResults = useCallback(() => {
    if (!columns.length) return;
    downloadCsv(
      "duckdb-query.csv",
      columns,
      filteredRows.map((row) => columns.map((c) => String(row[c] ?? "")))
    );
  }, [columns, filteredRows]);

  const toggleSort = (col: string) => {
    if (sortCol === col) setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    else {
      setSortCol(col);
      setSortDir("asc");
    }
  };

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
    [setPolicy, setBrushedCity, setRunLabel]
  );

  const profitCol = useMemo(
    () => columns.find((c) => /^profit$/i.test(c)) ?? null,
    [columns]
  );
  const policyCol = useMemo(
    () => columns.find((c) => /^policy$/i.test(c)) ?? null,
    [columns]
  );
  const runLabelCol = useMemo(
    () => columns.find((c) => /^run_label$/i.test(c)) ?? null,
    [columns]
  );
  const cityScaleCol = useMemo(
    () => columns.find((c) => /^city_scale$/i.test(c)) ?? null,
    [columns]
  );

  const highlightPolicies = useMemo(() => {
    if (highlightPoliciesProp?.length) return highlightPoliciesProp;
    if (activePolicy) return [activePolicy];
    return null;
  }, [highlightPoliciesProp, activePolicy]);

  const highlightRunLabels = useMemo(() => {
    if (highlightRunLabelsProp?.length) return highlightRunLabelsProp;
    const expanded = resolveBrushedRunLabels(
      portfolioRunLabels,
      activeRunLabel,
      brushedCity
    );
    if (expanded?.length) return expanded;
    if (activeRunLabel) return [activeRunLabel];
    return null;
  }, [highlightRunLabelsProp, portfolioRunLabels, activeRunLabel, brushedCity]);

  const highlightCityScale = useMemo(
    () => (brushedCity ? [brushedCity] : null),
    [brushedCity]
  );

  const rowMatchesHighlight = useCallback(
    (row: Record<string, unknown>) => {
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
        !cityScaleCol ||
        String(row[cityScaleCol] ?? "") === brushedCity;
      return policyMatch && runMatch && cityMatch;
    },
    [highlightPolicies, highlightRunLabels, brushedCity, policyCol, runLabelCol, cityScaleCol]
  );

  const applyProfitBrush = useCallback(() => {
    if (!profitCol || !rows.length || !onProfitRange) return;
    const vals = rows
      .map((r) => Number(r[profitCol]))
      .filter((v) => Number.isFinite(v));
    if (!vals.length) return;
    const min = Math.min(...vals);
    const max = Math.max(...vals);
    onProfitRange(min, max);
    toast.success("Profit brush applied", {
      description: `€${min.toFixed(0)} – €${max.toFixed(0)}`,
    });
  }, [profitCol, rows, onProfitRange]);

  return (
    <div className="card space-y-3">
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex items-center justify-between w-full text-left"
      >
        <div>
          <p className="text-xs font-semibold text-gray-300">DuckDB SQL Explorer (§G.6)</p>
          <p className="text-[10px] text-canvas-muted">Table: {tableName}</p>
        </div>
        {open ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
      </button>

      {open && (
        <>
          <div className="flex flex-wrap gap-1.5">
            {templates.map((tpl) => (
              <button
                key={tpl.id}
                onClick={() => setSql(tpl.sql)}
                className="btn-ghost text-xs"
              >
                {tpl.label}
              </button>
            ))}
          </div>

          <Suspense
            fallback={<p className="text-xs text-canvas-muted py-8 text-center">Loading editor…</p>}
          >
            <div className="rounded-lg border border-canvas-border overflow-hidden">
              <MonacoEditor
                height="140px"
                language="sql"
                theme={theme === "dark" ? "vs-dark" : "light"}
                value={sql}
                onChange={(v) => setSql(v ?? "")}
                options={{
                  minimap: { enabled: false },
                  fontSize: 11,
                  lineNumbers: "off",
                  wordWrap: "on",
                  scrollBeyondLastLine: false,
                  automaticLayout: true,
                }}
              />
            </div>
          </Suspense>

          <div className="flex items-center gap-2">
            <button
              onClick={() => void runQuery()}
              disabled={running}
              className="btn-primary text-xs flex items-center gap-1"
            >
              <Play size={12} />
              {running ? "Running…" : "Run query"}
            </button>
            {rows.length > 0 && (
              <button onClick={exportResults} className="btn-ghost text-xs flex items-center gap-1">
                <Download size={12} />
                Export CSV
              </button>
            )}
            {rows.length > 0 && (
              <span className="text-xs text-canvas-muted">
                {filterText.trim()
                  ? `${filteredRows.length.toLocaleString()} / ${rows.length.toLocaleString()} row(s)`
                  : `${rows.length.toLocaleString()} row(s)`}
              </span>
            )}
            {profitCol && onProfitRange && rows.length > 0 && (
              <button onClick={applyProfitBrush} className="btn-ghost text-xs">
                Brush profit range
              </button>
            )}
          </div>

          {error && <p className="text-xs text-accent-danger font-mono">{error}</p>}

          <SqlQueryResultsView
            chartSpec={chartSpec}
            chartOption={chartOption}
            chartAlternatives={chartAlternatives}
            setChartTypeOverride={setChartTypeOverride}
            sortedRows={sortedRows}
            filteredRows={filteredRows}
            rows={rows}
            columns={columns}
            logScale={logScale}
            handlePivotCrossFilter={applyCrossFilter}
            highlightPolicies={highlightPolicies}
            highlightRunLabels={highlightRunLabels}
            highlightCityScale={highlightCityScale}
            filterText={filterText}
            setFilterText={setFilterText}
            sortCol={sortCol}
            sortDir={sortDir}
            toggleSort={toggleSort}
            rowMatchesHighlight={rowMatchesHighlight}
            policyCol={policyCol}
            runLabelCol={runLabelCol}
            cityScaleCol={cityScaleCol}
            activePolicy={activePolicy}
            activeRunLabel={activeRunLabel}
            brushedCity={brushedCity}
            applyCrossFilter={applyCrossFilter}
            onDaySelect={onDaySelect}
          />
        </>
      )}
    </div>
  );
}
