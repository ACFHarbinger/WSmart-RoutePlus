/**
 * Typed configuration layer for the native report/deck generator (§H.0).
 *
 * Ports the archived `logic/gen/json/*` configs and `style/*.mplstyle`
 * sheets into importable, typed data. Nothing downstream (charts, reports,
 * deck) hardcodes a colour, label or path — everything reads from here.
 */
import metaJson from "./simulation_metadata.json";
import themesJson from "./themes.json";
import datasetCfgJson from "./dataset_analysis_config.json";
import simCfgJson from "./simulation_analysis_config.json";
import contentJson from "./presentation_content.json";
import referenceLinksJson from "./referenceLinks.json";

// ── Simulation metadata (parsing rules + palettes) ──────────────────────────

export interface StrategyPrefix {
  prefix: string;
  strategy: string;
  cf: string | null;
  sl_var: string | null;
}

export interface SimulationMeta {
  dir_city: Record<string, string>;
  city_short: Record<string, string>;
  dist_map: Record<string, string>;
  strategy_prefixes: StrategyPrefix[];
  strategy_names: Record<string, string>;
  constructors: [string, string][];
  improver_pattern: string;
  strategy_colors: Record<string, string>;
  variant_colors: Record<string, string>;
  improver_colors: Record<string, string>;
  horizon_colors: string[];
  scenario_colors: string[];
  constructor_colors: Record<string, string>;
  metric_colors: string[];
}

export const META = metaJson as unknown as SimulationMeta;

// ── Themes (themes.json + mplstyle sheets merged) ───────────────────────────

export interface GenTheme {
  name: "dark" | "light";
  fg: string;
  muted: string;
  faint: string;
  accentLine: string;
  guideLine: string;
  /** figure.facecolor from the mplstyle sheet */
  bg: string;
  /** axes.facecolor from the mplstyle sheet */
  axesBg: string;
  gridColor: string;
  gridAlpha: number;
  axisLabelColor: string;
  buttonBg: string;
  buttonBorder: string;
}

// mplstyle values, kept as data (ports style/dark.mplstyle + light.mplstyle)
const MPLSTYLE: Record<string, { bg: string; axesBg: string; grid: string; gridAlpha: number; label: string }> = {
  dark: { bg: "#1a1a2e", axesBg: "#16213e", grid: "#2d2d4e", gridAlpha: 0.5, label: "#e0e0e0" },
  light: { bg: "#ffffff", axesBg: "#ffffff", grid: "#c8c8c8", gridAlpha: 0.6, label: "#222222" },
};

export function loadTheme(name: "dark" | "light"): GenTheme {
  const t = (themesJson as Record<string, Record<string, string | number>>)[name];
  const m = MPLSTYLE[name];
  return {
    name,
    fg: String(t.fg),
    muted: String(t.muted),
    faint: String(t.faint),
    accentLine: String(t.accent_line),
    guideLine: String(t.guide_line),
    bg: m.bg,
    axesBg: m.axesBg,
    gridColor: m.grid,
    gridAlpha: m.gridAlpha,
    axisLabelColor: m.label,
    buttonBg: String(t.button_bg),
    buttonBorder: String(t.button_border),
  };
}

// ── Analysis configs ─────────────────────────────────────────────────────────

export interface HorizonSpec {
  days: number;
  csv: string;
  output_dir?: string;
}

export interface SimulationAnalysisConfig {
  theme: "dark" | "light";
  base_fontsize: number;
  scenarios: { city: string; N: number; dist: string }[] | null;
  policies: {
    strategies: string[] | null;
    constructors: string[] | null;
    improvers: string[] | null;
    acceptance: string[] | null;
  };
  horizons: HorizonSpec[];
  charts: Record<string, Record<string, unknown>>;
  out_md: string;
  figures_dir: string;
  private_dir: string;
}

export const SIM_CFG = simCfgJson as unknown as SimulationAnalysisConfig;

export interface DatasetAnalysisConfig {
  theme: "dark" | "light";
  city_labels: Record<string, string>;
  dist_labels: Record<string, string>;
  city_colors: Record<string, string>;
  dist_colors: Record<string, string>;
  npz_dir: string;
  npz_csv: string;
  td_csv: string;
  out_md: string;
  figures_dir: string;
  private_dir: string;
}

export const DATASET_CFG = datasetCfgJson as unknown as DatasetAnalysisConfig;

// ── Presentation content model (§H.0 content model) ─────────────────────────

export interface ContentSlide {
  title: string;
  equation?: string[];
  eq_size_pt?: number;
  eq_line_h?: number;
  caption?: string;
  figure?: string;
  figures?: string[];
  figures_dir?: string;
  diagram?: string;
  note?: string;
  show_bullets?: boolean;
  bullets?: string[];
  speaker_notes?: string[];
}

export interface PresentationContent {
  title: string;
  subtitle: string;
  author: string;
  coauthors: string[];
  research_groups: string[];
  agenda: string[];
  slides: Record<string, ContentSlide>;
  figure_slides: Record<string, ContentSlide>;
  acknowledgments?: Record<string, unknown>;
}

export const CONTENT = contentJson as unknown as PresentationContent;

export interface ReferenceLink {
  url: string;
  used_in: string;
  license_note: string;
}

export const REFERENCE_LINKS = referenceLinksJson as Record<string, ReferenceLink>;

// ── Shared labels ────────────────────────────────────────────────────────────

export const KGKM_LABEL = "KG / KM";

/** Display formatting for raw data labels: ACO_HH -> ACO-HH (data untouched). */
export function disp(label: string): string {
  return String(label).replace("ACO_HH", "ACO-HH");
}

export function dispAll(labels: string[]): string[] {
  return labels.map(disp);
}
