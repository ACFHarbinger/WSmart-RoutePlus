import { NOC_LIGHT_SURFACES, NOC_SURFACES } from "../../components/ui/telemetry/tokens";

export type ThemePreference = "dark" | "light" | "system";
export type EffectiveTheme = "dark" | "light";

export function getSystemTheme(): EffectiveTheme {
  if (typeof window === "undefined") return "dark";
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

export function resolveEffectiveTheme(preference: ThemePreference): EffectiveTheme {
  if (preference === "system") return getSystemTheme();
  return preference;
}

export function applyDomTheme(effective: EffectiveTheme): void {
  if (typeof document === "undefined") return;
  const root = document.documentElement;
  if (effective === "dark") {
    root.classList.add("dark");
    root.classList.remove("light");
  } else {
    root.classList.remove("dark");
    root.classList.add("light");
  }
  root.dataset.theme = effective;
}

const THEME_CYCLE: ThemePreference[] = ["dark", "light", "system"];

export function nextThemePreference(current: ThemePreference): ThemePreference {
  const idx = THEME_CYCLE.indexOf(current);
  return THEME_CYCLE[(idx + 1) % THEME_CYCLE.length];
}

export function getNocChartTheme(effective: EffectiveTheme = "dark") {
  return effective === "light" ? NOC_LIGHT_SURFACES : NOC_SURFACES;
}
