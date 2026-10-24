import { useEffect } from "react";
import { useAppStore } from "../../store/app";
import { applyDomTheme, getSystemTheme, resolveEffectiveTheme } from "../../utils/charts/theme";

/** Keeps DOM `dark`/`light` classes, `--noc-*` variables, and `effectiveTheme` in sync. */
export function useThemeSync(): void {
  const theme = useAppStore((s) => s.theme);
  const setEffectiveTheme = useAppStore((s) => s.setEffectiveTheme);

  useEffect(() => {
    const initial = resolveEffectiveTheme(theme);
    applyDomTheme(initial);
    setEffectiveTheme(initial);

    if (theme !== "system" || typeof window === "undefined") return;
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const sync = () => {
      const effective = getSystemTheme();
      applyDomTheme(effective);
      setEffectiveTheme(effective);
    };
    mq.addEventListener("change", sync);
    return () => mq.removeEventListener("change", sync);
  }, [theme, setEffectiveTheme]);
}
