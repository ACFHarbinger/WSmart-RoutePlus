import type { CSSProperties } from "react";

/**
 * Surface style helpers for NOC/Orion panels and cards (Slice V1).
 * Uses CSS custom properties (`--noc-*`) so light and dark themes
 * automatically resolve without inline conditional branches.
 */

export const nocPanelStyle: CSSProperties = {
  backgroundColor: "var(--noc-bg-panel)",
  borderColor: "var(--noc-border)",
  color: "var(--noc-text-primary)",
};

export const nocCardDenseStyle: CSSProperties = {
  backgroundColor: "var(--noc-bg-card)",
  borderColor: "var(--noc-border)",
  color: "var(--noc-text-primary)",
  padding: "0.75rem",
  borderRadius: "0.75rem",
};

export const nocSubsurfaceStyle: CSSProperties = {
  backgroundColor: "var(--noc-bg-deep)",
  borderColor: "var(--noc-border)",
  color: "var(--noc-text-secondary)",
};
