import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Search } from "lucide-react";
import { PALETTE_COMMANDS } from "../../constants/commands";
import { useWsrouteImport } from "../../hooks/files/useWsrouteImport";
import { invoke } from "@tauri-apps/api/core";
import { useAppStore } from "../../store/app";
import { useRecentHandoff } from "../../hooks/files/useRecentHandoff";
import { useLayoutStore } from "../../store/layout";
import { nextThemePreference } from "../../utils/charts/theme";
import { OpenPathToolbar } from "../common/OpenPathToolbar";
import { useRecentFilesStore, type RecentFile, type RecentFileKind } from "../../store/recentFiles";
import type { DayLogEntry } from "../../types";

function matchQuery(query: string, label: string, keywords?: string): boolean {
  const q = query.trim().toLowerCase();
  if (!q) return true;
  const haystack = `${label} ${keywords ?? ""}`.toLowerCase();
  return q.split(/\s+/).every((token) => haystack.includes(token));
}

const KNOWN_RECENT_KINDS: RecentFileKind[] = [
  "log",
  "run",
  "csv",
  "training",
  "checkpoint",
  "config",
];

function isKnownRecentKind(kind: string): kind is RecentFileKind {
  return (KNOWN_RECENT_KINDS as string[]).includes(kind);
}

export function CommandPalette() {
  const { theme, setTheme } = useAppStore();
  const { projectRoot, setMode, handoff } = useRecentHandoff();
  const { commandPaletteOpen, setCommandPaletteOpen, setShortcutsOpen, setGuidedTourOpen, setGuidedTourStep } =
    useLayoutStore();
  const recentFiles = useRecentFilesStore((s) => s.files);
  const refreshRecentLabels = useRecentFilesStore((s) => s.refreshRecentLabels);
  const importWsroute = useWsrouteImport();
  const [query, setQuery] = useState("");
  const [activeIndex, setActiveIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);

  const filteredCommands = useMemo(
    () => PALETTE_COMMANDS.filter((cmd) => matchQuery(query, cmd.label, cmd.keywords)),
    [query]
  );

  const filteredRecent = useMemo(() => {
    if (!query.trim()) return recentFiles;
    const q = query.trim().toLowerCase();
    return recentFiles.filter(
      (f) => f.label.toLowerCase().includes(q) || f.path.toLowerCase().includes(q)
    );
  }, [query, recentFiles]);

  const showRecent = filteredRecent.length > 0 && (query.trim() === "" || filteredCommands.length < 6);

  /** Flattened keyboard-nav list: recents (when shown) then commands. */
  type PaletteItem =
    | { type: "recent"; file: RecentFile }
    | { type: "command"; cmd: (typeof PALETTE_COMMANDS)[number] };

  const paletteItems = useMemo((): PaletteItem[] => {
    const items: PaletteItem[] = [];
    if (showRecent) {
      for (const file of filteredRecent) items.push({ type: "recent", file });
    }
    for (const cmd of filteredCommands) items.push({ type: "command", cmd });
    return items;
  }, [showRecent, filteredRecent, filteredCommands]);

  const closePalette = useCallback(() => {
    setCommandPaletteOpen(false);
    setQuery("");
    setActiveIndex(0);
  }, [setCommandPaletteOpen]);

  const openRecentFile = useCallback(
    async (file: RecentFile) => {
      if (!isKnownRecentKind(file.kind)) {
        setMode("data_explorer");
        closePalette();
        return;
      }

      const kind = file.kind;

      // Logs: prefer Simulation Summary; fall back to Digital Twin if load fails.
      if (kind === "log") {
        try {
          await invoke<DayLogEntry[]>("load_simulation_log", { path: file.path });
          handoff(file.path, kind, { storedLabel: file.label });
        } catch {
          handoff(file.path, kind, { storedLabel: file.label, mode: "simulation" });
        }
        closePalette();
        return;
      }

      handoff(file.path, kind, { storedLabel: file.label });
      closePalette();
    },
    [setMode, handoff, closePalette]
  );

  const runCommand = useCallback(
    (cmd: (typeof PALETTE_COMMANDS)[number]) => {
      if (cmd.mode) setMode(cmd.mode);
      else if (cmd.action === "toggle_theme") setTheme(nextThemePreference(theme));
      else if (cmd.action === "shortcuts_help") setShortcutsOpen(true);
      else if (cmd.action === "import_wsroute") void importWsroute();
      else if (cmd.action === "guided_tour") {
        setGuidedTourStep(0);
        setGuidedTourOpen(true);
      }
      closePalette();
    },
    [setMode, setTheme, theme, setShortcutsOpen, importWsroute, setGuidedTourOpen, setGuidedTourStep, closePalette]
  );

  const activateItem = useCallback(
    (item: PaletteItem) => {
      if (item.type === "recent") void openRecentFile(item.file);
      else runCommand(item.cmd);
    },
    [openRecentFile, runCommand]
  );

  useEffect(() => {
    if (commandPaletteOpen) {
      refreshRecentLabels(projectRoot);
      setQuery("");
      setActiveIndex(0);
      requestAnimationFrame(() => inputRef.current?.focus());
    }
  }, [commandPaletteOpen, projectRoot, refreshRecentLabels]);

  useEffect(() => {
    setActiveIndex((i) => (paletteItems.length ? Math.min(i, paletteItems.length - 1) : 0));
  }, [paletteItems.length, query]);

  if (!commandPaletteOpen) return null;

  const recentCount = showRecent ? filteredRecent.length : 0;

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center bg-black/60 p-4 pt-[12vh] backdrop-blur-[1px]"
      onClick={() => setCommandPaletteOpen(false)}
    >
      <div
        className="w-full max-w-lg overflow-hidden shadow-2xl bg-noc-bgPanel border border-noc-borderStrong rounded-xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center gap-2 px-3.5 py-2.5 border-b border-noc-border bg-noc-bgDeep/60">
          <Search size={14} className="text-noc-textMuted shrink-0" />
          <input
            ref={inputRef}
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "ArrowDown") {
                e.preventDefault();
                setActiveIndex((i) =>
                  paletteItems.length ? (i + 1) % paletteItems.length : 0
                );
              } else if (e.key === "ArrowUp") {
                e.preventDefault();
                setActiveIndex((i) =>
                  paletteItems.length
                    ? (i - 1 + paletteItems.length) % paletteItems.length
                    : 0
                );
              } else if (e.key === "Enter" && paletteItems[activeIndex]) {
                e.preventDefault();
                activateItem(paletteItems[activeIndex]!);
              } else if (e.key === "Escape") {
                e.preventDefault();
                setCommandPaletteOpen(false);
              }
            }}
            placeholder="Search views, actions, and recent files…"
            className="input-base flex-1 border-0 bg-transparent text-sm focus:ring-0 text-noc-textPrimary placeholder:text-noc-textMuted"
          />
          <kbd className="text-[10px] text-noc-textMuted font-mono shrink-0 bg-noc-bgDeep border border-noc-border px-1.5 py-0.5 rounded">Esc</kbd>
        </div>

        <ul className="max-h-72 overflow-y-auto py-1">
          {showRecent && (
            <>
              <li className="px-3 py-1 text-[10px] font-semibold uppercase tracking-wider text-noc-textMuted">
                Recent
              </li>
              {filteredRecent.map((file, i) => {
                const itemIndex = i;
                const active = activeIndex === itemIndex;
                const kindMeta = (
                  <span className="text-[10px] text-noc-textMuted shrink-0">{file.kind}</span>
                );
                return (
                  <li key={file.path}>
                    <div
                      role="option"
                      aria-selected={active}
                      onMouseEnter={() => setActiveIndex(itemIndex)}
                      onClick={() => void openRecentFile(file)}
                      className={`w-full flex items-center justify-between gap-2 px-3 py-2 text-left text-sm transition-colors cursor-pointer ${
                        active
                          ? "bg-noc-cyan/15 text-noc-cyan font-medium"
                          : "text-noc-textPrimary hover:bg-noc-bgDeep"
                      }`}
                    >
                      {isKnownRecentKind(file.kind) ? (
                        <OpenPathToolbar
                          path={file.path}
                          projectRoot={projectRoot}
                          kind={file.kind}
                          label={file.label}
                          storedLabel={file.label}
                          handoff={file.kind}
                          handoffOnAfterOpen={closePalette}
                          chipClassName="flex-1 min-w-0 max-w-none"
                          className="flex-1 min-w-0"
                        >
                          {kindMeta}
                        </OpenPathToolbar>
                      ) : (
                        <>
                          <span className="truncate flex-1 min-w-0">{file.label}</span>
                          {kindMeta}
                        </>
                      )}
                    </div>
                  </li>
                );
              })}
            </>
          )}
          {paletteItems.length === 0 && (
            <li className="px-3 py-4 text-xs text-noc-textMuted text-center">No matching commands</li>
          )}
          {filteredCommands.length > 0 && (
            <li className="px-3 py-1 text-[10px] font-semibold uppercase tracking-wider text-noc-textMuted">
              Commands
            </li>
          )}
          {filteredCommands.map((cmd, i) => {
            const itemIndex = recentCount + i;
            const active = activeIndex === itemIndex;
            return (
              <li key={cmd.id}>
                <button
                  type="button"
                  onMouseEnter={() => setActiveIndex(itemIndex)}
                  onClick={() => runCommand(cmd)}
                  className={`w-full flex items-center justify-between px-3 py-2 text-left text-sm transition-colors ${
                    active
                      ? "bg-noc-cyan/15 text-noc-cyan font-medium"
                      : "text-noc-textPrimary hover:bg-noc-bgDeep"
                  }`}
                >
                  <span>{cmd.label}</span>
                  <span className="text-[10px] text-noc-textMuted">{cmd.section}</span>
                </button>
              </li>
            );
          })}
        </ul>
      </div>
    </div>
  );
}
