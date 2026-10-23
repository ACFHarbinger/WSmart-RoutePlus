import { useCallback } from "react";
import { invoke } from "@tauri-apps/api/core";
import { toast } from "sonner";
import { useLayoutStore } from "../../store/layout";
import { recentKindFromPath, type RecentFileKind } from "../../store/recentFiles";
import type { WsrouteExtractResult } from "../../types";
import { highestPriorityKind, recentHandoffSpec } from "../../utils/runs/recentHandoff";
import { useFileDrop } from "./useFileDrop";
import { useRecentHandoff } from "./useRecentHandoff";
import { simOutputRoot } from "../../utils/runs/outputRoots";

function findPath(paths: string[], re: RegExp): string | undefined {
  return paths.find((p) => re.test(p));
}

/**
 * App-wide drop handler for Studio artefacts (§G.8 / §G.6 / §G.7 / §G.12–§G.14 / §G.17).
 * Routes logs, CSVs, checkpoints, configs, training/run directories, and `.wsroute`
 * bundles through shared ``recentKindFromPath`` + ``useRecentHandoff`` handoffs.
 */
export function useGlobalFileDrop() {
  const { projectRoot, setMode, handoff } = useRecentHandoff();
  const setFileDropDragging = useLayoutStore((s) => s.setFileDropDragging);

  const handoffKind = useCallback(
    (path: string, kind: RecentFileKind, toastOnSuccess = true) => {
      const spec = handoff(path, kind);
      if (toastOnSuccess) {
        toast.success(spec.successLabel, {
          description: path.split(/[/\\]/).pop(),
        });
      }
    },
    [handoff]
  );

  const handleDrop = useCallback(
    async (paths: string[]) => {
      const wsroute = findPath(paths, /\.wsroute$/i);

      if (wsroute) {
        if (!projectRoot) {
          toast.error("Set project root in Settings before importing bundles");
          return;
        }
        const destDir = `${simOutputRoot(projectRoot)}/.imports/${Date.now()}`;
        try {
          const result = await invoke<WsrouteExtractResult>("extract_wsroute_bundle", {
            path: wsroute,
            destDir,
          });
          toast.success(`Extracted ${result.extracted_files.length} files`);
          if (result.log_path) {
            handoffKind(result.log_path, "log", false);
            toast.success("Log loaded", {
              description: result.log_path.split(/[/\\]/).pop(),
            });
          } else {
            setMode("output_browser");
            toast.info("No .jsonl log in bundle — browse extracted files in Output Browser");
          }
        } catch (err) {
          toast.error("Failed to import bundle", { description: String(err) });
        }
        return;
      }

      // Classify every path; push all known kinds, navigate to highest priority.
      const classified: Array<{ path: string; kind: RecentFileKind }> = [];
      for (const path of paths) {
        const kind = recentKindFromPath(path);
        if (kind) classified.push({ path, kind });
      }

      if (classified.length === 0) {
        toast.error(
          "Drop a .wsroute, .jsonl, .csv, checkpoint, config, training logs/, or simulation run folder"
        );
        return;
      }

      const primaryKind = highestPriorityKind(classified.map((c) => c.kind));
      const primary = primaryKind
        ? classified.find((c) => c.kind === primaryKind)
        : undefined;
      if (!primary) return;

      // Push every path to recents; only primary navigates via pending-path handoff.
      for (const { path, kind } of classified) {
        const isPrimary = path === primary.path && kind === primary.kind;
        handoff(path, kind, { navigate: isPrimary });
      }

      const spec = recentHandoffSpec(primary.kind);
      const extra =
        classified.length > 1 ? ` (+${classified.length - 1} more in Recent)` : "";
      toast.success(`${spec.successLabel}${extra}`, {
        description: primary.path.split(/[/\\]/).pop(),
      });
    },
    [projectRoot, setMode, handoffKind, handoff]
  );

  return useFileDrop(handleDrop, true, setFileDropDragging);
}
