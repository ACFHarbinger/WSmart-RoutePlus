import { useMemo } from "react";
import { OrionHoneycombMap } from "../../ui/orion/OrionHoneycombMap";
import { useCorpusSceneState } from "../../../store/simulationCorpus";
import type { DayLogEntry } from "../../../types";
import { honeycombFromBins, honeycombSubtitle } from "../../../utils/orion/liveScenes";
import { binsFromCorpusPoints, binsFromLogEntries } from "./summaryHoneycomb";

/** Map-subtab honeycomb. Corpus bins win. Otherwise the open simulation log. */
export function SummaryHoneycomb({ entries }: { entries: readonly DayLogEntry[] }) {
  const { view } = useCorpusSceneState();
  const bins = useMemo(
    () => (view != null ? binsFromCorpusPoints(view.points) : binsFromLogEntries(entries)),
    [view, entries]
  );
  const scene = useMemo(() => honeycombFromBins(bins), [bins]);

  if (!scene) {
    return <p className="text-sm text-canvas-muted">This view has no bins to place.</p>;
  }

  return (
    <OrionHoneycombMap
      title="Bin honeycomb"
      subtitle={honeycombSubtitle(scene)}
      zones={scene.zones}
      trajectories={scene.trajectories}
      height={340}
    />
  );
}
