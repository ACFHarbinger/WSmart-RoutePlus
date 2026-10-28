import { useCorpusSceneState } from "../../../../store/simulationCorpus";
import { DayBinFieldScene } from "./scenes/DayBinFieldScene";
import { DivergingBarsScene } from "./scenes/DivergingBarsScene";
import { MetricFlowScene } from "./scenes/MetricFlowScene";
import { PolicyOrbitScene } from "./scenes/PolicyOrbitScene";
import { CorpusGlobeScene } from "./scenes/CorpusGlobeScene";
import { CorpusHexScene } from "./scenes/CorpusHexScene";

/** Mounts the corpus scenes. Leaving Simulation Summary unmounts the globe canvas. */
export function CorpusScenes() {
  const { view, metric, highlight, setHighlight } = useCorpusSceneState();
  if (!view) return null;
  const scene = { view, metric, highlight, onHighlight: setHighlight };
  return (
    <div className="space-y-4" data-testid="corpus-scenes">
      <PolicyOrbitScene {...scene} />
      <DivergingBarsScene {...scene} />
      <MetricFlowScene {...scene} />
      <DayBinFieldScene {...scene} />
      <CorpusGlobeScene {...scene} />
      <CorpusHexScene {...scene} />
    </div>
  );
}
