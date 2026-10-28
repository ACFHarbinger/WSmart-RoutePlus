import type { CorpusHighlight, CorpusMetric, CorpusView } from "../../../../types/corpus";

/** Props every corpus scene accepts. Scenes do not read the store. */
export interface CorpusSceneProps {
  view: CorpusView;
  metric: CorpusMetric;
  highlight: CorpusHighlight | null;
  onHighlight: (next: CorpusHighlight | null) => void;
}
