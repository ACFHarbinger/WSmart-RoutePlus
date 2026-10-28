import { useEffect, useMemo, type ChangeEvent } from "react";

import { useSimulationCorpusStore } from "../../../../store/simulationCorpus";
import type { CorpusScanResult } from "../../../../types/corpus";

/** Discrete day step. Reduced motion does not turn this into a tween. */
const DAY_STEP_MS = 800;

function finiteDay(value: number | null): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

/** Sorted unique day numbers from the scan. Null and non-finite days are not an index. */
function corpusPlaybackDays(scan: CorpusScanResult | null): number[] {
  if (!scan) return [];
  const days = new Set<number>();
  for (const file of scan.files) {
    for (const row of file.days) {
      if (finiteDay(row.day)) days.add(row.day);
    }
  }
  return [...days].sort((left, right) => left - right);
}

function advancePlayback(): void {
  const state = useSimulationCorpusStore.getState();
  const days = corpusPlaybackDays(state.scan);
  if (days.length === 0) {
    state.setPlaybackPlaying(false);
    return;
  }
  const index = state.playbackDay == null ? -1 : days.indexOf(state.playbackDay);
  const nextIndex = index + 1;
  if (nextIndex >= days.length) {
    state.setPlaybackPlaying(false);
    return;
  }
  const nextDay = days[nextIndex];
  if (nextDay == null) {
    state.setPlaybackPlaying(false);
    return;
  }
  useSimulationCorpusStore.setState({
    playbackDay: nextDay,
    playbackPlaying: nextIndex < days.length - 1,
  });
}

/** Day scrubber for the corpus. Visible once a scan exists. Does not tween the slider. */
export function CorpusPlayback() {
  const scan = useSimulationCorpusStore((state) => state.scan);
  const playbackDay = useSimulationCorpusStore((state) => state.playbackDay);
  const playbackPlaying = useSimulationCorpusStore((state) => state.playbackPlaying);
  const setPlaybackDay = useSimulationCorpusStore((state) => state.setPlaybackDay);
  const setPlaybackPlaying = useSimulationCorpusStore((state) => state.setPlaybackPlaying);
  const days = useMemo(() => corpusPlaybackDays(scan), [scan]);

  useEffect(() => {
    if (!playbackPlaying) return;
    const timer = window.setInterval(advancePlayback, DAY_STEP_MS);
    return () => window.clearInterval(timer);
  }, [playbackPlaying]);

  if (!scan) return null;

  const index = playbackDay == null ? -1 : days.indexOf(playbackDay);
  const sliderIndex = index < 0 ? 0 : index;

  function play(): void {
    if (days.length === 0) return;
    const current = playbackDay == null ? -1 : days.indexOf(playbackDay);
    if (current < 0) setPlaybackDay(days[0] ?? null);
    const start = current < 0 ? 0 : current;
    setPlaybackPlaying(start < days.length - 1);
  }

  function onPlaybackClick(): void {
    if (playbackPlaying) {
      setPlaybackPlaying(false);
      return;
    }
    play();
  }

  function onSlide(event: ChangeEvent<HTMLInputElement>): void {
    const next = days[Number(event.currentTarget.value)];
    if (next == null) return;
    setPlaybackPlaying(false);
    setPlaybackDay(next);
  }

  return (
    <section className="card space-y-2" aria-label="Corpus playback">
      <h2 className="text-sm font-semibold">Day playback</h2>
      {days.length === 0 ? (
        <p className="text-xs text-canvas-muted">This corpus has no day index.</p>
      ) : (
        <div className="flex flex-wrap items-center gap-2">
          <button
            type="button"
            className="btn-ghost text-xs"
            onClick={onPlaybackClick}
          >
            {playbackPlaying ? "Pause" : "Play"}
          </button>
          <label className="flex items-center gap-2 text-xs text-canvas-muted">
            Day
            <input
              type="range"
              min={0}
              max={Math.max(0, days.length - 1)}
              step={1}
              value={sliderIndex}
              onChange={onSlide}
              aria-label="Corpus day"
              aria-valuetext={playbackDay == null ? "not selected" : `day ${playbackDay}`}
            />
          </label>
          <span className="text-xs text-canvas-muted tabular-nums">
            {playbackDay == null ? "—" : playbackDay}
          </span>
        </div>
      )}
    </section>
  );
}
