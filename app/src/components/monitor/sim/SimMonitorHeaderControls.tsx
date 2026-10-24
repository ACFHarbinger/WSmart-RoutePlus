import { ChevronLeft, ChevronRight, Pause, Play } from "lucide-react";
import { POLICY_COLORS } from "./SimMonitorPanels";

export function SimMonitorDayScrubber({
  dayRange,
  displayDay,
  setSelectedDay,
  isPlaying,
  setIsPlaying,
  playbackSpeed,
  setPlaybackSpeed,
  selectedDay,
  isWatching,
}: {
  dayRange: { min: number; max: number };
  displayDay: number;
  setSelectedDay: (d: number | null) => void;
  isPlaying: boolean;
  setIsPlaying: (p: boolean) => void;
  playbackSpeed: 1 | 2 | 4;
  setPlaybackSpeed: (s: 1 | 2 | 4) => void;
  selectedDay: number | null;
  isWatching: boolean;
}) {
  if (dayRange.max <= 0) return null;

  return (
    <div className="flex items-center gap-1.5 ml-auto">
      <span className="text-xs text-canvas-muted">Day</span>
      <button
        onClick={() => setSelectedDay(Math.max(dayRange.min, displayDay - 1))}
        disabled={displayDay <= dayRange.min}
        className="btn-ghost p-1 disabled:opacity-30"
        title="Previous day"
      >
        <ChevronLeft size={12} />
      </button>
      <input
        type="range"
        min={dayRange.min}
        max={dayRange.max}
        value={displayDay}
        onChange={(e) => setSelectedDay(Number(e.target.value))}
        className="w-28 accent-accent-primary"
      />
      <button
        onClick={() => setSelectedDay(Math.min(dayRange.max, displayDay + 1))}
        disabled={displayDay >= dayRange.max}
        className="btn-ghost p-1 disabled:opacity-30"
        title="Next day"
      >
        <ChevronRight size={12} />
      </button>
      <span className="text-xs font-mono text-canvas-muted w-14 text-center tabular-nums">
        {displayDay}/{dayRange.max}
      </span>
      <button
        onClick={() => {
          if (isPlaying) {
            setIsPlaying(false);
          } else {
            if (selectedDay === null) setSelectedDay(dayRange.min);
            setIsPlaying(true);
          }
        }}
        className="btn-ghost p-1"
        title={isPlaying ? "Pause day playback" : "Play through simulation days"}
      >
        {isPlaying ? <Pause size={12} /> : <Play size={12} />}
      </button>
      <select
        className="select-base text-[10px] py-0.5 w-14"
        value={playbackSpeed}
        onChange={(e) => setPlaybackSpeed(Number(e.target.value) as 1 | 2 | 4)}
        title="Playback speed"
      >
        <option value={1}>1×</option>
        <option value={2}>2×</option>
        <option value={4}>4×</option>
      </select>
      {selectedDay !== null ? (
        <button
          onClick={() => {
            setIsPlaying(false);
            setSelectedDay(null);
          }}
          className="btn-ghost text-xs text-accent-primary"
          title="Jump to latest day and follow live updates"
        >
          Latest ↓
        </button>
      ) : isWatching ? (
        <span className="flex items-center gap-1 text-xs text-accent-success">
          <span className="w-1.5 h-1.5 rounded-full bg-accent-success animate-pulse" />
          Following
        </span>
      ) : null}
    </div>
  );
}

export function SimMonitorChartPolicySelector({
  policies,
  activeChartPolicies,
  toggleChartPolicy,
}: {
  policies: string[];
  activeChartPolicies: string[];
  toggleChartPolicy: (p: string) => void;
}) {
  if (policies.length <= 1) return null;

  return (
    <div className="flex items-center gap-2 flex-wrap">
      <span className="text-xs text-canvas-muted shrink-0">Charts:</span>
      {policies.map((p, i) => {
        const color = POLICY_COLORS[i % POLICY_COLORS.length];
        const active = activeChartPolicies.includes(p);
        return (
          <button
            key={p}
            onClick={() => toggleChartPolicy(p)}
            className={`text-xs px-2 py-0.5 rounded-full border transition-opacity ${
              active ? "opacity-100" : "opacity-35"
            }`}
            style={{
              borderColor: color,
              color: active ? color : undefined,
              background: active ? `${color}18` : undefined,
            }}
          >
            {p}
          </button>
        );
      })}
    </div>
  );
}
