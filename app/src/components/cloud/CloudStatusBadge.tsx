export interface CloudStatusBadgeProps {
  state: string;
  size?: "sm" | "md";
  className?: string;
}

const STATE_CONFIG: Record<string, { label: string; classes: string; pulse?: boolean }> = {
  // Job & Task common states
  running: {
    label: "Running",
    classes: "bg-accent-success/15 text-accent-success border-accent-success/30",
    pulse: true,
  },
  succeeded: {
    label: "Succeeded",
    classes: "bg-emerald-500/15 text-emerald-400 border-emerald-500/30",
  },
  failed: {
    label: "Failed",
    classes: "bg-accent-danger/15 text-accent-danger border-accent-danger/30",
  },
  cancelled: {
    label: "Cancelled",
    classes: "bg-canvas-muted/20 text-canvas-muted border-canvas-border",
  },
  queued: {
    label: "Queued",
    classes: "bg-amber-500/15 text-amber-400 border-amber-500/30",
  },
  // Task specific states
  placing: {
    label: "Placing",
    classes: "bg-indigo-500/15 text-indigo-400 border-indigo-500/30",
    pulse: true,
  },
  starting: {
    label: "Starting",
    classes: "bg-cyan-500/15 text-cyan-400 border-cyan-500/30",
    pulse: true,
  },
  preempted: {
    label: "Preempted",
    classes: "bg-orange-500/15 text-orange-400 border-orange-500/30",
  },
  retrying: {
    label: "Retrying",
    classes: "bg-yellow-500/15 text-yellow-400 border-yellow-500/30",
    pulse: true,
  },
};

export function CloudStatusBadge({
  state,
  size = "md",
  className = "",
}: CloudStatusBadgeProps) {
  const normState = state.toLowerCase();
  const config = STATE_CONFIG[normState] ?? {
    label: state,
    classes: "bg-canvas-elevated text-gray-300 border-canvas-border",
  };

  const pad = size === "sm" ? "px-1.5 py-0.2 text-[10px]" : "px-2.5 py-0.5 text-xs";

  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border font-medium ${pad} ${config.classes} ${className}`}
    >
      {config.pulse && (
        <span className="w-1.5 h-1.5 rounded-full bg-current animate-pulse" />
      )}
      <span>{config.label}</span>
    </span>
  );
}
