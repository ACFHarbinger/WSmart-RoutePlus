/**
 * Priority, placement targets and warm-pool inputs of the cloud Launcher (CR-09).
 */
export interface PlacementFieldsProps {
  priority: number;
  targets: string;
  pool: boolean;
  onChange: (updates: { priority?: number; targets?: string; pool?: boolean }) => void;
}

/** One `provider/compute[/region[/machine_type[/spot]]][:max_tasks]` per non-empty line. */
export function parseTargets(text: string): string[] {
  return text
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line.length > 0);
}

export function PlacementFields({ priority, targets, pool, onChange }: PlacementFieldsProps) {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-3 gap-4" data-testid="placement-fields">
      <div className="space-y-1">
        <label className="text-xs font-medium text-canvas-muted">Priority (--priority)</label>
        <input
          type="number"
          step={1}
          className="input-base font-mono text-xs"
          value={priority}
          onChange={(e) => onChange({ priority: Math.trunc(Number(e.target.value) || 0) })}
        />
        <label className="flex items-center gap-2 pt-2 text-xs text-gray-200">
          <input type="checkbox" checked={pool} onChange={(e) => onChange({ pool: e.target.checked })} />
          Run on the warm pool (--pool)
        </label>
      </div>
      <div className="sm:col-span-2 space-y-1">
        <label className="text-xs font-medium text-canvas-muted">Placement targets (--target, one per line)</label>
        <textarea
          rows={3}
          className="input-base font-mono text-xs"
          placeholder={"gcp/batch/europe-west1/e2-standard-4/spot:20\naws/batch/eu-west-1/c6i.xlarge:10"}
          value={targets}
          onChange={(e) => onChange({ targets: e.target.value })}
        />
        <p className="text-[10px] text-canvas-muted">
          Tried cheapest first; a full or failing target spills tasks to the next. Empty: the provider and compute above.
        </p>
      </div>
    </div>
  );
}
