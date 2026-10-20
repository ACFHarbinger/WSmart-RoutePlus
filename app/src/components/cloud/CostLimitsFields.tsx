/**
 * Machine type, region and job budget inputs of the cloud Launcher (CR-10).
 */
export interface CostLimitsFieldsProps {
  machineType: string;
  region: string;
  budgetUsd: number;
  onChange: (updates: { machineType?: string; region?: string; budgetUsd?: number }) => void;
}

export function CostLimitsFields({ machineType, region, budgetUsd, onChange }: CostLimitsFieldsProps) {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-3 gap-4" data-testid="cost-limits-fields">
      <div className="space-y-1">
        <label className="text-xs font-medium text-canvas-muted">Machine Type (--machine-type)</label>
        <input
          type="text"
          className="input-base font-mono text-xs"
          placeholder="backend default, e.g. c6i.large"
          value={machineType}
          onChange={(e) => onChange({ machineType: e.target.value })}
        />
      </div>
      <div className="space-y-1">
        <label className="text-xs font-medium text-canvas-muted">Region (--region)</label>
        <input
          type="text"
          className="input-base font-mono text-xs"
          placeholder="backend default"
          value={region}
          onChange={(e) => onChange({ region: e.target.value })}
        />
      </div>
      <div className="space-y-1">
        <label className="text-xs font-medium text-canvas-muted">Budget Cap, USD (--budget)</label>
        <input
          type="number"
          min={0}
          step={0.5}
          className="input-base font-mono text-xs"
          placeholder="0 = no cap"
          value={budgetUsd || ""}
          onChange={(e) => onChange({ budgetUsd: Math.max(0, Number(e.target.value) || 0) })}
        />
        <p className="text-[10px] text-canvas-muted">
          Refused if the estimate exceeds it; cancelled when the accrued cost reaches it.
        </p>
      </div>
    </div>
  );
}
