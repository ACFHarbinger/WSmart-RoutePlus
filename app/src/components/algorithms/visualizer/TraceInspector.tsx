import type { TraceEvent } from "../../../types/algorithms";
import {
  eventArguments,
  eventFields,
  eventInterior,
  eventKind,
  eventName,
  formatRoutes,
  solutionRoutes,
} from "../../../pages/algorithms/visualizer/flowModel";
import { RouteThumbnail } from "./RouteThumbnail";

interface TraceInspectorProps {
  event: TraceEvent | null;
  coordinates: Array<[number, number]> | null;
}

function RouteBlock({
  label,
  testId,
  routes,
  coordinates,
}: {
  label: string;
  testId: string;
  routes: number[][];
  coordinates: Array<[number, number]> | null;
}) {
  return (
    <div className="space-y-1">
      <h3 className="text-xs uppercase tracking-wide text-noc-textMuted">{label}</h3>
      <p className="text-sm font-mono text-noc-textPrimary break-all" data-testid={testId}>
        {formatRoutes(routes)}
      </p>
      <RouteThumbnail routes={routes} coordinates={coordinates} />
    </div>
  );
}

export function TraceInspector({ event, coordinates }: TraceInspectorProps) {
  if (!event) {
    return (
      <section className="bg-noc-card border border-noc-border rounded-xl p-3">
        <h2 className="text-sm font-medium text-noc-textPrimary">Inspector</h2>
        <p className="mt-2 text-xs text-noc-textMuted">Select a step in the flow.</p>
      </section>
    );
  }
  const before = solutionRoutes(eventFields(event).before);
  const after = solutionRoutes(eventFields(event).after);
  const interior = eventInterior(event);
  return (
    <section className="bg-noc-card border border-noc-border rounded-xl p-3 space-y-3 overflow-auto max-h-[640px]">
      <div>
        <h2 className="text-sm font-medium text-noc-textPrimary">Inspector</h2>
        <p className="text-sm text-noc-cyan" data-testid="inspector-title">
          {eventName(event)}
        </p>
        <p className="text-[10px] uppercase tracking-wide text-noc-textMuted">{eventKind(event)}</p>
      </div>
      {interior === "not_instrumented" ? (
        <div className="space-y-1">
          <p className="text-xs text-noc-amber">not_instrumented</p>
          <pre className="text-[11px] leading-4 text-noc-textMuted bg-noc-deep border border-noc-border rounded-lg p-2 overflow-auto">
            {JSON.stringify(event, null, 2)}
          </pre>
        </div>
      ) : null}
      <div>
        <h3 className="text-xs uppercase tracking-wide text-noc-textMuted">Arguments</h3>
        <pre className="text-[11px] leading-4 text-noc-textSecondary bg-noc-deep border border-noc-border rounded-lg p-2 overflow-auto">
          {JSON.stringify(eventArguments(event), null, 2)}
        </pre>
      </div>
      <RouteBlock label="Before routes" testId="before-routes" routes={before} coordinates={coordinates} />
      <RouteBlock label="After routes" testId="after-routes" routes={after} coordinates={coordinates} />
    </section>
  );
}
