import type { BuilderEdge, BuilderNode, PendingPort } from "../../../pages/algorithms/builder/graphTypes";
import { statusClass } from "../../../pages/algorithms/builder/catalogView";

interface BuilderCanvasProps {
  nodes: BuilderNode[];
  edges: BuilderEdge[];
  pending: PendingPort | null;
  selectedUid: string | null;
  onSelect: (uid: string) => void;
  onPort: (uid: string, side: "in" | "out") => void;
  onConnect: (uid: string) => void;
  onRemove: (uid: string) => void;
}

export function BuilderCanvas({
  nodes,
  edges,
  pending,
  selectedUid,
  onSelect,
  onPort,
  onConnect,
  onRemove,
}: BuilderCanvasProps) {
  const label = (uid: string) => nodes.find((node) => node.uid === uid)?.entryId ?? uid;
  return (
    <div className="min-w-0 flex-1 overflow-auto bg-noc-deep p-4">
      <div className="mb-3 flex items-baseline justify-between gap-3">
        <h2 className="text-xs font-semibold uppercase tracking-wide text-noc-textMuted">Canvas</h2>
        {pending && (
          <p className="text-[11px] text-noc-cyan">
            Connecting {label(pending.uid)}. Choose the other port or Connect.
          </p>
        )}
      </div>
      {nodes.length === 0 ? (
        <p className="rounded-xl border border-dashed border-noc-border bg-noc-card px-4 py-8 text-sm text-noc-textMuted">
          No modules on the canvas. Add packing, routing, acceptance, and operators.
        </p>
      ) : (
        <ul className="space-y-3">
          {nodes.map((node) => {
            const armed = pending?.uid === node.uid;
            return (
              <li key={node.uid}>
                <article
                  data-testid={`canvas-node-${node.entryId}`}
                  className={`rounded-xl border px-3 py-2 ${statusClass(node.status)} ${
                    selectedUid === node.uid ? "ring-1 ring-noc-cyan" : ""
                  }`}
                  onClick={() => onSelect(node.uid)}
                >
                  <div className="flex flex-wrap items-center gap-2">
                    <button
                      type="button"
                      className={`rounded border px-2 py-0.5 text-[11px] ${
                        armed && pending?.side === "out" ? "border-noc-cyan" : "border-noc-border"
                      }`}
                      aria-label={`${node.entryId} output port`}
                      onClick={(event) => {
                        event.stopPropagation();
                        onPort(node.uid, "out");
                      }}
                    >
                      out
                    </button>
                    <div className="min-w-0 flex-1">
                      <p className="truncate text-sm font-medium">{node.displayName}</p>
                      <p className="text-[10px] uppercase tracking-wide opacity-80">
                        {node.family} · {node.status}
                      </p>
                    </div>
                    <button
                      type="button"
                      className={`rounded border px-2 py-0.5 text-[11px] text-noc-textPrimary ${
                        armed ? "border-noc-cyan" : "border-noc-border"
                      }`}
                      aria-label={`Connect ${node.entryId}`}
                      onClick={(event) => {
                        event.stopPropagation();
                        onConnect(node.uid);
                      }}
                    >
                      Connect
                    </button>
                    <button
                      type="button"
                      className="rounded border border-noc-border px-2 py-0.5 text-[11px]"
                      aria-label={`${node.entryId} input port`}
                      onClick={(event) => {
                        event.stopPropagation();
                        onPort(node.uid, "in");
                      }}
                    >
                      in
                    </button>
                    <button
                      type="button"
                      className="rounded border border-noc-border px-2 py-0.5 text-[11px] text-noc-textMuted"
                      aria-label={`Remove ${node.entryId}`}
                      onClick={(event) => {
                        event.stopPropagation();
                        onRemove(node.uid);
                      }}
                    >
                      Remove
                    </button>
                  </div>
                </article>
              </li>
            );
          })}
        </ul>
      )}
      {edges.length > 0 && (
        <ul className="mt-4 space-y-1" aria-label="Connections">
          {edges.map((edge) => (
            <li key={edge.id} data-testid="builder-edge" className="text-xs text-noc-textSecondary">
              {label(edge.source)} → {label(edge.target)}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
