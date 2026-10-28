import type { ExecutionTraceDocument, SolutionRecord, TraceEvent } from "../../../types/algorithms";

/** Loose view of a trace event so either JSON or camelCase client fields render. */
interface TraceEventFields {
  id: string;
  parent_id?: string | null;
  parentId?: string | null;
  name?: string;
  kind?: string;
  changed_solution?: boolean;
  changedSolution?: boolean;
  arguments?: Record<string, unknown>;
  before?: SolutionRecord | null;
  after?: SolutionRecord | null;
  interior?: string | null;
}

export interface FlowNode {
  event: TraceEvent;
  children: FlowNode[];
}

export type FlowRow =
  | { kind: "event"; event: TraceEvent; depth: number }
  | { kind: "gap"; hiddenCount: number; depth: number };

export interface VisibleFlowOptions {
  expandedStageIds: ReadonlySet<string>;
  solutionChangesOnly: boolean;
  /** When false and operators exceed 40, keep the first 20 and the last 10. */
  expandOperators: boolean;
}

const OPERATOR_LIMIT = 40;
const OPERATOR_HEAD = 20;
const OPERATOR_TAIL = 10;

export function eventFields(event: TraceEvent): TraceEventFields {
  return event as unknown as TraceEventFields;
}

export function eventName(event: TraceEvent): string {
  const name = eventFields(event).name;
  return typeof name === "string" && name ? name : eventFields(event).id;
}

export function eventKind(event: TraceEvent): string {
  const kind = eventFields(event).kind;
  return typeof kind === "string" ? kind : "";
}

export function eventParentId(event: TraceEvent): string | null {
  const parent = eventFields(event).parent_id ?? eventFields(event).parentId;
  return typeof parent === "string" ? parent : null;
}

export function eventChanged(event: TraceEvent): boolean {
  return Boolean(eventFields(event).changed_solution ?? eventFields(event).changedSolution);
}

export function eventInterior(event: TraceEvent): string | null {
  const interior = eventFields(event).interior;
  return typeof interior === "string" ? interior : null;
}

export function eventArguments(event: TraceEvent): Record<string, unknown> {
  const args = eventFields(event).arguments;
  return args && typeof args === "object" ? args : {};
}

export function solutionRoutes(record: SolutionRecord | null | undefined): number[][] {
  if (!record || !Array.isArray(record.routes)) return [];
  return record.routes.filter((route) => Array.isArray(route));
}

export function formatRoutes(routes: number[][]): string {
  if (!routes.length) return "—";
  return routes.map((route) => route.join(" → ")).join(" | ");
}

export function coordinatesFromInstance(instance: unknown): Array<[number, number]> | null {
  if (!instance || typeof instance !== "object") return null;
  const raw = (instance as Record<string, unknown>).coordinates ?? (instance as Record<string, unknown>).coords;
  if (!Array.isArray(raw)) return null;
  const points: Array<[number, number]> = [];
  for (const item of raw) {
    if (!Array.isArray(item) || item.length < 2) return null;
    const x = Number(item[0]);
    const y = Number(item[1]);
    if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
    points.push([x, y]);
  }
  return points;
}

/** Group trace events under parent_id. Missing parents stay at the root, in file order. */
export function buildFlow(events: TraceEvent[] | null | undefined): FlowNode[] {
  const list = Array.isArray(events) ? events : [];
  const nodes = new Map<string, FlowNode>();
  for (const event of list) {
    const id = eventFields(event).id;
    if (!id || nodes.has(id)) continue;
    nodes.set(id, { event, children: [] });
  }
  const roots: FlowNode[] = [];
  for (const event of list) {
    const id = eventFields(event).id;
    const node = id ? nodes.get(id) : undefined;
    if (!node) continue;
    const parentId = eventParentId(event);
    const parent = parentId ? nodes.get(parentId) : undefined;
    if (parent && parent !== node) parent.children.push(node);
    else roots.push(node);
  }
  return roots;
}

function keepEvent(event: TraceEvent, solutionChangesOnly: boolean): boolean {
  if (!solutionChangesOnly) return true;
  if (eventInterior(event) === "not_instrumented") return true;
  if (eventKind(event) === "incumbent") return true;
  return eventChanged(event);
}

function appendVisible(node: FlowNode, depth: number, options: VisibleFlowOptions, into: FlowRow[]): void {
  const show = keepEvent(node.event, options.solutionChangesOnly);
  if (show) into.push({ kind: "event", event: node.event, depth });
  const isStage = eventKind(node.event) === "stage";
  if (isStage && !options.expandedStageIds.has(eventFields(node.event).id)) return;
  const childDepth = show ? depth + 1 : depth;
  for (const child of node.children) appendVisible(child, childDepth, options, into);
}

/**
 * Visible flow rows. More than 40 operator events collapse to the first 20,
 * one gap row, and the last 10. Hidden operators are omitted, not mounted.
 */
export function visibleFlowRows(roots: FlowNode[], options: VisibleFlowOptions): FlowRow[] {
  const rows: FlowRow[] = [];
  for (const root of roots) appendVisible(root, 0, options, rows);
  if (options.expandOperators) return rows;
  const operatorIndexes: number[] = [];
  rows.forEach((row, index) => {
    if (row.kind === "event" && eventKind(row.event) === "operator") operatorIndexes.push(index);
  });
  if (operatorIndexes.length <= OPERATOR_LIMIT) return rows;
  const hidden = new Set(operatorIndexes.slice(OPERATOR_HEAD, operatorIndexes.length - OPERATOR_TAIL));
  const compact: FlowRow[] = [];
  let gapInserted = false;
  rows.forEach((row, index) => {
    if (!hidden.has(index)) {
      compact.push(row);
      return;
    }
    if (!gapInserted) {
      compact.push({ kind: "gap", hiddenCount: hidden.size, depth: row.kind === "event" ? row.depth : 0 });
      gapInserted = true;
    }
  });
  return compact;
}

export function stageIds(document: ExecutionTraceDocument | null | undefined): Set<string> {
  const ids = new Set<string>();
  for (const event of document?.events ?? []) {
    if (eventKind(event) === "stage") ids.add(eventFields(event).id);
  }
  return ids;
}
