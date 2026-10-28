export interface BuilderNode {
  uid: string;
  entryId: string;
  family: string;
  displayName: string;
  status: string;
  role?: string;
  parameters: Record<string, string>;
}

export interface BuilderEdge {
  id: string;
  source: string;
  target: string;
}

export interface PendingPort {
  uid: string;
  side: "in" | "out";
}
