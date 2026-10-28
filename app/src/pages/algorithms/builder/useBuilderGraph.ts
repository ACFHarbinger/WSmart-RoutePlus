import { useCallback, useEffect, useRef, useState } from "react";
import type { CatalogEntry } from "../../../types/algorithms";
import {
  declaredParameters,
  entryError,
  entryLabel,
  entryRole,
  isNotRoutePolicy,
} from "./catalogView";
import { draftPersist, readDraft, restoreCanvas, writeDraft } from "./draftBridge";
import type { BuilderEdge, BuilderNode, PendingPort } from "./graphTypes";
import { linkRefusal } from "./linkRules";

export function useBuilderGraph() {
  const [nodes, setNodes] = useState<BuilderNode[]>([]);
  const [edges, setEdges] = useState<BuilderEdge[]>([]);
  const [pending, setPending] = useState<PendingPort | null>(null);
  const [selectedUid, setSelectedUid] = useState<string | null>(null);
  const [alert, setAlert] = useState<string | null>(null);
  const [ready, setReady] = useState(false);
  const seq = useRef(1);

  useEffect(() => {
    let seeded = false;
    const seed = () => {
      if (seeded) return;
      seeded = true;
      const restored = restoreCanvas(readDraft());
      if (restored.nodes.length > 0) {
        setNodes(restored.nodes);
        setEdges(restored.edges);
      }
      setReady(true);
    };
    const persist = draftPersist();
    if (!persist || persist.hasHydrated()) {
      seed();
      return;
    }
    return persist.onFinishHydration(() => seed());
  }, []);

  useEffect(() => {
    if (!ready) return;
    writeDraft(nodes, edges);
  }, [ready, nodes, edges]);

  const addEntry = useCallback((entry: CatalogEntry) => {
    if (isNotRoutePolicy(entry)) return;
    const error = entryError(entry);
    if (error) {
      setAlert(error);
      return;
    }
    const parameters: Record<string, string> = {};
    for (const param of declaredParameters(entry)) {
      if (param.default !== undefined) parameters[param.name] = String(param.default);
    }
    const uid = `${entry.id}-${seq.current}`;
    seq.current += 1;
    const status = (entry as { status?: string }).status;
    const node: BuilderNode = {
      uid,
      entryId: entry.id,
      family: entry.family,
      displayName: entryLabel(entry),
      status: typeof status === "string" ? status : "present",
      role: entryRole(entry),
      parameters,
    };
    setNodes((prev) => [...prev, node]);
    setSelectedUid(uid);
    setAlert(null);
  }, []);

  const commitLink = useCallback(
    (sourceUid: string, targetUid: string) => {
      if (sourceUid === targetUid) {
        setAlert("Refused: a module cannot connect to itself.");
        return;
      }
      const source = nodes.find((node) => node.uid === sourceUid);
      const target = nodes.find((node) => node.uid === targetUid);
      if (!source || !target) return;
      const reason = linkRefusal(source, target);
      if (reason) {
        setAlert(reason);
        return;
      }
      setEdges((prev) => {
        if (prev.some((edge) => edge.source === sourceUid && edge.target === targetUid)) return prev;
        return [...prev, { id: `e-${sourceUid}-${targetUid}`, source: sourceUid, target: targetUid }];
      });
      setAlert(null);
    },
    [nodes]
  );

  const onPort = useCallback(
    (uid: string, side: "in" | "out") => {
      if (!pending) {
        setPending({ uid, side });
        return;
      }
      if (pending.uid === uid && pending.side === side) {
        setPending(null);
        return;
      }
      if (pending.side === side) {
        setPending({ uid, side });
        return;
      }
      const sourceUid = pending.side === "out" ? pending.uid : uid;
      const targetUid = pending.side === "out" ? uid : pending.uid;
      setPending(null);
      commitLink(sourceUid, targetUid);
    },
    [commitLink, pending]
  );

  const onConnectButton = useCallback(
    (uid: string) => {
      if (!pending) {
        setPending({ uid, side: "out" });
        return;
      }
      if (pending.uid === uid) {
        setPending(null);
        return;
      }
      const sourceUid = pending.side === "out" ? pending.uid : uid;
      const targetUid = pending.side === "out" ? uid : pending.uid;
      setPending(null);
      commitLink(sourceUid, targetUid);
    },
    [commitLink, pending]
  );

  const removeNode = useCallback((uid: string) => {
    setNodes((prev) => prev.filter((node) => node.uid !== uid));
    setEdges((prev) => prev.filter((edge) => edge.source !== uid && edge.target !== uid));
    setSelectedUid((current) => (current === uid ? null : current));
    setPending((current) => (current?.uid === uid ? null : current));
  }, []);

  const setParameter = useCallback((uid: string, name: string, value: string) => {
    setNodes((prev) =>
      prev.map((node) =>
        node.uid === uid ? { ...node, parameters: { ...node.parameters, [name]: value } } : node
      )
    );
  }, []);

  const clearCanvas = useCallback(() => {
    setNodes([]);
    setEdges([]);
    setPending(null);
    setSelectedUid(null);
    setAlert(null);
  }, []);

  return {
    nodes,
    edges,
    pending,
    selectedUid,
    setSelectedUid,
    alert,
    addEntry,
    onPort,
    onConnectButton,
    removeNode,
    setParameter,
    clearCanvas,
  };
}
