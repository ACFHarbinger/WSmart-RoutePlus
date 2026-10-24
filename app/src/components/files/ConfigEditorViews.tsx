import { FolderOpen } from "lucide-react";
import { OpenPathToolbar } from "../common/OpenPathToolbar";

export type FieldType = "string" | "number" | "boolean";

export function inferFieldType(value: string): FieldType {
  if (value === "true" || value === "false") return "boolean";
  if (value !== "" && !isNaN(Number(value)) && !value.startsWith("[")) return "number";
  return "string";
}

export function rowsToYaml(rows: Array<{ key: string; value: string }>): string {
  return rows.map((r) => `${r.key}: ${r.value}`).join("\n");
}

export function parseYamlFlat(yaml: string): Array<{ key: string; value: string }> {
  const rows: Array<{ key: string; value: string }> = [];
  const stack: string[] = [];
  for (const line of yaml.split("\n")) {
    const stripped = line.replace(/\t/g, "  ");
    const trimmed = stripped.trimStart();
    if (!trimmed || trimmed.startsWith("#")) continue;

    const indent = (stripped.length - trimmed.length) / 2;
    stack.length = indent;

    const colonIdx = trimmed.indexOf(":");
    if (colonIdx === -1) continue;

    const key = trimmed.slice(0, colonIdx).trim();
    const value = trimmed.slice(colonIdx + 1).trim();
    stack[indent] = key;

    if (value && !value.startsWith("#")) {
      rows.push({ key: stack.filter(Boolean).join("."), value });
    }
  }
  return rows;
}

export function ConfigFormView({
  rows,
  updateFormRow,
}: {
  rows: Array<{ key: string; value: string }>;
  updateFormRow: (index: number, field: "key" | "value", val: string) => void;
}) {
  return (
    <div className="card overflow-auto max-h-[60vh] space-y-2 p-2">
      {rows.map((r, i) => {
        const ftype = inferFieldType(r.value);
        return (
          <div key={`${r.key}-${i}`} className="flex items-center gap-3 text-xs">
            <span className="font-mono text-accent-secondary w-1/3 truncate" title={r.key}>
              {r.key}
            </span>
            {ftype === "boolean" ? (
              <label className="flex items-center gap-2 text-gray-300">
                <input
                  type="checkbox"
                  className="accent-accent-primary"
                  checked={r.value === "true"}
                  onChange={(e) =>
                    updateFormRow(i, "value", e.target.checked ? "true" : "false")
                  }
                />
                {r.value}
              </label>
            ) : ftype === "number" ? (
              <input
                type="number"
                className="input-base font-mono text-xs flex-1"
                value={r.value}
                onChange={(e) => updateFormRow(i, "value", e.target.value)}
              />
            ) : (
              <input
                type="text"
                className="input-base font-mono text-xs flex-1"
                value={r.value}
                onChange={(e) => updateFormRow(i, "value", e.target.value)}
              />
            )}
          </div>
        );
      })}
      {rows.length === 0 && (
        <p className="text-canvas-muted text-sm text-center py-8">
          No scalar key-value pairs found.
        </p>
      )}
    </div>
  );
}

export function ConfigTableView({
  rows,
}: {
  rows: Array<{ key: string; value: string }>;
}) {
  return (
    <div className="card overflow-auto">
      <table className="w-full text-xs">
        <thead>
          <tr className="border-b border-canvas-border">
            <th className="text-left py-2 px-3 text-canvas-muted font-medium w-1/2">Key</th>
            <th className="text-left py-2 px-3 text-canvas-muted font-medium w-1/2">Value</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-canvas-border">
          {rows.map((r, i) => (
            <tr key={i} className="hover:bg-canvas-hover">
              <td className="py-1.5 px-3 font-mono text-accent-secondary">{r.key}</td>
              <td className="py-1.5 px-3 text-gray-300">{r.value}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function ConfigDiffView({
  openFile,
  diffPath,
  diffContent,
  changedKeys,
  diffMap,
  filePath,
  projectRoot,
}: {
  openFile: (target: "primary" | "diff") => void;
  diffPath: string | null;
  diffContent: string;
  changedKeys: string[];
  diffMap: Record<string, { primary?: string; diff?: string }>;
  filePath: string | null;
  projectRoot: string | null;
}) {
  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        <button
          onClick={() => openFile("diff")}
          className="btn-ghost flex items-center gap-2 text-sm"
        >
          <FolderOpen size={13} />
          {diffPath ? "Change comparison file…" : "Open comparison file…"}
        </button>
        {diffPath ? (
          <OpenPathToolbar
            path={diffPath}
            projectRoot={projectRoot}
            kind="config"
            chipClassName="max-w-md"
          />
        ) : null}
      </div>

      {!diffContent && (
        <div className="card text-canvas-muted text-sm">
          Open a second YAML file to compare against the primary config.
        </div>
      )}

      {diffContent && (
        <div className="card overflow-auto">
          <p className="text-xs text-canvas-muted mb-3 flex flex-wrap items-center gap-1.5">
            <span>{changedKeys.length} difference(s) between</span>
            {filePath ? (
              <OpenPathToolbar
                path={filePath}
                projectRoot={projectRoot}
                kind="config"
              />
            ) : null}
            <span>and</span>
            {diffPath ? (
              <OpenPathToolbar
                path={diffPath}
                projectRoot={projectRoot}
                kind="config"
              />
            ) : null}
          </p>
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-canvas-border">
                <th className="text-left py-2 px-3 text-canvas-muted font-medium">Key</th>
                <th className="text-left py-2 px-3 text-canvas-muted font-medium">Primary</th>
                <th className="text-left py-2 px-3 text-canvas-muted font-medium">Comparison</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-canvas-border">
              {changedKeys.map((k) => (
                <tr key={k} className="bg-accent-warning/5">
                  <td className="py-1.5 px-3 font-mono text-accent-secondary">{k}</td>
                  <td className="py-1.5 px-3 text-accent-danger font-mono">
                    {diffMap[k]?.primary || <span className="text-canvas-muted italic">—</span>}
                  </td>
                  <td className="py-1.5 px-3 text-accent-success font-mono">
                    {diffMap[k]?.diff || <span className="text-canvas-muted italic">—</span>}
                  </td>
                </tr>
              ))}
              {changedKeys.length === 0 && (
                <tr>
                  <td colSpan={3} className="py-8 text-center text-accent-success">
                    No differences found — configs are identical.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
