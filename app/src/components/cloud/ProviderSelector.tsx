import { useEffect, useState } from "react";
import {
  Cloud,
  CheckCircle2,
  KeyRound,
  RefreshCw,
  Zap,
  ShieldAlert,
} from "lucide-react";
import type {
  CloudBackendEntry,
  CloudComputeKind,
  CloudProvider,
} from "../../types/cloud";
import { hasCloudCredential } from "../../utils/cloud";
import { CredentialModal } from "./CredentialModal";

export interface ProviderSelectorProps {
  provider: CloudProvider;
  compute: CloudComputeKind;
  cloudProfile: string;
  onProviderChange: (provider: CloudProvider) => void;
  onComputeChange: (compute: CloudComputeKind) => void;
  onProfileChange: (profile: string) => void;
  backends: CloudBackendEntry[];
  loadingBackends?: boolean;
  onRefreshBackends?: () => void;
}

const PROVIDERS: Array<{ id: CloudProvider; label: string; icon: string }> = [
  { id: "local", label: "Local Process", icon: "💻" },
  { id: "gcp", label: "Google Cloud", icon: "🌐" },
  { id: "aws", label: "Amazon Web Services", icon: "☁️" },
  { id: "azure", label: "Microsoft Azure", icon: "🔷" },
];

const COMPUTE_KINDS: Array<{
  id: CloudComputeKind;
  label: string;
  desc: string;
}> = [
  { id: "process", label: "Process", desc: "Local sub-processes / threads" },
  { id: "batch", label: "Batch", desc: "Managed batch queue (Cloud Batch / AWS Batch / Azure Batch)" },
  { id: "serverless", label: "Serverless", desc: "Serverless containers (Cloud Run / Fargate / Container Apps)" },
  { id: "vm", label: "Virtual Machines", desc: "Dedicated worker VMs (Compute Engine / EC2 / Azure VM)" },
];

export function ProviderSelector({
  provider,
  compute,
  cloudProfile,
  onProviderChange,
  onComputeChange,
  onProfileChange,
  backends,
  loadingBackends,
  onRefreshBackends,
}: ProviderSelectorProps) {
  const [hasCredential, setHasCredential] = useState<boolean>(false);
  const [isCredModalOpen, setIsCredModalOpen] = useState<boolean>(false);

  // Check keychain credential whenever provider or profile changes
  useEffect(() => {
    let active = true;
    if (provider === "local") {
      setHasCredential(true);
      return;
    }
    hasCloudCredential(cloudProfile, provider)
      .then((has) => {
        if (active) setHasCredential(has);
      })
      .catch(() => {
        if (active) setHasCredential(false);
      });
    return () => {
      active = false;
    };
  }, [provider, cloudProfile]);

  // Find backend capabilities if registered
  const activeBackend = backends.find(
    (b) => b.provider === provider && b.compute === compute
  );

  return (
    <div className="card bg-canvas-surface border border-canvas-border rounded-xl p-4 space-y-4">
      <div className="flex items-center justify-between border-b border-canvas-border pb-3">
        <div className="flex items-center gap-2">
          <Cloud size={16} className="text-accent-primary" />
          <h3 className="text-sm font-semibold text-gray-200">
            Target Cloud & Compute Environment
          </h3>
        </div>
        <div className="flex items-center gap-2">
          {onRefreshBackends && (
            <button
              type="button"
              onClick={onRefreshBackends}
              disabled={loadingBackends}
              className="text-xs text-canvas-muted hover:text-gray-200 flex items-center gap-1 transition-colors"
              title="Refresh detected backends"
            >
              <RefreshCw
                size={12}
                className={loadingBackends ? "animate-spin" : ""}
              />
              <span>Refresh</span>
            </button>
          )}
        </div>
      </div>

      {/* Provider selection */}
      <div className="space-y-1.5">
        <label className="text-xs font-medium text-canvas-muted">Provider</label>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
          {PROVIDERS.map((p) => {
            const isSelected = provider === p.id;
            const isConfigured =
              p.id === "local" || backends.some((b) => b.provider === p.id);
            return (
              <button
                key={p.id}
                type="button"
                onClick={() => onProviderChange(p.id)}
                className={`flex flex-col items-center justify-center p-3 rounded-xl border text-center transition-all ${
                  isSelected
                    ? "bg-accent-primary/10 border-accent-primary text-gray-100 shadow-sm"
                    : "bg-canvas-elevated/40 border-canvas-border text-canvas-muted hover:border-canvas-muted/60 hover:text-gray-200"
                }`}
              >
                <span className="text-lg mb-1">{p.icon}</span>
                <span className="text-xs font-semibold">{p.label}</span>
                <span className="text-[10px] mt-0.5 text-canvas-muted">
                  {isConfigured ? "Available" : "SDK Optional"}
                </span>
              </button>
            );
          })}
        </div>
      </div>

      {/* Compute kind */}
      <div className="space-y-1.5">
        <div className="flex items-center justify-between">
          <label className="text-xs font-medium text-canvas-muted">
            Compute Topology
          </label>
          {activeBackend?.capabilities.supports_spot && (
            <span className="text-[10px] text-accent-success flex items-center gap-1">
              <Zap size={11} /> Spot instances supported
            </span>
          )}
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
          {COMPUTE_KINDS.map((c) => {
            const isSelected = compute === c.id;
            return (
              <button
                key={c.id}
                type="button"
                onClick={() => onComputeChange(c.id)}
                className={`p-2.5 rounded-lg border text-left transition-all ${
                  isSelected
                    ? "bg-canvas-elevated border-accent-primary text-gray-100"
                    : "bg-canvas-elevated/20 border-canvas-border text-canvas-muted hover:border-canvas-muted/60 hover:text-gray-200"
                }`}
              >
                <div className="flex items-center justify-between">
                  <span className="text-xs font-semibold">{c.label}</span>
                  {isSelected && (
                    <CheckCircle2 size={13} className="text-accent-primary" />
                  )}
                </div>
                <p className="text-[11px] text-canvas-muted mt-0.5">{c.desc}</p>
              </button>
            );
          })}
        </div>
      </div>

      {/* Profile & Keychain info (for non-local providers) */}
      {provider !== "local" && (
        <div className="bg-canvas-elevated/30 border border-canvas-border rounded-lg p-3 flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs">
          <div className="flex items-center gap-2">
            <label className="text-canvas-muted whitespace-nowrap">
              Credential Profile:
            </label>
            <input
              type="text"
              className="input-base font-mono text-xs w-36 py-1 px-2"
              value={cloudProfile}
              onChange={(e) => onProfileChange(e.target.value)}
              placeholder="default"
            />
          </div>

          <div className="flex items-center gap-2">
            {hasCredential ? (
              <div className="flex items-center gap-1 text-accent-success font-medium">
                <CheckCircle2 size={13} />
                <span>Keychain credential set</span>
              </div>
            ) : (
              <div className="flex items-center gap-1 text-accent-warning">
                <ShieldAlert size={13} />
                <span>No keychain credential</span>
              </div>
            )}
            <button
              type="button"
              onClick={() => setIsCredModalOpen(true)}
              className="btn-ghost text-xs px-2.5 py-1 border border-canvas-border flex items-center gap-1"
            >
              <KeyRound size={12} />
              <span>{hasCredential ? "Manage Key" : "Add Key"}</span>
            </button>
          </div>
        </div>
      )}

      {/* Credential Modal */}
      <CredentialModal
        isOpen={isCredModalOpen}
        onClose={() => setIsCredModalOpen(false)}
        provider={provider}
        profile={cloudProfile}
        hasCredential={hasCredential}
        onCredentialUpdated={() => {
          hasCloudCredential(cloudProfile, provider)
            .then(setHasCredential)
            .catch(() => setHasCredential(false));
        }}
      />
    </div>
  );
}
