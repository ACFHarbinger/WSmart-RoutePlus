import { useState, type FormEvent } from "react";
import { KeyRound, Check, AlertCircle, X, Trash2 } from "lucide-react";
import {
  deleteCloudCredential,
  storeCloudCredential,
} from "../../utils/cloud";
import type { CloudProvider } from "../../types/cloud";

interface CredentialModalProps {
  isOpen: boolean;
  onClose: () => void;
  provider: CloudProvider;
  profile: string;
  hasCredential: boolean;
  onCredentialUpdated: () => void;
}

export function CredentialModal({
  isOpen,
  onClose,
  provider,
  profile,
  hasCredential,
  onCredentialUpdated,
}: CredentialModalProps) {
  const [secret, setSecret] = useState("");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (!isOpen) return null;

  const handleSave = async (e: FormEvent) => {
    e.preventDefault();
    if (!secret.trim()) {
      setError("Please enter a credential or API key.");
      return;
    }

    setSaving(true);
    setError(null);
    try {
      await storeCloudCredential(profile, provider, secret.trim());
      setSecret("");
      onCredentialUpdated();
      onClose();
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSaving(false);
    }
  };

  const handleDelete = async () => {
    if (!confirm(`Delete saved credential for ${provider}/${profile} from keychain?`)) {
      return;
    }
    setSaving(true);
    setError(null);
    try {
      await deleteCloudCredential(profile, provider);
      onCredentialUpdated();
      onClose();
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4">
      <div className="bg-canvas-surface border border-canvas-border rounded-xl shadow-2xl w-full max-w-md p-5 space-y-4">
        <div className="flex items-center justify-between border-b border-canvas-border pb-3">
          <div className="flex items-center gap-2">
            <KeyRound size={18} className="text-accent-primary" />
            <h3 className="text-sm font-semibold text-gray-100">
              OS Keychain Credential
            </h3>
          </div>
          <button
            onClick={onClose}
            className="text-canvas-muted hover:text-gray-200 transition-colors"
          >
            <X size={16} />
          </button>
        </div>

        <div className="space-y-2 text-xs text-canvas-muted">
          <p>
            Target:{" "}
            <span className="font-mono text-gray-200 font-semibold uppercase">
              {provider}
            </span>{" "}
            / Profile:{" "}
            <span className="font-mono text-gray-200 font-semibold">
              {profile}
            </span>
          </p>
          <p>
            Credentials are encrypted and saved in your operating system keychain
            under service{" "}
            <code className="text-accent-primary font-mono">wsmart-route.cloud</code>
            . Secrets are never stored on disk or logged.
          </p>
          {hasCredential && (
            <div className="flex items-center gap-1.5 text-accent-success font-medium py-1">
              <Check size={14} />
              <span>A credential is currently active for this profile.</span>
            </div>
          )}
        </div>

        <form onSubmit={handleSave} className="space-y-3">
          <div className="space-y-1">
            <label className="text-xs text-canvas-muted">
              {hasCredential ? "Update Secret / Token" : "Secret / Token / JSON Key"}
            </label>
            <textarea
              rows={3}
              className="input-base font-mono text-xs resize-none"
              placeholder="Paste service account key, API secret, or access token..."
              value={secret}
              onChange={(e) => setSecret(e.target.value)}
              disabled={saving}
            />
          </div>

          {error && (
            <div className="flex items-start gap-1.5 text-xs text-accent-danger bg-accent-danger/10 border border-accent-danger/20 rounded p-2">
              <AlertCircle size={14} className="shrink-0 mt-0.5" />
              <span>{error}</span>
            </div>
          )}

          <div className="flex items-center justify-between pt-2 border-t border-canvas-border">
            {hasCredential ? (
              <button
                type="button"
                onClick={handleDelete}
                disabled={saving}
                className="btn-ghost text-accent-danger hover:bg-accent-danger/10 text-xs flex items-center gap-1 px-2.5 py-1.5"
              >
                <Trash2 size={13} />
                Delete
              </button>
            ) : (
              <div />
            )}

            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={onClose}
                disabled={saving}
                className="btn-ghost text-xs px-3 py-1.5"
              >
                Cancel
              </button>
              <button
                type="submit"
                disabled={saving || !secret.trim()}
                className="btn-primary text-xs px-4 py-1.5"
              >
                {saving ? "Saving..." : "Save to Keychain"}
              </button>
            </div>
          </div>
        </form>
      </div>
    </div>
  );
}
