/**
 * Thin TypeScript client wrapping Tauri cloud commands and events (CR-13).
 */

import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import type {
  CloudBackendEntry,
  CloudCancelResult,
  CloudDownloadProgressEvent,
  CloudDownloadResult,
  CloudJobStatus,
  CloudJobStatusUpdateEvent,
  CloudJobSummary,
  CloudRetryResult,
  CloudSubmitRequest,
  CloudSubmitResult,
  CloudTaskLogsResult,
  CloudWorkerTickResult,
} from "../../types/cloud";

export interface CloudCliOptions {
  projectRoot?: string;
  root?: string;
  pythonExecutable?: string;
}

/**
 * List summary status of all cloud jobs.
 */
export async function listCloudJobs(options?: CloudCliOptions): Promise<CloudJobSummary[]> {
  return invoke<CloudJobSummary[]>("cloud_list_jobs", {
    projectRoot: options?.projectRoot,
    root: options?.root,
    pythonExecutable: options?.pythonExecutable,
  });
}

/**
 * Retrieve detailed status of a specific cloud job and its tasks.
 */
export async function getCloudJobStatus(
  jobId: string,
  options?: CloudCliOptions
): Promise<CloudJobStatus> {
  return invoke<CloudJobStatus>("cloud_job_status", {
    jobId,
    projectRoot: options?.projectRoot,
    root: options?.root,
    pythonExecutable: options?.pythonExecutable,
  });
}

/**
 * Submit a job to the cloud runner.
 */
export async function submitCloudJob(
  request: CloudSubmitRequest,
  options?: CloudCliOptions
): Promise<CloudSubmitResult> {
  return invoke<CloudSubmitResult>("cloud_submit", {
    request,
    projectRoot: options?.projectRoot,
    root: options?.root,
    pythonExecutable: options?.pythonExecutable,
  });
}

/**
 * Cancel a running or queued cloud job.
 */
export async function cancelCloudJob(
  jobId: string,
  options?: CloudCliOptions
): Promise<CloudCancelResult> {
  return invoke<CloudCancelResult>("cloud_cancel", {
    jobId,
    projectRoot: options?.projectRoot,
    root: options?.root,
    pythonExecutable: options?.pythonExecutable,
  });
}

/**
 * Requeue failed and cancelled tasks for a job.
 */
export async function retryCloudJob(
  jobId: string,
  options?: CloudCliOptions
): Promise<CloudRetryResult> {
  return invoke<CloudRetryResult>("cloud_retry", {
    jobId,
    projectRoot: options?.projectRoot,
    root: options?.root,
    pythonExecutable: options?.pythonExecutable,
  });
}

/**
 * Fetch logs for a specific task attempt.
 */
export async function getCloudTaskLogs(
  taskId: string,
  tail?: number,
  options?: CloudCliOptions
): Promise<CloudTaskLogsResult> {
  return invoke<CloudTaskLogsResult>("cloud_task_logs", {
    taskId,
    tail,
    projectRoot: options?.projectRoot,
    root: options?.root,
    pythonExecutable: options?.pythonExecutable,
  });
}

/**
 * Download job results from object store to local destination directory.
 */
export async function downloadCloudJobResults(
  jobId: string,
  dest?: string,
  options?: CloudCliOptions
): Promise<CloudDownloadResult> {
  return invoke<CloudDownloadResult>("cloud_download", {
    jobId,
    dest,
    projectRoot: options?.projectRoot,
    root: options?.root,
    pythonExecutable: options?.pythonExecutable,
  });
}

/**
 * Verify a local file's SHA-256 hash against expected hexadecimal digest.
 */
export async function verifyCloudDownload(
  filePath: string,
  expectedSha256: string
): Promise<boolean> {
  return invoke<boolean>("cloud_verify_download", {
    filePath,
    expectedSha256,
  });
}

/**
 * List configured compute backends and capabilities.
 */
export async function listCloudBackends(
  options?: CloudCliOptions
): Promise<CloudBackendEntry[]> {
  return invoke<CloudBackendEntry[]>("cloud_backends", {
    projectRoot: options?.projectRoot,
    root: options?.root,
    pythonExecutable: options?.pythonExecutable,
  });
}

/**
 * Run a single worker scheduling pass.
 */
export async function tickCloudWorker(
  job?: string,
  options?: CloudCliOptions
): Promise<CloudWorkerTickResult> {
  return invoke<CloudWorkerTickResult>("cloud_worker_tick", {
    job,
    projectRoot: options?.projectRoot,
    root: options?.root,
    pythonExecutable: options?.pythonExecutable,
  });
}

/**
 * Store a credential secret securely in the OS keychain.
 */
export async function storeCloudCredential(
  profile: string,
  provider: string,
  secret: string
): Promise<void> {
  return invoke<void>("cloud_store_credential", {
    profile,
    provider,
    secret,
  });
}

/**
 * Delete a credential from the OS keychain.
 */
export async function deleteCloudCredential(
  profile: string,
  provider: string
): Promise<void> {
  return invoke<void>("cloud_delete_credential", {
    profile,
    provider,
  });
}

/**
 * Check if a credential is saved in the OS keychain without retrieving the secret.
 */
export async function hasCloudCredential(
  profile: string,
  provider: string
): Promise<boolean> {
  return invoke<boolean>("cloud_has_credential", {
    profile,
    provider,
  });
}

/**
 * Start background polling for a job and listen for status update events.
 * Returns an unlisten cleanup function that cancels both the listener and the backend poller.
 */
export async function watchCloudJob(
  jobId: string,
  onUpdate: (event: CloudJobStatusUpdateEvent) => void,
  intervalMs: number = 2000,
  options?: CloudCliOptions
): Promise<UnlistenFn> {
  const unlisten = await listen<CloudJobStatusUpdateEvent>("cloud://job-updated", (event) => {
    if (event.payload.job_id === jobId) {
      onUpdate(event.payload);
    }
  });

  try {
    await invoke("cloud_watch_job", {
      jobId,
      intervalMs,
      projectRoot: options?.projectRoot,
      root: options?.root,
      pythonExecutable: options?.pythonExecutable,
    });
  } catch (err) {
    unlisten();
    throw err;
  }

  return async () => {
    unlisten();
    invoke("cloud_unwatch_job", { jobId }).catch(() => {});
  };
}

/**
 * Stop watching all jobs.
 */
export async function stopAllCloudWatchers(): Promise<void> {
  return invoke<void>("cloud_stop_all_watchers");
}

/**
 * List all jobs currently being watched by the backend.
 */
export async function listWatchedCloudJobs(): Promise<string[]> {
  return invoke<string[]>("cloud_list_watched_jobs");
}

/**
 * Subscribe to download progress events.
 */
export async function onCloudDownloadProgress(
  onProgress: (event: CloudDownloadProgressEvent) => void
): Promise<UnlistenFn> {
  return listen<CloudDownloadProgressEvent>("cloud://download-progress", (event) => {
    onProgress(event.payload);
  });
}

export {
  computeJobAnalysis,
  exportTasksToCsv,
  extractTaskDuration,
  extractPolicyFromCellId,
  type AnalysisComputeOptions,
} from "./analysis";

