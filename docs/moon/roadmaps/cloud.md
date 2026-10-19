> Split out of [`docs/moon/ROADMAP.md`](../ROADMAP.md) like the other themes. See that file for the Anchor Index.

---

## J — Cloud Runner (v1.1.0)

> Run WSmart-Route simulations on GCP, AWS and Azure, with Firebase or Supabase as the control plane, and launch, monitor and manage them from WSmart-Route Studio. Feature set agreed with the owner on 2026-10-19. Epic: [#120](https://github.com/ACFHarbinger/WSmart-Route/issues/120), milestone `v1.1.0`, project "WSmart+ Route".

### §J.1 — Design

**Layout**

| Part | Location |
| --- | --- |
| Runner (contracts, runtime, backends, control planes, scheduler, cost, prediction, metrics) | `logic/controllers/cloud/` |
| Container image, control-plane schemas and security rules | `infra/cloud/` |
| Studio commands and screens | `app/src-tauri/src/commands/cloud*.rs`, `app/src/pages/cloud/` |
| Google Drive sync (unchanged, a different tool) | `logic/cloud/` |

**Flow**

```
Studio / CLI ──submit──▶ ControlPlane (Firebase | Supabase | local SQLite)
                              │  queue, job/task state, events, metrics, results index
                              ▼
                         Scheduler ──place──▶ ComputeBackend (GCP | AWS | Azure | local)
                              │                  batch · serverless containers · VMs
                              ▼
                      Task runtime (release image)
                      stage inputs ◀── ObjectStore (GCS | S3 | Blob | filesystem)
                      run test_sim with daily checkpoints, heartbeats, metrics
                      upload checkpoints, logs and results ──▶ ObjectStore
                              │
Studio / CLI ◀──download into outputs/──┘
```

**Decisions**

- **Job:** one `main.py test_sim ...` command or a manifest of cells, in the format of the rerun manifests; a single command is a one-cell manifest. One task per cell, resumed from its daily checkpoints.
- **Compute:** all three kinds on every provider, selectable per job:

  | | Batch | Serverless containers | VMs |
  | --- | --- | --- | --- |
  | GCP | GCP Batch | Cloud Run Jobs | Compute Engine |
  | AWS | AWS Batch | ECS Fargate | EC2 |
  | Azure | Azure Batch | Container Apps Jobs | Azure VMs |

  Spot or preemptible capacity is a per-job option.
- **Control plane:** Firebase (Firestore, Auth) or Supabase (Postgres, Realtime, Auth) behind one `ControlPlane` interface, plus local SQLite for offline use and tests. Neither runs simulations.
- **Code:** a release image built from `logic/package/build_release.py`, tagged by commit, pushed to Artifact Registry, ECR and ACR. Data and model weights are not baked in.
- **Data:** content-addressed inputs and all outputs in the provider's object storage; results download into the `outputs/` layout.
- **Gurobi:** a WLS key from the provider's secret manager, or a license supplied with the job. Jobs that need Gurobi fail before starting without one.
- **Credentials:** local CLI logins (gcloud ADC, AWS profiles, `az login`) or service-account keys in the OS keychain, referenced by profile name.
- **Release:** an optional `[cloud]` extra with lazily imported provider SDKs.
- **Testing:** in-process fakes only in this round; no real cloud calls.

### §J.2 — Work packages

| Package | Issue | Owner | Priority | Depends on |
| --- | --- | --- | --- | --- |
| CR-01 Core contracts, job model and local backend | [#121](https://github.com/ACFHarbinger/WSmart-Route/issues/121) | Claude | P0 | — |
| CR-02 Task runtime inside the container | [#122](https://github.com/ACFHarbinger/WSmart-Route/issues/122) | Claude | P0 | CR-01 |
| CR-03 Release container image and registries | [#123](https://github.com/ACFHarbinger/WSmart-Route/issues/123) | Claude | P0 | CR-02 |
| CR-04 Control planes: Firebase and Supabase | [#124](https://github.com/ACFHarbinger/WSmart-Route/issues/124) | Claude | P0 | CR-01 |
| CR-05 GCP backends | [#125](https://github.com/ACFHarbinger/WSmart-Route/issues/125) | Claude | P1 | CR-01–03 |
| CR-06 AWS backends | [#126](https://github.com/ACFHarbinger/WSmart-Route/issues/126) | Gemini | P1 | CR-05 structure |
| CR-07 Azure backends | [#127](https://github.com/ACFHarbinger/WSmart-Route/issues/127) | Gemini | P1 | CR-05 structure |
| CR-08 Credentials and secrets | [#128](https://github.com/ACFHarbinger/WSmart-Route/issues/128) | Claude | P1 | CR-01 |
| CR-09 Scheduler: placement, auto-scaling, warm pool, retries, limits | [#129](https://github.com/ACFHarbinger/WSmart-Route/issues/129) | Claude | P1 | CR-01, CR-02, CR-11 |
| CR-10 Cost: estimates, budget caps, accrual | [#130](https://github.com/ACFHarbinger/WSmart-Route/issues/130) | Claude | P1 | CR-01, CR-11 |
| CR-11 Runtime and memory prediction | [#131](https://github.com/ACFHarbinger/WSmart-Route/issues/131) | Claude | P2 | CR-12 |
| CR-12 Instrumentation and tracking-module updates | [#132](https://github.com/ACFHarbinger/WSmart-Route/issues/132) | Claude | P1 | CR-01, CR-02 |
| CR-13 Studio backend: control-plane client, keychain, downloads | [#133](https://github.com/ACFHarbinger/WSmart-Route/issues/133) | Gemini | P1 | CR-01, CR-04, CR-08 |
| CR-14 Studio Launcher | [#134](https://github.com/ACFHarbinger/WSmart-Route/issues/134) | Gemini | P1 | CR-13, CR-10 |
| CR-15 Studio Live monitor and manager actions | [#135](https://github.com/ACFHarbinger/WSmart-Route/issues/135) | Gemini | P1 | CR-13, CR-12 |
| CR-16 Studio Analysis dashboard | [#136](https://github.com/ACFHarbinger/WSmart-Route/issues/136) | Gemini | P2 | CR-15, CR-10, CR-11 |
| CR-17 Docs, release extra and test gate | [#137](https://github.com/ACFHarbinger/WSmart-Route/issues/137) | Claude | P2 | all |

**Order:** CR-01 → CR-02, CR-04, CR-08 → CR-03, CR-12 → CR-05, CR-11, CR-13 → CR-06, CR-07, CR-09, CR-10, CR-14, CR-15 → CR-16 → CR-17.

**Split:** Claude writes the contracts, runtime, control planes, GCP, scheduling, cost, prediction and observability, and reviews and integrates. Gemini writes AWS, Azure and the Studio packages against the contracts. Coordination: `.agent/bus/<date>.md` and `.agent/tasks/cloud-runner-2026-10-19.md`.

### §J.3 — Future work (not in v1.1.0)

- [ ] **First real smoke test per provider:** a one-day Rio Maior 100 cell on each compute kind, behind a small budget cap, with real credentials supplied by the owner. Nothing in v1.1.0 is validated against a real API.
- [ ] **Emulator-based tests:** Firebase emulator, local Supabase, LocalStack, Azurite and a fake GCS server, between the in-process fakes and the real smoke.
- [ ] **Brokered, multi-user control plane:** users sign in to Firebase or Supabase, and a backend holds the provider credentials and starts jobs on their behalf.
- [ ] **More platforms:** Oracle, Cloudflare and Kubernetes backends (the `infra/cloud/` templates for Oracle and Cloudflare predate this design).
