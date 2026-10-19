"""Cloud runner: run simulation jobs locally or on GCP, AWS and Azure.

A job is one ``test_sim`` command or a manifest of cells; each cell runs as a
task that resumes from its daily checkpoints. Design: ``docs/moon/roadmaps/cloud.md``.

Sub-modules
-----------
- :mod:`.model`      — job and task records, the task state machine
- :mod:`.manifest`   — job specs from manifests and single commands
- :mod:`.interfaces` — ``ControlPlane``, ``ComputeBackend``, ``ObjectStore``, ``SecretStore``, ``ImageRegistry``
- :mod:`.service`    — ``CloudService``: submit, schedule, cancel, retry, download
- :mod:`.local`      — SQLite control plane, filesystem store, subprocess backend
- :mod:`.testing`    — fakes and the contract suites every implementation must pass
- :mod:`.cli`        — the ``main.py cloud`` command

Provider SDKs are imported only inside their implementations.

Example::

    python main.py cloud submit --manifest cells.json --max-parallel 4 --wait
"""
