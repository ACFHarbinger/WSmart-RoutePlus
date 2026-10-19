"""Authentication of the cloud runner: provider credentials, secrets and profiles.

- :mod:`.credentials`: CLI login detection, the OS keychain store and SDK credential objects;
- :mod:`.secret_refs`: ``WSR_SECRET_REFS``, resolved by the task runtime with the machine's identity;
- :mod:`.config`: the cloud profiles file (no secrets in it).

Provider SDKs and ``keyring`` are imported only where a credential is built.
"""
