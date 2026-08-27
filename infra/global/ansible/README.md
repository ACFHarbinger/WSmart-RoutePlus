# ansible/

> **Optional.** Config-management/provisioning for the optional REST API /
> remote-inference backend (see `docs/moon/roadmaps/new_features.md` §E.3) on
> bare-metal or VM targets outside the Kubernetes/container world (e.g. a
> bastion host, a self-hosted CI runner, a GPU box). Not needed while
> everything runs locally/in a notebook, or if §E.3 is never built — delete
> this directory in that case.

```bash
ansible-playbook -i inventory/hosts.ini playbook.yml
```

| Path | Purpose |
| --- | --- |
| `ansible.cfg` | Local Ansible config (inventory path, SSH settings) |
| `inventory/hosts.ini` | Target hosts, grouped |
| `playbook.yml` | Entry-point playbook, applies the `app` role |
| `roles/app/` | Example role: installs and configures the API server on a host |
