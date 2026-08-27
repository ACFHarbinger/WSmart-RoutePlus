# terraform/

> **Optional.** Only relevant if the optional REST API / remote-inference
> backend (`docs/moon/roadmaps/new_features.md` §E.3) is stood up behind
> managed cloud resources (registry, a GPU-backed compute instance, a k8s
> cluster). Running everything locally needs none of this — delete
> `infra/global/terraform/` entirely in that case.

No provider is wired up yet — this is a starting point, not a real stack.

```bash
cd infra/global/terraform
terraform init
terraform plan -var-file=environments/dev.tfvars
terraform apply -var-file=environments/dev.tfvars
```

| File | Purpose |
| --- | --- |
| `versions.tf` | Required Terraform + provider versions, remote state backend (commented, fill in before first `init`) |
| `variables.tf` | Input variables |
| `main.tf` | Resources — currently empty, add your provider blocks and resources here |
| `outputs.tf` | Values to surface after `apply` (e.g. registry URL, instance endpoint) |
| `environments/*.tfvars` | Per-environment variable values |

> **TODO:** Pick a cloud provider, uncomment/configure the matching provider
> block in `versions.tf`, and replace the placeholder resources in `main.tf`.
