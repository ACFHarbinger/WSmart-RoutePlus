"""AWS provider package for WSmart-Route cloud runner.

Provides compute backends (Batch, ECS Fargate, EC2), S3 object store,
Secrets Manager secret store, ECR image registry, and pricing estimates.
No provider SDKs are imported at module level.
"""

from __future__ import annotations

from logic.controllers.cloud.providers.aws.batch import AwsBatchBackend
from logic.controllers.cloud.providers.aws.pricing import price_per_hour
from logic.controllers.cloud.providers.aws.registry import EcrImageRegistry
from logic.controllers.cloud.providers.aws.secrets import AwsSecretsManagerStore
from logic.controllers.cloud.providers.aws.serverless import EcsFargateBackend
from logic.controllers.cloud.providers.aws.storage import S3ObjectStore
from logic.controllers.cloud.providers.aws.vm import Ec2Backend

__all__ = [
    "AwsBatchBackend",
    "AwsSecretsManagerStore",
    "EcrImageRegistry",
    "EcsFargateBackend",
    "Ec2Backend",
    "S3ObjectStore",
    "price_per_hour",
]
