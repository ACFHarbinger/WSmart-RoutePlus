"""Lazy AWS SDK (boto3) client factories.

boto3 is imported strictly inside functions to keep the core package independent
of provider dependencies.
"""

from __future__ import annotations

from typing import Any, Optional


def make_boto3_client(service_name: str, profile: str = "default", region: Optional[str] = None) -> Any:
    """Create a boto3 client for the specified AWS service.

    Args:
        service_name: AWS service identifier ('s3', 'secretsmanager', 'batch', 'ecs', 'ec2', 'logs', 'ecr').
        profile: AWS credentials profile name.
        region: AWS region override.

    Returns:
        boto3 client instance.
    """
    import boto3

    kwargs: dict[str, Any] = {}
    if profile and profile != "default":
        kwargs["profile_name"] = profile
    if region:
        kwargs["region_name"] = region

    session = boto3.Session(**kwargs)
    return session.client(service_name)


__all__ = ["make_boto3_client"]
