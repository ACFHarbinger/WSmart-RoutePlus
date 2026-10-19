"""Lazy AWS SDK (boto3) client factories.

boto3 is imported strictly inside functions to keep the core package independent
of provider dependencies.
"""

from __future__ import annotations

from typing import Any, Optional


def make_boto3_client(
    service_name: str,
    credentials: Optional[Any] = None,
    profile: str = "default",
    region: Optional[str] = None,
) -> Any:
    """Create a boto3 client for the specified AWS service.

    Args:
        service_name: AWS service identifier ('s3', 'secretsmanager', 'batch', 'ecs', 'ec2', 'logs', 'ecr').
        credentials: Optional ProviderCredentials instance or profile name.
        profile: AWS credentials profile name (fallback if credentials not given).
        region: AWS region override.

    Returns:
        boto3 client instance.
    """
    import boto3

    kwargs: dict[str, Any] = {}
    if credentials is not None:
        if hasattr(credentials, "aws_session_kwargs"):
            kwargs.update(credentials.aws_session_kwargs())
        elif isinstance(credentials, str) and credentials != "default":
            kwargs["profile_name"] = credentials
    elif profile and profile != "default":
        kwargs["profile_name"] = profile

    if region:
        kwargs["region_name"] = region

    session = boto3.Session(**kwargs)
    return session.client(service_name)


__all__ = ["make_boto3_client"]
