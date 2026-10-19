"""Static hourly price estimates for AWS compute instances and Fargate tasks.

Attributes:
    AWS_PRICES: Static table mapping (region, machine_type) to (on_demand_usd_h, spot_usd_h).
    price_per_hour: Calculate estimated USD/h for an instance or task configuration.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

# Baseline rates in USD/h: (on_demand, spot)
# Default region: us-east-1
_DEFAULT_RATES: Dict[str, Tuple[float, float]] = {
    # Compute optimized
    "c6i.large": (0.085, 0.034),
    "c6i.xlarge": (0.170, 0.068),
    "c6i.2xlarge": (0.340, 0.136),
    "c6i.4xlarge": (0.680, 0.272),
    "c6i.8xlarge": (1.360, 0.544),
    "c7i.large": (0.08925, 0.0357),
    "c7i.xlarge": (0.1785, 0.0714),
    "c7i.2xlarge": (0.357, 0.1428),
    # General purpose
    "m6i.large": (0.096, 0.038),
    "m6i.xlarge": (0.192, 0.076),
    "m6i.2xlarge": (0.384, 0.152),
    "m6i.4xlarge": (0.768, 0.307),
    "t3.medium": (0.0416, 0.0125),
    "t3.large": (0.0832, 0.025),
    "t3.xlarge": (0.1664, 0.050),
    # Serverless Fargate (1 vCPU, 2 GiB approx)
    "fargate": (0.04048, 0.01214),
    "fargate.1vcpu": (0.04048, 0.01214),
    "fargate.2vcpu": (0.08096, 0.02428),
    "fargate.4vcpu": (0.16192, 0.04856),
}

# Regional multipliers relative to us-east-1
_REGION_FACTORS: Dict[str, float] = {
    "us-east-1": 1.0,
    "us-east-2": 1.0,
    "us-west-2": 1.0,
    "eu-west-1": 1.08,
    "eu-central-1": 1.12,
    "ap-northeast-1": 1.15,
}


def price_per_hour(
    machine_type: Optional[str] = None,
    region: Optional[str] = None,
    spot: bool = False,
) -> Optional[float]:
    """Calculate hourly price estimate in USD.

    Args:
        machine_type: EC2 instance type (e.g. 'c6i.large') or 'fargate'.
        region: AWS region (defaults to us-east-1).
        spot: Whether spot capacity is requested.

    Returns:
        Estimated USD/h, or None if instance type is unrecognized.
    """
    key = (machine_type or "c6i.large").lower()
    rates = _DEFAULT_RATES.get(key)
    if rates is None:
        return None

    base_rate = rates[1] if spot else rates[0]
    reg_key = (region or "us-east-1").lower()
    factor = _REGION_FACTORS.get(reg_key, 1.0)
    return round(base_rate * factor, 5)


__all__ = ["price_per_hour"]
