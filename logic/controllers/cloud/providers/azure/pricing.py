"""Static hourly price estimates for Azure compute instances and Container Apps.

Attributes:
    AZURE_PRICES: Static table mapping machine_type to (on_demand_usd_h, spot_usd_h).
    price_per_hour: Calculate estimated USD/h for an Azure instance or job execution.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

# Baseline rates in USD/h: (on_demand, spot)
# Default region: eastus
_DEFAULT_RATES: Dict[str, Tuple[float, float]] = {
    # General purpose (D-series v5)
    "standard_d2s_v5": (0.096, 0.0192),
    "standard_d4s_v5": (0.192, 0.0384),
    "standard_d8s_v5": (0.384, 0.0768),
    "standard_d16s_v5": (0.768, 0.1536),
    # Compute optimized (F-series v2)
    "standard_f2s_v2": (0.085, 0.017),
    "standard_f4s_v2": (0.169, 0.0338),
    "standard_f8s_v2": (0.338, 0.0676),
    # Memory optimized (E-series v5)
    "standard_e2s_v5": (0.126, 0.0252),
    "standard_e4s_v5": (0.252, 0.0504),
    # Burstable (B-series)
    "standard_b2s": (0.0416, 0.0083),
    "standard_b4ms": (0.166, 0.0332),
    # Container Apps Jobs (1 vCPU, 2 GiB baseline)
    "container_apps": (0.036, 0.0108),
    "container_apps.1vcpu": (0.036, 0.0108),
    "container_apps.2vcpu": (0.072, 0.0216),
}

# Regional multipliers relative to eastus
_REGION_FACTORS: Dict[str, float] = {
    "eastus": 1.0,
    "eastus2": 1.0,
    "westus2": 1.0,
    "centralus": 1.0,
    "westeurope": 1.09,
    "northeurope": 1.08,
    "uksouth": 1.11,
    "japaneast": 1.14,
}


def price_per_hour(
    machine_type: Optional[str] = None,
    region: Optional[str] = None,
    spot: bool = False,
) -> Optional[float]:
    """Calculate hourly price estimate in USD.

    Args:
        machine_type: Azure VM size (e.g. 'Standard_D2s_v5') or 'container_apps'.
        region: Azure region (defaults to eastus).
        spot: Whether spot capacity is requested.

    Returns:
        Estimated USD/h, or None if machine type is unrecognized.
    """
    key = (machine_type or "standard_d2s_v5").lower()
    rates = _DEFAULT_RATES.get(key)
    if rates is None:
        return None

    base_rate = rates[1] if spot else rates[0]
    reg_key = (region or "eastus").lower()
    factor = _REGION_FACTORS.get(reg_key, 1.0)
    return round(base_rate * factor, 5)


__all__ = ["price_per_hour"]
