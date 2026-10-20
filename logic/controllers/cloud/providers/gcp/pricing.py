"""Static hourly price hints for Google Cloud machine types (CR-05).

On-demand and spot USD per instance-hour in ``us-central1`` for the machine types the
runner uses, scaled by a regional factor. Spot prices vary daily and these are list
prices: treat every value as an estimate (CR-10 overrides them from ``prices.json``).
Cloud Run Jobs bill per vCPU-second and GiB-second; ``cloudrun.<n>vcpu-<m>gib`` entries
give the hourly equivalent of the usual sizes (no spot).

Attributes:
    GCP_MACHINES: Machine type -> vCPUs and memory (GiB).
    price_per_hour: USD per hour, or None for an unknown machine type.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

# (on-demand, spot) USD/h in us-central1
_RATES: Dict[str, Tuple[float, float]] = {
    "e2-standard-2": (0.0670, 0.0201),
    "e2-standard-4": (0.1340, 0.0402),
    "e2-standard-8": (0.2681, 0.0804),
    "e2-highmem-4": (0.1808, 0.0542),
    "n2-standard-2": (0.0971, 0.0235),
    "n2-standard-4": (0.1942, 0.0470),
    "n2-standard-8": (0.3885, 0.0940),
    "c2-standard-4": (0.2088, 0.0505),
    "c2-standard-8": (0.4176, 0.1010),
    # Cloud Run Jobs: 0.000018 USD/vCPU-s + 0.000002 USD/GiB-s, no spot discount
    "cloudrun.1vcpu-2gib": (0.0792, 0.0792),
    "cloudrun.2vcpu-4gib": (0.1584, 0.1584),
    "cloudrun.4vcpu-8gib": (0.3168, 0.3168),
}

GCP_MACHINES: Dict[str, Dict[str, float]] = {
    "e2-standard-2": {"cpus": 2.0, "memory_gb": 8.0},
    "e2-standard-4": {"cpus": 4.0, "memory_gb": 16.0},
    "e2-standard-8": {"cpus": 8.0, "memory_gb": 32.0},
    "e2-highmem-4": {"cpus": 4.0, "memory_gb": 32.0},
    "n2-standard-2": {"cpus": 2.0, "memory_gb": 8.0},
    "n2-standard-4": {"cpus": 4.0, "memory_gb": 16.0},
    "n2-standard-8": {"cpus": 8.0, "memory_gb": 32.0},
    "c2-standard-4": {"cpus": 4.0, "memory_gb": 16.0},
    "c2-standard-8": {"cpus": 8.0, "memory_gb": 32.0},
}

_REGION_FACTORS: Dict[str, float] = {
    "us-central1": 1.0,
    "us-east1": 1.0,
    "us-west1": 1.0,
    "europe-west1": 1.10,
    "europe-west4": 1.10,
    "europe-southwest1": 1.15,
    "asia-east1": 1.15,
}

DEFAULT_MACHINE = "e2-standard-4"


def price_per_hour(
    machine_type: Optional[str] = None, region: Optional[str] = None, spot: bool = False
) -> Optional[float]:
    """USD per instance-hour of ``machine_type`` (default ``e2-standard-4``) in ``region``."""
    rates = _RATES.get((machine_type or DEFAULT_MACHINE).lower())
    if rates is None:
        return None
    return round((rates[1] if spot else rates[0]) * _REGION_FACTORS.get((region or "us-central1").lower(), 1.0), 5)


__all__ = ["GCP_MACHINES", "price_per_hour"]
