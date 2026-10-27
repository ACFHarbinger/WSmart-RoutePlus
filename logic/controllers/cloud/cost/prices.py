"""Hourly price table per provider, region, machine type and spot (CR-10).

Lookup order for ``price_per_hour(provider, machine_type, region, spot)``:

1. the override file (``<cloud root>/prices.json``), written by :meth:`PriceTable.update`
   or by hand: ``{"<provider>": {"<region>|*": {"<machine_type>": [on_demand, spot]}}}``;
2. the provider's static table, ``logic.controllers.cloud.providers.<provider>.pricing.price_per_hour``;
3. None (unknown price: estimates and accrual report None, never zero).

Prices are list prices in USD per instance-hour. They are estimates: discounts,
sustained-use credits, billing minimums and storage or egress charges are not included.

Attributes:
    PriceFetcher: What :meth:`PriceTable.update` calls to refresh a provider's prices.
    PriceTable: The table.
"""

from __future__ import annotations

import importlib
import json
import time
from pathlib import Path
from typing import Callable, Dict, Optional, Protocol, Tuple, cast

Rates = Dict[str, Tuple[float, float]]  # machine type -> (on-demand, spot) USD/h


class PriceFetcher(Protocol):
    """A provider pricing source (its pricing API in production, a fake in tests)."""

    def __call__(self, provider: str, region: str) -> Rates: ...


class PriceTable:
    """Hourly prices with file overrides over the providers' static tables.

    Args:
        path: The override file (None: static tables only).
        clock: Time source for the ``updated_at`` stamp.
    """

    def __init__(self, path: Optional[Path] = None, clock: Callable[[], float] = time.time) -> None:
        self.path = Path(path) if path is not None else None
        self.clock = clock
        self.overrides: Dict[str, Dict[str, Dict[str, list]]] = {}
        self.updated_at: Dict[str, float] = {}
        if self.path is not None and self.path.is_file():
            data = json.loads(self.path.read_text())
            self.updated_at = data.pop("_updated_at", {})
            self.overrides = data

    def price_per_hour(
        self, provider: str, machine_type: Optional[str], region: Optional[str], spot: bool
    ) -> Optional[float]:
        """USD per instance-hour, or None when unknown."""
        by_region = self.overrides.get(provider, {})
        for key in (region or "", "*"):
            rates = by_region.get(key, {}).get(machine_type or "")
            if rates:
                return float(rates[1] if spot else rates[0])
        try:
            module = importlib.import_module(f"logic.controllers.cloud.providers.{provider}.pricing")
        except ImportError:
            return None
        lookup = getattr(module, "price_per_hour", None)
        if not callable(lookup):
            return None
        return cast(Optional[float], lookup(machine_type, region, spot))

    def update(self, provider: str, region: str, fetcher: PriceFetcher) -> int:
        """Replace a provider region's overrides with ``fetcher``'s prices; returns how many were stored."""
        rates = fetcher(provider, region)
        self.overrides.setdefault(provider, {})[region] = {m: [float(od), float(sp)] for m, (od, sp) in rates.items()}
        self.updated_at[f"{provider}/{region}"] = self.clock()
        self.save()
        return len(rates)

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({**self.overrides, "_updated_at": self.updated_at}, indent=2, sort_keys=True))
        tmp.replace(self.path)


def provider_fetcher(provider: str) -> Optional[PriceFetcher]:
    """The provider pricing module's ``fetch_prices(provider, region)``, if it has one."""
    try:
        module = importlib.import_module(f"logic.controllers.cloud.providers.{provider}.pricing")
    except ImportError:
        return None
    fetch = getattr(module, "fetch_prices", None)
    return cast(PriceFetcher, fetch) if callable(fetch) else None
