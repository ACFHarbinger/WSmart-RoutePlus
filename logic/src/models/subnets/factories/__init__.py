"""Neural component factories. Implementations load on first access."""

from typing import Any

from logic.src.utils.registry import lazy_getattr

_LAZY = {
    "NeuralComponentFactory": "logic.src.models.subnets.factories.base:NeuralComponentFactory",
    "_create_decoder_by_type": "logic.src.models.subnets.factories.base:_create_decoder_by_type",
    "AttentionComponentFactory": "logic.src.models.subnets.factories.attention:AttentionComponentFactory",
    "GCNComponentFactory": "logic.src.models.subnets.factories.gcn:GCNComponentFactory",
    "GACComponentFactory": "logic.src.models.subnets.factories.gac:GACComponentFactory",
    "TGCComponentFactory": "logic.src.models.subnets.factories.tgc:TGCComponentFactory",
    "GGACComponentFactory": "logic.src.models.subnets.factories.ggac:GGACComponentFactory",
    "MLPComponentFactory": "logic.src.models.subnets.factories.mlp:MLPComponentFactory",
    "MoEComponentFactory": "logic.src.models.subnets.factories.moe:MoEComponentFactory",
    "NARComponentFactory": "logic.src.models.subnets.factories.nar:NARComponentFactory",
    "MDAMComponentFactory": "logic.src.models.subnets.factories.mdam:MDAMComponentFactory",
    "GFACSComponentFactory": "logic.src.models.subnets.factories.gfacs:GFACSComponentFactory",
}



__all__ = list(_LAZY)  # pyrefly: ignore[unresolvable-dunder-all]  # lazy re-exports; resolved through __getattr__


def __getattr__(name: str) -> Any:
    return lazy_getattr(globals(), _LAZY, name)
