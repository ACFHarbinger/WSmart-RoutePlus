"""Neural models. Classes load on first access so the package import stays light."""

from typing import Any

from logic.src.utils.registry import lazy_getattr

_LAZY = {
    "AttentionModel": "logic.src.models.core.attention_model:AttentionModel",
    "TemporalAttentionModel": "logic.src.models.core.temporal_attention_model:TemporalAttentionModel",
    "CriticNetwork": "logic.src.models.common.critic_network.policy:CriticNetwork",
    "MandatoryManager": "logic.src.models.meta.hrl_manager:MandatoryManager",
    "WeightAdjustmentRNN": "logic.src.models.meta.weight_adjustment_rnn:WeightAdjustmentRNN",
    "HyperNetwork": "logic.src.models.meta.hypernet:HyperNetwork",
    "HyperNetworkOptimizer": "logic.src.models.meta.hypernet:HyperNetworkOptimizer",
    "MatNet": "logic.src.models.core.matnet:MatNet",
    "MDAM": "logic.src.models.core.mdam:MDAM",
    "PolyNet": "logic.src.models.core.polynet:PolyNet",
    "GLOP": "logic.src.models.core.glop:GLOP",
    "DeepACO": "logic.src.models.core.deepaco:DeepACO",
    "GFACS": "logic.src.models.core.gfacs:GFACS",
    "NARGNN": "logic.src.models.core.nargnn:NARGNN",
    "DACT": "logic.src.models.core.dact:DACT",
    "N2S": "logic.src.models.core.n2s:N2S",
    "NeuOpt": "logic.src.models.core.neuopt:NeuOpt",
    "MoEAttentionModel": "logic.src.models.core.moe:MoEAttentionModel",
    "MoETemporalAttentionModel": "logic.src.models.core.moe:MoETemporalAttentionModel",
    "PointerNetwork": "logic.src.models.core.pointer_network:PointerNetwork",
    "TransductiveModel": "logic.src.models.common.transductive.base:TransductiveModel",
    "ActiveSearch": "logic.src.models.common.transductive.active_search:ActiveSearch",
    "EAS": "logic.src.models.common.transductive.eas:EAS",
    "GraphAttentionEncoder": "logic.src.models.subnets:GraphAttentionEncoder",
    "GraphAttConvEncoder": "logic.src.models.subnets:GraphAttConvEncoder",
    "GatedGraphAttConvEncoder": "logic.src.models.subnets:GatedGraphAttConvEncoder",
    "TransGraphConvEncoder": "logic.src.models.subnets:TransGraphConvEncoder",
    "GatedRecurrentUnitFillPredictor": "logic.src.models.subnets:GatedRecurrentUnitFillPredictor",
    "common": "logic.src.models.common",
    "core": "logic.src.models.core",
    "meta": "logic.src.models.meta",
    "subnets": "logic.src.models.subnets",
}



__all__ = list(_LAZY)  # pyrefly: ignore[unresolvable-dunder-all]  # lazy re-exports; resolved through __getattr__


def __getattr__(name: str) -> Any:
    if name in ("common", "core", "meta", "subnets"):
        import importlib

        module = importlib.import_module(_LAZY[name])
        globals()[name] = module
        return module
    return lazy_getattr(globals(), _LAZY, name)
