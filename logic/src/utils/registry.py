"""Generic name-and-tag registry for policies, models and related adapters.

One ``Registry`` type is the union of the old stage registries (name → class,
``create``) and the tag registry (object → tags, set queries). Stage instances
scope name lookup; the root ``ALL`` instance is the union of their tag indexes.
"""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Type, TypeVar, Union

T = TypeVar("T")


def import_target(target: str) -> Any:
    """Load ``module:attribute`` and return the attribute."""
    module_name, _, attr = target.partition(":")
    if not module_name or not attr:
        raise ValueError(f"target must be 'module:Attribute', got {target!r}")
    module = importlib.import_module(module_name)
    obj: Any = module
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


def lazy_getattr(module_globals: Dict[str, Any], mapping: Dict[str, str], name: str) -> Any:
    """Resolve a module-level ``__getattr__`` entry and cache it on the module.

    ``mapping`` values are either ``module:attribute`` or a module path.
    """
    target = mapping.get(name)
    if target is None:
        raise AttributeError(f"module has no attribute {name!r}")
    obj = import_target(target) if ":" in target else importlib.import_module(target)
    module_globals[name] = obj
    return obj


def _as_params(cfg: Any) -> Dict[str, Any]:
    if cfg is None:
        return {}
    if isinstance(cfg, Mapping):
        return dict(cfg)
    if hasattr(cfg, "__dataclass_fields__"):
        return asdict(cfg)
    if hasattr(cfg, "__dict__"):
        return {key: value for key, value in vars(cfg).items() if not key.startswith("_")}
    return {}


def instantiate(cls: Type[T], cfg: Any = None, **kwargs: Any) -> T:
    """Build ``cls`` from a config object and/or kwargs, ignoring unknown fields."""
    if cls.__init__ is object.__init__ or getattr(cls.__init__, "__module__", None) == "typing":
        return cls()
    params = _as_params(cfg)
    params.update(kwargs)
    signature = inspect.signature(cls.__init__)
    accepted = set(signature.parameters) - {"self"}
    if "config" in accepted and not kwargs:
        return cls(config=cfg)  # type: ignore[call-arg]
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
        return cls(**params)  # type: ignore[arg-type]
    filtered = {key: value for key, value in params.items() if key in accepted}
    return cls(**filtered)  # type: ignore[arg-type]


@dataclass
class _Spec:
    name: str
    target: Optional[str] = None
    obj: Any = None
    tags: Set[Any] = field(default_factory=set)
    config: Any = None

    def resolve(self) -> Any:
        if self.obj is None:
            if self.target is None:
                raise KeyError(f"registry entry {self.name!r} has no class or target")
            self.obj = import_target(self.target)
        return self.obj


class Registry:
    """Name lookup, lazy construction and tag queries.

    ``register`` accepts both the stage-registry form ``register("alns")`` and
    the tag-only form ``register(PolicyTag.EXACT, ...)``.
    """

    def __init__(self, name: str, parent: Optional["Registry"] = None) -> None:
        self.name = name
        self._parent = parent
        self._specs: Dict[str, _Spec] = {}
        self._children: List["Registry"] = []
        if parent is not None:
            parent._children.append(self)

    def declare(
        self,
        name: str,
        target: str,
        *,
        tags: Iterable[Any] = (),
        config: Any = None,
        aliases: Iterable[str] = (),
    ) -> None:
        """Register a lazy ``module:Class`` target under ``name`` and aliases."""
        tag_set = set(tags)
        for key in (name, *aliases):
            self._bind(str(key).lower(), target=target, tags=tag_set, config=config)

    def register(
        self,
        name: Any = None,
        *more: Any,
        tags: Iterable[Any] = (),
        config: Any = None,
        target: Optional[str] = None,
    ) -> Callable[[Any], Any]:
        """Decorator (and lazy-declare helper) for a registry entry."""
        extra_tags = list(tags)
        resolved_name: Optional[str] = None
        if isinstance(name, str):
            resolved_name = name
            extra_tags.extend(more)
        elif name is not None:
            extra_tags.append(name)
            extra_tags.extend(more)

        if target is not None and resolved_name is not None:
            self._bind(resolved_name.lower(), target=target, tags=set(extra_tags), config=config)
            return lambda obj: obj

        def decorator(obj: Any) -> Any:
            key = (resolved_name or getattr(obj, "__name__", str(obj))).lower()
            self._bind(key, obj=obj, tags=set(extra_tags), config=config)
            return obj

        return decorator

    def _bind(
        self,
        name: str,
        *,
        obj: Any = None,
        target: Optional[str] = None,
        tags: Optional[Set[Any]] = None,
        config: Any = None,
    ) -> _Spec:
        spec = self._specs.get(name)
        if spec is None:
            spec = _Spec(name=name)
            self._specs[name] = spec
        if obj is not None:
            spec.obj = obj
        if target is not None:
            spec.target = target
        if tags:
            spec.tags.update(tags)
        if config is not None:
            spec.config = config
        if spec.obj is not None and spec.tags and self._parent is not None:
            self._parent._note_object(spec.obj, spec.tags)
        return spec

    def _note_object(self, obj: Any, tags: Set[Any]) -> None:
        for spec in self._specs.values():
            if spec.obj is obj:
                spec.tags.update(tags)
                return
        self._specs[getattr(obj, "__name__", str(obj)).lower()] = _Spec(
            name=getattr(obj, "__name__", str(obj)).lower(),
            obj=obj,
            tags=set(tags),
        )

    def get(self, name: str) -> Optional[Any]:
        spec = self._specs.get(str(name).lower())
        if spec is None:
            return None
        obj = spec.resolve()
        if spec.tags and self._parent is not None:
            self._parent._note_object(obj, spec.tags)
        return obj

    def names(self) -> List[str]:
        return list(self._specs.keys())

    def create(self, name: str, cfg: Any = None, **kwargs: Any) -> Any:
        cls = self.get(name)
        if cls is None:
            raise ValueError(f"Unknown {self.name} entry: {name}")
        return instantiate(cls, cfg, **kwargs)

    @staticmethod
    def get_name(obj: Any) -> str:
        return getattr(obj, "__name__", str(obj))

    def _iter_specs(self) -> Iterable[_Spec]:
        yield from self._specs.values()
        for child in self._children:
            yield from child._specs.values()

    def _matches(
        self,
        spec: _Spec,
        predicate: Callable[[Set[Any]], bool],
        expected_type: Optional[Type[Any]] = None,
        require_tags: bool = True,
    ) -> bool:
        if require_tags and not spec.tags:
            return False
        if spec.tags and not predicate(spec.tags):
            return False
        if expected_type is None:
            return True
        obj = spec.resolve()
        return isinstance(obj, type) and issubclass(obj, expected_type)

    def query_intersection(
        self, *tags: Any, expected_type: Optional[Type[Any]] = None
    ) -> Union[List[Any], List[Type[Any]]]:
        query = set(tags)
        results: List[Any] = []
        for spec in self._iter_specs():
            if self._matches(spec, query.issubset, expected_type):
                results.append(spec.resolve())
        return list({id(obj): obj for obj in results}.values())

    def query_union(self, *tags: Any, expected_type: Optional[Type[Any]] = None) -> Union[List[Any], List[Type[Any]]]:
        query = set(tags)
        results: List[Any] = []
        for spec in self._iter_specs():
            if self._matches(spec, lambda obj_tags: not query.isdisjoint(obj_tags), expected_type):
                results.append(spec.resolve())
        return list({id(obj): obj for obj in results}.values())

    def query_difference(
        self,
        require: List[Any],
        exclude: List[Any],
        expected_type: Optional[Type[Any]] = None,
    ) -> Union[List[Any], List[Type[Any]]]:
        require_set = set(require)
        exclude_set = set(exclude)
        results: List[Any] = []
        for spec in self._iter_specs():
            if self._matches(
                spec,
                lambda obj_tags: require_set.issubset(obj_tags) and exclude_set.isdisjoint(obj_tags),
                expected_type,
            ):
                results.append(spec.resolve())
        return list({id(obj): obj for obj in results}.values())

    def get_all(self) -> Dict[Any, Set[Any]]:
        result: Dict[Any, Set[Any]] = {}
        for spec in self._iter_specs():
            if spec.obj is not None or spec.target is not None:
                obj = spec.resolve()
                result.setdefault(obj, set()).update(spec.tags)
        return result

    # Compatibility names used by the old stage registries and a few tests.
    def get_route_improver_class(self, name: str) -> Optional[Any]:
        return self.get(name)

    def list_strategies(self) -> List[str]:
        return self.names()

    def list_improvers(self) -> List[str]:
        return self.names()

    def list_routing_algorithms(self) -> List[str]:
        return self.names()

    def list_policies(self) -> List[str]:
        return sorted(self.names())

    @property
    def _registry(self) -> Dict[str, _Spec]:
        return self._specs

    @property
    def _strategies(self) -> Dict[str, _Spec]:
        return self._specs


ALL = Registry("all")
ROUTING = Registry("routing", parent=ALL)
PACKING = Registry("packing", parent=ALL)
IMPROVEMENT = Registry("improvement", parent=ALL)
ACCEPTANCE = Registry("acceptance", parent=ALL)
JOINT = Registry("joint", parent=ALL)
