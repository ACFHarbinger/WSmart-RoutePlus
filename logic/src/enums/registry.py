"""Tag-query facade over the shared ``ALL`` registry.

``GlobalRegistry`` remains the import name used by existing adapters. It
forwards to ``ALL``, which is the union of the stage registries.
"""

from typing import Any, Callable, Dict, List, Optional, Set, Type, TypeVar, Union

from logic.src.utils.registry import ALL

from .environment_tags import EnvironmentTag
from .model_tags import ModelTag
from .operator_tags import OperatorTag
from .policy_tags import PolicyTag
from .trainer_tags import TrainerTag

AnyTag = Union[PolicyTag, ModelTag, EnvironmentTag, OperatorTag, TrainerTag]
T_Algorithm = Union[Type, Callable[..., Any]]
T = TypeVar("T")


class GlobalRegistry:
    """Compatibility wrapper around ``ALL`` for tag-only registration."""

    @classmethod
    def register(cls, *tags: AnyTag) -> Callable[[T_Algorithm], T_Algorithm]:
        return ALL.register(*tags)

    @classmethod
    def get_name(cls, obj: T_Algorithm) -> str:
        return ALL.get_name(obj)

    @classmethod
    def query_intersection(
        cls, *tags: AnyTag, expected_type: Optional[Type[T]] = None
    ) -> Union[List[T_Algorithm], List[Type[T]]]:
        return ALL.query_intersection(*tags, expected_type=expected_type)

    @classmethod
    def query_union(
        cls, *tags: AnyTag, expected_type: Optional[Type[T]] = None
    ) -> Union[List[T_Algorithm], List[Type[T]]]:
        return ALL.query_union(*tags, expected_type=expected_type)

    @classmethod
    def query_difference(
        cls, require: List[AnyTag], exclude: List[AnyTag], expected_type: Optional[Type[T]] = None
    ) -> Union[List[T_Algorithm], List[Type[T]]]:
        return ALL.query_difference(require, exclude, expected_type=expected_type)

    @classmethod
    def get_all(cls) -> Dict[T_Algorithm, Set[AnyTag]]:
        return ALL.get_all()
