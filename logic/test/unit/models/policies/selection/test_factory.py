"""Tests for selector factory functions."""

from dataclasses import dataclass
from typing import Optional

import pytest
from logic.src.policies.mandatory_selection.vectorized.factory import create_selector_from_config
from logic.src.policies.mandatory_selection.vectorized.last_minute import LastMinuteSelector
from logic.src.policies.mandatory_selection.vectorized.regular import RegularSelector

pytestmark = [pytest.mark.unit, pytest.mark.fast]





@dataclass
class MockConfig:
    strategy: str
    frequency: Optional[int] = None
    threshold: Optional[float] = None

def test_create_selector_from_dict():
    """Verify creation from a dictionary (ITraversable)."""
    cfg = {"strategy": "regular", "frequency": 5}
    selector = create_selector_from_config(cfg)
    assert isinstance(selector, RegularSelector)
    assert selector.frequency == 5

def test_create_selector_from_object():
    """Verify creation from an object with attributes."""
    cfg = MockConfig(strategy="last_minute", threshold=0.8)
    selector = create_selector_from_config(cfg)
    assert isinstance(selector, LastMinuteSelector)
    # Note: factory maps 'threshold' for last_minute strategy
    # LastMinuteSelector internal attribute might be different,
    # but here we check type and that it didn't crash.
    assert selector.threshold == 0.8

def test_create_selector_none():
    """Verify None handling."""
    assert create_selector_from_config(None) is None

def test_create_selector_unknown():
    """Verify unknown strategy handling (should raise in get_vectorized_selector)."""
    cfg = {"strategy": "unknown_strategy"}
    with pytest.raises(ValueError, match="Unknown strategy"):
        create_selector_from_config(cfg)

def test_create_selector_empty_params():
    """Verify strategy with default params."""
    cfg = {"strategy": "regular"}
    selector = create_selector_from_config(cfg)
    assert isinstance(selector, RegularSelector)
    assert selector.frequency == 3  # default value in factory
