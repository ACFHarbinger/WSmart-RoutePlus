


import numpy as np
import pytest
from logic.src.interfaces.context.packing_context import PackingContext
from logic.src.policies.packing.last_minute import LastMinuteSelection
from logic.src.policies.packing.revenue import RevenueThresholdSelection
from logic.src.policies.packing.service_level import ServiceLevelSelection

pytestmark = [pytest.mark.unit, pytest.mark.fast]





@pytest.fixture
def base_context():
    return PackingContext(
        bin_ids=np.arange(1, 6),
        current_fill=np.array([10.0, 95.0, 30.0, 85.0, 50.0]),
        threshold=90.0,
        vehicle_capacity=100.0,
        bin_volume=1.0,
        bin_density=100.0,
        revenue_kg=2.0
    )

class TestLastMinuteSelection:
    def test_pack(self, base_context):
        strategy = LastMinuteSelection()
        mandatory, _ = strategy.pack(base_context)
        # Bins > 90 are Bin 2 (index 1) which is 95.
        assert mandatory == [2]

    def test_pack_low_threshold(self, base_context):
        base_context.threshold = 40.0
        strategy = LastMinuteSelection()
        mandatory, _ = strategy.pack(base_context)
        # Bins > 40 are indices 1, 3, 4 -> IDs 2, 4, 5
        assert mandatory == [2, 4, 5]

class TestServiceLevelSelection:
    def test_pack_no_data(self, base_context):
        strategy = ServiceLevelSelection()
        mandatory, _ = strategy.pack(base_context)
        assert mandatory == []

    def test_pack_with_data(self, base_context):
        base_context.accumulation_rates = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        base_context.std_deviations = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        base_context.threshold = 2.0 # Conf limit k=2
        # predicted = fill + means + k*std
        # Bin 4: 85 + 5 + 2*1 = 92 (No)
        # Bin 2: 95 + 5 + 2*1 = 102 (Yes)
        # Let's make Bin 4 also exceed 100
        base_context.current_fill[3] = 94
        # Bin 4: 94 + 5 + 2*1 = 101 (Yes)

        strategy = ServiceLevelSelection()
        mandatory, _ = strategy.pack(base_context)
        assert 2 in mandatory
        assert 4 in mandatory
        assert len(mandatory) == 2

class TestRevenueThresholdSelection:
    def test_pack(self, base_context):
        # cap = vol * density = 1 * 100 = 100kg
        # rev = (fill/100) * cap * rate = fill * rate
        # Bins: [10*2, 95*2, 30*2, 85*2, 50*2] = [20, 190, 60, 170, 100]
        # Threshold 150
        base_context.threshold = 150.0
        strategy = RevenueThresholdSelection()
        mandatory, _ = strategy.pack(base_context)
        # Expected: Bin 2 (190) and Bin 4 (170)
        assert mandatory == [2, 4]
