


from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from logic.src.pipeline.simulations.actions import (
    CollectAction,
    FillAction,
    PackingAction,
    RouteImprovementAction,
)

pytestmark = [pytest.mark.unit, pytest.mark.fast]



class SimpleContext(dict):
    def __getattr__(self, name):
        return self.get(name)

@pytest.fixture
def mock_bins():
    bins = MagicMock()
    bins.collectlevl = 90.0
    bins.is_stochastic.return_value = True
    bins.load_filling.return_value = (1, np.ones(5)*10, np.ones(5)*50, 0.0)
    bins.collect.return_value = ([2], 50.0, 1, 100.0)
    return bins

@pytest.fixture
def base_context(mock_bins):
    return SimpleContext({
        'bins': mock_bins,
        'day': 1,
        'policy_name': 'regular',
        'policy': 'regular',
        'full_policy': 'regular7',
        'n_bins': 5,
        'threshold': None,
        'day_count': 1,
        'area': 'riomaior',
        'waste_type': 'plastic',
        'distance_matrix': np.zeros((6, 6))
    })

class TestPackingActionDetailed:
    def test_regular_mapping(self, base_context):
        action = PackingAction()
        base_context['full_policy'] = 'regular7'
        action.execute(base_context)
        assert 'mandatory' in base_context

class TestFillActionDetailed:
    def test_stochastic_fill(self, base_context, mock_bins):
        action = FillAction()
        action.execute(base_context)
        assert 'new_overflows' in base_context
        assert base_context['new_overflows'] == 1

class TestCollectActionDetailed:
    def test_collect_execution(self, base_context, mock_bins):
        base_context['tour'] = [0, 2, 0]
        base_context['cost'] = 10.0
        action = CollectAction()
        action.execute(base_context)
        assert base_context['total_collected'] == 50.0
        assert base_context['ncol'] == 1

class TestRouteImprovementActionDetailed:
    @patch("logic.src.policies.route_improvement.base.factory.RouteImproverFactory.create")
    def test_route_improvement_execution(self, mock_create, base_context):
        mock_proc = MagicMock()
        mock_proc.process.return_value = ([0, 1, 2, 0], {"improvement": 10.0})
        mock_create.return_value = mock_proc

        base_context['route_improvement'] = 'fast_tsp'
        base_context['tour'] = [0, 2, 1, 0]
        base_context['distance_matrix'] = np.zeros((6,6))

        action = RouteImprovementAction()
        action.execute(base_context)
        assert base_context['tour'] == [0, 1, 2, 0]
