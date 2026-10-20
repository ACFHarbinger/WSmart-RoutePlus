

import pytest
import torch
from logic.src.models.meta.hrl_manager import MandatoryManager

pytestmark = [pytest.mark.unit, pytest.mark.fast]





def test_gatlst_manager_init():
    manager = MandatoryManager(
        input_dim_static=2,
        input_dim_dynamic=10,
        global_input_dim=2,
        hidden_dim=16,
        lstm_hidden=8,
        num_layers_gat=2,
        n_heads=2,
        device="cpu"
    )
    assert isinstance(manager, MandatoryManager)
    assert manager.embed_dim == 16  # hidden_dim is the legacy constructor name; the attribute is embed_dim

def test_gatlst_manager_forward():
    batch_size = 4
    num_nodes = 10
    history_len = 10

    manager = MandatoryManager(
        input_dim_static=2,
        input_dim_dynamic=history_len,
        global_input_dim=2,
        hidden_dim=16,
        lstm_hidden=8,
        num_layers_gat=2,
        n_heads=2,
        device="cpu"
    )

    static = torch.randn(batch_size, num_nodes, 2)
    dynamic = torch.randn(batch_size, num_nodes, history_len)
    global_features = torch.randn(batch_size, 2)

    mandatory_logits, gate_logits, value = manager(static, dynamic, global_features)

    assert mandatory_logits.shape == (batch_size, num_nodes, 2)
    assert gate_logits.shape == (batch_size, 2)
    assert value.shape == (batch_size, 1)

def test_gatlst_manager_select_action():
    batch_size = 4
    num_nodes = 10
    history_len = 10

    manager = MandatoryManager(
        input_dim_static=2,
        input_dim_dynamic=history_len,
        global_input_dim=2,
        hidden_dim=16,
        lstm_hidden=8,
        num_layers_gat=2,
        n_heads=2,
        device="cpu"
    )

    static = torch.randn(batch_size, num_nodes, 2)
    dynamic = torch.randn(batch_size, num_nodes, history_len)
    global_features = torch.randn(batch_size, 2)

    mandatory_action, gate_action, value = manager.select_action(
        static, dynamic, global_features, deterministic=False
    )

    assert mandatory_action.shape == (batch_size, num_nodes)
    assert gate_action.shape == (batch_size,)
    assert value.shape == (batch_size, 1)
