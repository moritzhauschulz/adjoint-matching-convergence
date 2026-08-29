"""DriftMLP time-conditioning: sinusoidal embedding vs. raw scalar t."""
import pytest
import torch

from adjoint_sampling import DriftMLP


@pytest.mark.parametrize("time_embedding,expected_t_features", [
    ("sinusoidal", 8),
    ("raw", 1),
])
def test_time_embedding_input_width(time_embedding, expected_t_features):
    d, t_emb_dim = 2, 8
    net = DriftMLP(d=d, hidden_dim=16, n_layers=2, t_emb_dim=t_emb_dim,
                   time_embedding=time_embedding)
    assert net.net[0].in_features == d + expected_t_features

    x = torch.randn(5, d)
    t = torch.rand(5)
    assert net(x, t).shape == (5, d)


def test_time_embedding_rejects_unknown():
    with pytest.raises(ValueError):
        DriftMLP(d=1, time_embedding="learned")
