from __future__ import annotations

import pytest
import torch

from activation_manager.VectorMediator import VectorMediator
from vector_update_strategy.AngularVectorMediator import AngularVectorMediator
from vector_update_strategy.CmdVectorMediator import CmdVectorMediator
from vector_update_strategy.ColdKernelGradientMediator import ColdKernelGradientMediator
from vector_update_strategy.CpcaVectorMediator import CpcaVectorMediator
from vector_update_strategy.MbsCmdVectorMediator import MbsCmdVectorMediator


def test_vector_mediator_initializes_strategy_and_dispatches_updates(activation_batch):
    acts, labels, starts, attention = activation_batch
    strategy = CmdVectorMediator()
    mediator = VectorMediator(layers_to_extract=0, hidden_size=2, update_strategy=strategy)

    mediator.update_vectors(
        activations={0: acts},
        labels=labels,
        starts=starts,
        tokens_window=-1,
        attention_mask=attention,
        group_ids=None,
    )
    strategy.finalize()

    assert mediator.layers_to_extract == [0]
    assert torch.allclose(strategy.correction_vectors[0], torch.tensor([2.0, -2.0]))


def test_vector_mediator_reports_bad_configuration_and_update_errors(activation_batch):
    with pytest.raises(ValueError, match="hidden_size"):
        VectorMediator(layers_to_extract=[0], hidden_size=0)

    mediator = VectorMediator(layers_to_extract=[0], hidden_size=2)
    with pytest.raises(RuntimeError, match="update_strategy"):
        mediator.update_vectors({}, torch.tensor([]), torch.tensor([]), -1, None, None)

    class FailingStrategy(CmdVectorMediator):
        def update(self, *args, **kwargs):
            raise ValueError("boom")

    acts, labels, starts, attention = activation_batch
    mediator = VectorMediator(layers_to_extract=[0], hidden_size=2, update_strategy=FailingStrategy())
    with pytest.raises(RuntimeError, match="boom"):
        mediator.update_vectors({0: acts}, labels, starts, -1, attention, None)
    with pytest.raises(RuntimeError, match="Vector update failed"):
        mediator.assert_ok()


def test_cmd_vector_mediator_uses_last_or_mean_response_representation(activation_batch):
    acts, labels, starts, attention = activation_batch
    mediator = CmdVectorMediator(use_last_token_for_response=True)
    mediator.init_layers([0], 2)
    mediator.update(0, acts, labels, starts, -1, attention, None)
    mediator.finalize()

    assert torch.allclose(mediator.correction_vectors[0], torch.tensor([2.0, -2.0]))

    mean_mediator = CmdVectorMediator(use_last_token_for_response=False)
    mean_mediator.init_layers([0], 2)
    mean_mediator.update(0, acts, labels, torch.tensor([0, 0]), -1, attention, None)
    mean_mediator.finalize()
    assert torch.allclose(mean_mediator.correction_vectors[0], torch.tensor([1.5, -1.5]))


def test_cpca_vector_mediator_returns_direction_aligned_with_mean_difference(activation_batch):
    acts, labels, starts, attention = activation_batch
    mediator = CpcaVectorMediator()
    mediator.init_layers([0], 2)
    mediator.update(0, acts, labels, starts, -1, attention, None)
    mediator.finalize()

    vec = mediator.correction_vectors[0]
    assert vec.shape == (2,)
    assert torch.dot(vec, torch.tensor([2.0, -2.0])) > 0


def test_mbs_vector_mediator_routes_rows_to_their_declared_layers():
    acts = torch.tensor(
        [
            [[0.0, 0.0], [2.0, 0.0]],
            [[0.0, 0.0], [0.0, 2.0]],
            [[0.0, 0.0], [1.0, 1.0]],
            [[0.0, 0.0], [1.0, -1.0]],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
    starts = torch.ones(4, dtype=torch.long)
    attention = torch.ones(4, 2, dtype=torch.bool)
    group_ids = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    mediator = MbsCmdVectorMediator()
    mediator.init_layers([0, 1], 2)

    mediator.update(0, acts, labels, starts, -1, attention, group_ids)
    mediator.update(1, acts, labels, starts, -1, attention, group_ids)
    mediator.finalize()

    assert torch.allclose(mediator.correction_vectors[0], torch.tensor([2.0, -2.0]))
    assert torch.allclose(mediator.correction_vectors[1], torch.tensor([0.0, 2.0]))


def test_cold_kernel_gradient_mediator_accumulates_negative_mean_gradients():
    mediator = ColdKernelGradientMediator(normalize_to_unit_norm=False)
    mediator.init_layers([0], 2)
    grads = {0: torch.tensor([[[1.0, 0.0], [3.0, 0.0]], [[0.0, 2.0], [0.0, 4.0]]])}
    mask = torch.tensor([[False, True], [True, True]])

    mediator.update_from_gradients(grads, mask)
    mediator.finalize()

    assert torch.allclose(mediator.correction_vectors[0], torch.tensor([-1.5, -1.5]))

    with pytest.raises(RuntimeError, match="update_from_gradients"):
        mediator.update(0, grads[0], torch.ones(2), torch.zeros(2, dtype=torch.long), -1, mask, None)


def test_angular_vector_mediator_builds_orthonormal_planes():
    mediator = AngularVectorMediator()
    mediator.init_layers([0, 1], 2)
    acts = torch.tensor(
        [
            [[0.0, 0.0], [2.0, 0.0]],
            [[0.0, 0.0], [0.0, 2.0]],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([1.0, 0.0])
    starts = torch.ones(2, dtype=torch.long)
    attention = torch.ones(2, 2, dtype=torch.bool)

    acts_layer_1 = torch.tensor(
        [
            [[0.0, 0.0], [2.0, 2.0]],
            [[0.0, 0.0], [0.0, 0.0]],
        ],
        dtype=torch.float32,
    )

    mediator.update(0, acts, labels, starts, -1, attention, None)
    mediator.update(1, acts_layer_1, labels, starts, -1, attention, None)
    mediator.finalize()

    planes = mediator.get_angular_params()
    assert set(planes) == {0, 1}
    for first, second in planes.values():
        assert torch.allclose(first.norm(), torch.tensor(1.0), atol=1e-6)
        assert torch.allclose(second.norm(), torch.tensor(1.0), atol=1e-6)
        assert abs(float(torch.dot(first, second))) < 1e-5
