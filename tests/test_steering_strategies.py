from __future__ import annotations

import pytest
import torch

from steering_strategy.AngularSteeringStrategy import AngularSteeringStrategy
from steering_strategy.GeneralSteeringStrategy import GeneralSteeringStrategy
from steering_strategy.MbsSteeringStrategy import MbsSteeringStrategy


def test_general_strategy_adds_normalized_delta_only_to_masked_tokens():
    strategy = GeneralSteeringStrategy({0: torch.tensor([3.0, 4.0])})
    hidden = torch.zeros(2, 3, 2)
    mask = torch.tensor([[False, True, False], [True, False, True]])

    out = strategy.steer(0, hidden, alpha=2.0, mask_tok=mask)

    expected_delta = torch.tensor([1.2, 1.6])
    assert torch.allclose(out[0, 1], expected_delta)
    assert torch.allclose(out[1, 0], expected_delta)
    assert torch.allclose(out[1, 2], expected_delta)
    assert torch.equal(out[0, 0], torch.zeros(2))


def test_general_strategy_ignores_missing_layers_and_rejects_dict_alpha():
    strategy = GeneralSteeringStrategy({0: torch.tensor([1.0, 0.0])})
    hidden = torch.zeros(1, 1, 2)
    mask = torch.ones(1, 1, dtype=torch.bool)

    assert strategy.steer(99, hidden, alpha=1.0, mask_tok=mask) is hidden
    with pytest.raises(TypeError, match="does not support alpha mappings"):
        strategy.steer(0, hidden, alpha={0: 1.0}, mask_tok=mask)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_runtime_strategies_preserve_supported_activation_dtypes(dtype):
    mask = torch.ones(1, 1, dtype=torch.bool)

    general = GeneralSteeringStrategy({0: torch.tensor([1.0, 0.0])})
    out = general.steer(0, torch.zeros(1, 1, 2, dtype=dtype), alpha=1.0, mask_tok=mask)
    assert out.dtype == dtype

    angular = AngularSteeringStrategy(
        {0: (torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]))},
        adaptive=False,
        cpu_store_dtype=dtype,
    )
    out = angular.steer(0, torch.tensor([[[1.0, 0.0]]], dtype=dtype), alpha=90.0, mask_tok=mask)
    assert out.dtype == dtype


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_general_strategy_clamps_overflow_to_output_dtype(dtype):
    strategy = GeneralSteeringStrategy({0: torch.tensor([1.0, 0.0])})
    hidden = torch.zeros(1, 1, 2, dtype=dtype)
    mask = torch.ones(1, 1, dtype=torch.bool)

    out = strategy.steer(0, hidden, alpha=1e40, mask_tok=mask)

    out32 = out.to(torch.float32)
    assert out.dtype == dtype
    assert torch.isfinite(out32).all()
    assert out32.abs().max() <= torch.finfo(dtype).max


def test_general_strategy_normalizes_huge_vectors_without_float32_norm_overflow():
    strategy = GeneralSteeringStrategy({0: torch.tensor([1e20, 0.0])})
    hidden = torch.zeros(1, 1, 2)
    mask = torch.ones(1, 1, dtype=torch.bool)

    out = strategy.steer(0, hidden, alpha=1.0, mask_tok=mask)

    assert torch.allclose(out[0, 0], torch.tensor([1.0, 0.0]))


def test_mbs_strategy_uses_per_layer_alpha_values():
    strategy = MbsSteeringStrategy({0: torch.tensor([1.0, 0.0]), 1: torch.tensor([0.0, 1.0])})
    hidden = torch.zeros(1, 1, 2)
    mask = torch.ones(1, 1, dtype=torch.bool)

    out = strategy.steer(1, hidden, alpha={1: 3.0}, mask_tok=mask)

    assert torch.allclose(out[0, 0], torch.tensor([0.0, 3.0]))
    assert strategy.steer(0, hidden, alpha={1: 3.0}, mask_tok=mask) is hidden


def test_angular_strategy_rotates_plane_component_to_target_angle():
    strategy = AngularSteeringStrategy(
        {0: (torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]))},
        adaptive=False,
    )
    hidden = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])
    mask = torch.tensor([[True, False]])

    out = strategy.steer(0, hidden, alpha=90.0, mask_tok=mask)

    assert torch.allclose(out[0, 0], torch.tensor([0.0, 1.0]), atol=1e-6)
    assert torch.allclose(out[0, 1], torch.tensor([0.0, 0.0]), atol=1e-6)

    unchanged = strategy.steer(0, hidden, alpha=float("nan"), mask_tok=mask)
    assert torch.allclose(unchanged, hidden)


def test_strategy_constructors_validate_empty_or_inconsistent_inputs():
    with pytest.raises(ValueError, match="empty"):
        GeneralSteeringStrategy({})

    with pytest.raises(ValueError, match="1D tensor"):
        GeneralSteeringStrategy({0: torch.ones(1, 2)})

    with pytest.raises(ValueError, match="empty"):
        AngularSteeringStrategy({})
