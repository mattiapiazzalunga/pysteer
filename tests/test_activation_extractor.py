from __future__ import annotations

import pytest
import torch

from activation_manager.ActivationExtractor import ActivationExtractor


def count_forward_hooks(model) -> int:
    return sum(
        len(getattr(module, "_forward_hooks", {})) + len(getattr(module, "_forward_pre_hooks", {}))
        for module in model.modules()
    )


def test_activation_extractor_collects_selected_layers_and_cleans_hooks(tiny_model):
    extractor = ActivationExtractor(tiny_model, [0, 1], offload_to_cpu=True)
    input_ids = torch.tensor([[3, 4, 5]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    with extractor:
        assert count_forward_hooks(tiny_model) == 2
        tiny_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        activations = extractor.finalize()
        assert set(activations) == {0, 1}
        assert activations[0].shape == (1, 3, 4)
        assert activations[0].device.type == "cpu"

    assert count_forward_hooks(tiny_model) == 0
    assert extractor.activations == {}


def test_activation_extractor_temporarily_disables_and_restores_hooks(tiny_model):
    extractor = ActivationExtractor(tiny_model, 0)
    extractor.attach()

    with extractor.temporarily_disabled():
        assert count_forward_hooks(tiny_model) == 0

    assert count_forward_hooks(tiny_model) == 1
    extractor.remove()
    assert count_forward_hooks(tiny_model) == 0


def test_activation_extractor_validates_model_and_layer_indexes(tiny_model):
    with pytest.raises(TypeError, match="torch.nn.Module"):
        ActivationExtractor(object(), [0])

    with pytest.raises(IndexError, match="out of range"):
        ActivationExtractor(tiny_model, [99])


def test_activation_extractor_finalize_rejects_incompatible_chunks(tiny_model):
    extractor = ActivationExtractor(tiny_model, 0)
    extractor._chunks[0] = [torch.zeros(1, 2, 4), torch.zeros(2, 1, 4)]

    with pytest.raises(RuntimeError, match="incompatible chunks"):
        extractor.finalize()
