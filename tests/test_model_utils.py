from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from utils.ModelUtils import ModelUtils


def test_ensure_bsh_accepts_explicit_layouts():
    bsh = torch.zeros(2, 3, 4)
    sbh = torch.zeros(3, 2, 4)

    assert ModelUtils.ensure_bsh(bsh, 4, from_layout="BSH").shape == (2, 3, 4)
    assert ModelUtils.ensure_bsh(sbh, 4, from_layout="SBH").shape == (2, 3, 4)


def test_ensure_bsh_rejects_ambiguous_or_wrong_shapes():
    with pytest.raises(ValueError, match="Expected 3D"):
        ModelUtils.ensure_bsh(torch.zeros(2, 4), 4)

    with pytest.raises(ValueError, match="hidden_size"):
        ModelUtils.ensure_bsh(torch.zeros(2, 3, 5), 4)

    with pytest.raises(ValueError, match="Ambiguous"):
        ModelUtils.ensure_bsh(torch.zeros(2, 2, 4), 4, batch_size=2)


def test_find_transformer_layers_uses_known_paths(tiny_model):
    layers = ModelUtils.find_transformer_layers(tiny_model)

    assert len(layers) == 2
    assert layers == list(tiny_model.layers)


def test_find_transformer_layers_can_use_heuristic_blocks():
    class HeuristicBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.self_attn = nn.Linear(4, 4)
            self.mlp = nn.Linear(4, 4)
            self.input_layernorm = nn.LayerNorm(4)

    class HeuristicModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(hidden_size=4)
            self.stack = nn.ModuleList([HeuristicBlock(), HeuristicBlock()])

    model = HeuristicModel()

    assert ModelUtils.find_transformer_layers(model) == list(model.stack)


@pytest.mark.parametrize(
    ("model_type", "path", "expected_attr"),
    [
        ("llama", "model.layers", "num_hidden_layers"),
        ("mistral", "model.layers", "num_hidden_layers"),
        ("qwen2", "model.layers", "num_hidden_layers"),
        ("qwen3", "model.layers", "num_hidden_layers"),
        ("gemma2", "model.layers", "num_hidden_layers"),
        ("gemma3", "language_model.model.layers", "num_hidden_layers"),
        ("phi3", "model.layers", "num_hidden_layers"),
        ("olmo2", "model.layers", "num_hidden_layers"),
        ("gpt2", "transformer.h", "n_layer"),
        ("gpt_neox", "gpt_neox.layers", "num_hidden_layers"),
        ("falcon", "transformer.h", "num_hidden_layers"),
        ("bloom", "transformer.h", "n_layer"),
        ("opt", "model.decoder.layers", "num_hidden_layers"),
        ("mpt", "transformer.blocks", "n_layers"),
        ("openelm", "transformer.layers", "num_transformer_layers"),
    ],
)
def test_find_transformer_layers_covers_common_hf_decoder_layouts(model_type, path, expected_attr):
    class ResidualBlock(nn.Module):
        pass

    class HfLikeModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            config_values = {
                "hidden_size": 4,
                "model_type": model_type,
                expected_attr: 2,
            }
            self.config = SimpleNamespace(**config_values)
            self._install_path(path, nn.ModuleList([ResidualBlock(), ResidualBlock()]))

        def _install_path(self, dotted_path: str, module_list: nn.ModuleList) -> None:
            current = self
            parts = dotted_path.split(".")
            for part in parts[:-1]:
                child = getattr(current, part, None)
                if child is None:
                    child = nn.Module()
                    setattr(current, part, child)
                current = child
            setattr(current, parts[-1], module_list)

    model = HfLikeModel()
    expected_layers = ModelUtils._get_by_path(model, path)

    assert ModelUtils.find_transformer_layers(model) == list(expected_layers)


def test_unwrap_model_and_hidden_size_helpers(tiny_model):
    wrapped = SimpleNamespace(module=tiny_model)

    assert ModelUtils.unwrap_model(wrapped) is tiny_model
    assert ModelUtils.get_hidden_size(tiny_model.config) == 4
    assert torch.equal(
        ModelUtils.sanitize_fp16(torch.tensor([float("nan"), float("inf"), -float("inf"), 2.0])),
        torch.tensor([0.0, 0.0, 0.0, 2.0]),
    )


def test_steering_dtype_helpers_support_expected_public_dtypes():
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        assert ModelUtils.validate_steering_dtype(dtype) == dtype
        assert ModelUtils.steering_work_dtype(dtype) == torch.float32

    assert ModelUtils.finite_scalar_for_dtype(1e40, torch.float32) == torch.finfo(torch.float32).max
    assert ModelUtils.finite_scalar_for_dtype(-1e40, torch.float16) == -torch.finfo(torch.float16).max
    assert ModelUtils.finite_scalar_for_dtype(float("nan"), torch.float32) == 0.0
    assert torch.allclose(ModelUtils.safe_normalize(torch.tensor([1e20, 0.0]), dim=0), torch.tensor([1.0, 0.0]))

    with pytest.raises(ValueError, match="torch.float16"):
        ModelUtils.validate_steering_dtype(torch.float64)

    with pytest.raises(TypeError, match="torch.dtype"):
        ModelUtils.validate_steering_dtype("fp16")
