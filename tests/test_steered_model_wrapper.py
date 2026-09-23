from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from activation_manager.SteeredModelWrapper import SteeredModelWrapper
from enums.ApplyFromModeEnum import ApplyFromModeEnum
from steering_strategy.GeneralSteeringStrategy import GeneralSteeringStrategy


def count_forward_hooks(model) -> int:
    return sum(
        len(getattr(module, "_forward_hooks", {})) + len(getattr(module, "_forward_pre_hooks", {}))
        for module in model.modules()
    )


@dataclass(frozen=True)
class OutputBox:
    last_hidden_state: torch.Tensor


def test_wrapper_extracts_and_replaces_hidden_in_common_output_shapes():
    tensor = torch.zeros(1, 2, 3)
    replacement = torch.ones_like(tensor)

    hidden, kind, meta = SteeredModelWrapper._extract_hidden_from_output((tensor, "rest"))
    assert hidden is tensor
    assert SteeredModelWrapper._replace_hidden_in_output((tensor, "rest"), replacement, kind, meta)[0].eq(1).all()

    hidden, kind, meta = SteeredModelWrapper._extract_hidden_from_output({"last_hidden_state": tensor})
    assert hidden is tensor
    assert SteeredModelWrapper._replace_hidden_in_output(
        {"last_hidden_state": tensor},
        replacement,
        kind,
        meta,
    )["last_hidden_state"].eq(1).all()

    box = OutputBox(last_hidden_state=tensor)
    hidden, kind, meta = SteeredModelWrapper._extract_hidden_from_output(box)
    replaced = SteeredModelWrapper._replace_hidden_in_output(box, replacement, kind, meta)
    assert replaced.last_hidden_state.eq(1).all()


def test_wrapper_attention_mask_helpers_normalize_shapes(tiny_model):
    wrapper = SteeredModelWrapper(tiny_model, strategy=None)

    assert SteeredModelWrapper._attention_mask_to_nonpad_2d(torch.tensor([1, 0])).shape == (1, 2)
    assert SteeredModelWrapper._attention_mask_to_nonpad_2d(torch.ones(2, 1, 3)).shape == (2, 3)
    assert SteeredModelWrapper._attention_mask_to_nonpad_2d(torch.ones(2, 1, 3, 3)).shape == (2, 3)

    labels = torch.tensor([[-100, -100, 4, 5]])
    attn = torch.tensor([[1, 1, 1, 1]])
    prompt_len = wrapper._infer_prompt_len_from_labels(
        labels=labels,
        attention_mask=attn,
        input_ids=None,
        pad_token_id=0,
    )
    assert prompt_len.tolist() == [2]


def test_wrapper_requires_context_manager_when_strategy_is_present(tiny_model):
    strategy = GeneralSteeringStrategy({0: torch.tensor([1.0, 0.0, 0.0, 0.0])})
    wrapper = SteeredModelWrapper(tiny_model, strategy=strategy, padding_side="right")
    input_ids = torch.tensor([[3, 4, 5]], dtype=torch.long)
    attention = torch.ones_like(input_ids)

    with pytest.raises(RuntimeError, match="context manager"):
        wrapper(input_ids=input_ids, attention_mask=attention)

    with wrapper as active:
        assert count_forward_hooks(tiny_model) == 1
        out = active(input_ids=input_ids, attention_mask=attention)
        assert out.logits.shape[:2] == input_ids.shape

    assert count_forward_hooks(tiny_model) == 0


def test_wrapper_validates_fixed_start_configuration(tiny_model):
    with pytest.raises(ValueError, match="apply_from_token"):
        SteeredModelWrapper(
            tiny_model,
            strategy=None,
            apply_from_mode=ApplyFromModeEnum.FIXED,
            apply_from_token=-1,
        )

    with pytest.raises(ValueError, match="finite"):
        SteeredModelWrapper(tiny_model, strategy=None, alpha=float("inf"))


def test_wrapper_merges_generation_kwargs_inside_context(tiny_model):
    strategy = GeneralSteeringStrategy({0: torch.tensor([1.0, 0.0, 0.0, 0.0])})
    wrapper = SteeredModelWrapper(
        tiny_model,
        strategy=strategy,
        generation_kwargs={"max_new_tokens": 2},
        padding_side="right",
    )
    input_ids = torch.tensor([[3, 4]], dtype=torch.long)

    with wrapper as active:
        generated = active.generate(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))

    assert generated.shape == (1, 4)
