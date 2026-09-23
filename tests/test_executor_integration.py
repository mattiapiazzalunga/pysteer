from __future__ import annotations

import pytest
import torch

pd = pytest.importorskip("pandas")

from activation_manager.SteeredModelWrapper import SteeredModelWrapper
from executor import Executor
from steering_engine.executor_services import count_forward_hooks


def test_executor_trains_cmd_with_tiny_model_and_returns_clean_runtime_wrapper(tiny_model, tiny_tokenizer):
    train_df = pd.DataFrame(
        [
            {"prompt": "question", "response": "good", "reference": 1},
            {"prompt": "question", "response": "bad", "reference": 0},
        ]
    )
    executor = Executor(
        model=tiny_model,
        tokenizer=tiny_tokenizer,
        train_df=train_df,
        method="cmd",
        layers_to_extract=[0],
        batch_size=2,
        dtype=torch.float32,
        alpha=0.25,
        free_training_artifacts_after_build=True,
    )

    wrapper = executor.representation_extractor()

    assert isinstance(wrapper, SteeredModelWrapper)
    assert executor.activation_extractor is None
    assert executor.vector_mediator is None
    assert count_forward_hooks(tiny_model) == 0

    input_ids = torch.tensor([[3, 4, 5]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    with wrapper as active:
        out = active(input_ids=input_ids, attention_mask=attention_mask)

    assert out.logits.shape[:2] == input_ids.shape
    assert count_forward_hooks(tiny_model) == 0


def test_executor_accepts_case_insensitive_registered_method_alias(tiny_model, tiny_tokenizer):
    train_df = pd.DataFrame(
        [
            {"prompt": "p", "response": "yes", "reference": 1},
            {"prompt": "p", "response": "no", "reference": 0},
        ]
    )

    executor = Executor(
        model=tiny_model,
        tokenizer=tiny_tokenizer,
        train_df=train_df,
        method="CMD",
        layers_to_extract=0,
        dtype=torch.float32,
    )

    assert executor.method == "cmd"
