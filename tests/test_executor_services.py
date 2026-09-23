from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

pd = pytest.importorskip("pandas")

from enums.TaskTypeEnum import TaskTypeEnum
from executor import Executor
from steering_engine.defaults import build_default_method_registry
from steering_engine.executor_services import ExecutorDatasetBuilder
from steering_engine.executor_services import ExecutorMethodContext
from steering_engine.executor_services import ExecutorTokenizerIO
from steering_engine.executor_services import PromptDataset
from steering_engine.executor_services import assert_no_forward_hooks
from steering_engine.executor_services import count_forward_hooks
from steering_engine.executor_services import identity_collate


def _owner(method: str = "cmd", layers=(0, 1)):
    owner = SimpleNamespace()
    owner.method = method
    owner.layers_to_extract = list(layers)
    owner.batch_size = 2
    owner.task_type = TaskTypeEnum.UNSUPERVISED
    owner.method_registry = build_default_method_registry()
    return owner


def _builder(method: str = "cmd", layers=(0, 1)):
    owner = _owner(method, layers=layers)
    methods = ExecutorMethodContext(owner)
    return ExecutorDatasetBuilder(owner, methods)


def test_prompt_dataset_and_identity_collate_are_predictable():
    rows = [{"prompt": "a"}, {"prompt": "b"}]
    dataset = PromptDataset(rows)

    assert len(dataset) == 2
    assert dataset[1] == {"prompt": "b"}
    assert identity_collate(rows) is rows


def test_dataset_builder_validates_cmd_training_rows():
    builder = _builder("cmd")
    ok = pd.DataFrame(
        [
            {"prompt": "p", "response": "yes", "reference": 1},
            {"prompt": "p", "response": "no", "reference": 0},
        ]
    )

    loader = builder.prepare(ok)
    assert len(loader.dataset) == 2

    with pytest.raises(ValueError, match="Missing required columns"):
        builder.prepare(pd.DataFrame([{"prompt": "p", "reference": 1}]))

    with pytest.raises(ValueError, match="reference"):
        builder.prepare(pd.DataFrame([{"prompt": "p", "response": "r", "reference": 0.5}]))

    with pytest.raises(ValueError, match="at least one row"):
        builder.prepare(pd.DataFrame(columns=["prompt", "response", "reference"]))

    with pytest.raises(ValueError, match="both positive"):
        builder.prepare(
            pd.DataFrame(
                [
                    {"prompt": "p", "response": "yes", "reference": 1},
                    {"prompt": "p", "response": "also yes", "reference": 1},
                ]
            )
        )


def test_dataset_builder_enforces_mbs_layer_balance():
    builder = _builder("mbs_cmd", layers=(0, 1))

    with pytest.raises(ValueError, match="both positive and negative"):
        builder.prepare(
            pd.DataFrame(
                [
                    {"prompt": "p", "response": "r", "reference": 1, "mbs_layer": 0},
                    {"prompt": "p", "response": "r", "reference": 0, "mbs_layer": 0},
                    {"prompt": "p", "response": "r", "reference": 1, "mbs_layer": 1},
                ]
            )
        )

    loader = builder.prepare(
        pd.DataFrame(
            [
                {"prompt": "p", "response": "r", "reference": 1, "mbs_layer": 0},
                {"prompt": "p", "response": "r", "reference": 0, "mbs_layer": 0},
                {"prompt": "p", "response": "r", "reference": 1, "mbs_layer": 1},
                {"prompt": "p", "response": "r", "reference": 0, "mbs_layer": 1},
            ]
        )
    )
    assert len(loader.dataset) == 4


def test_tokenizer_io_pad_token_and_device_helpers(tiny_model, tiny_tokenizer):
    tiny_tokenizer.pad_token = None
    tiny_tokenizer.pad_token_id = None
    tiny_model.config.pad_token_id = None
    tiny_model.generation_config.pad_token_id = None
    owner = SimpleNamespace(model=tiny_model, tokenizer=tiny_tokenizer)
    io = ExecutorTokenizerIO(owner)

    io.ensure_pad_token()

    assert tiny_tokenizer.pad_token == tiny_tokenizer.eos_token
    assert tiny_tokenizer.pad_token_id is not None
    assert tiny_model.config.pad_token_id == tiny_tokenizer.pad_token_id
    assert ExecutorTokenizerIO.is_missing_response_value(None) is True
    assert ExecutorTokenizerIO.is_missing_response_value(["already", "tokenized"]) is False
    assert io.trim_generated_tokens([4, 5, tiny_tokenizer.eos_token_id, 9]) == [4, 5]
    assert ExecutorTokenizerIO.as_torch_device("disk").type == "cpu"
    assert ExecutorTokenizerIO.infer_input_device(tiny_model).type == "cpu"


def test_executor_dtype_policy_supports_fp16_bf16_fp32_without_prequantizing_artifacts():
    executor = Executor.__new__(Executor)
    raw = torch.tensor([1e20, -1e20], dtype=torch.float32)

    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        Executor.dtype.fset(executor, dtype)

        assert executor.dtype == dtype
        cast = executor._cast_strategy_tensor(raw)
        assert cast.dtype == torch.float32
        assert torch.isfinite(cast).all()

    with pytest.raises(ValueError, match="torch.float16"):
        Executor.dtype.fset(executor, torch.float64)

    with pytest.raises(TypeError, match="torch.dtype"):
        Executor.dtype.fset(executor, "fp16")


def test_hook_count_helpers_detect_leaks(tiny_model):
    handle = tiny_model.layers[0].register_forward_hook(lambda *_: None)

    try:
        assert count_forward_hooks(tiny_model) == 1
        with pytest.raises(RuntimeError, match="HOOK LEAK"):
            assert_no_forward_hooks(tiny_model)
    finally:
        handle.remove()

    assert_no_forward_hooks(tiny_model)
