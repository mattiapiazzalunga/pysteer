from __future__ import annotations

import pytest

from enums.ModelTypeEnum import ModelTypeEnum
from prompt_generator.Gemma3PromptGenerator import Gemma3PromptGenerator
from prompt_generator.Llama3Point1PromptGenerator import Llama3Point1PromptGenerator
from prompt_generator.MistralV0Point3PromptGenerator import MistralV0Point3PromptGenerator
from prompt_generator.OLMo2PromptGenerator import OLMo2PromptGenerator
from prompt_generator.Qwen2Point5PromptGenerator import Qwen2Point5PromptGenerator
from utils.StringUtils import StringUtils


@pytest.mark.parametrize(
    ("generator_cls", "base_prefix", "instruct_marker"),
    [
        (Llama3Point1PromptGenerator, "<|begin_of_text|>", "<|start_header_id|>assistant"),
        (Gemma3PromptGenerator, "<bos>", "<start_of_turn>model"),
        (MistralV0Point3PromptGenerator, "<s>", "[/INST]"),
        (OLMo2PromptGenerator, "System", "<|assistant|>"),
        (Qwen2Point5PromptGenerator, "System", "<|im_start|>assistant"),
    ],
)
def test_prompt_generators_render_base_and_instruct_templates(generator_cls, base_prefix, instruct_marker):
    base = generator_cls(ModelTypeEnum.BASE)
    instruct = generator_cls(ModelTypeEnum.INSTRUCT)

    base_prompt = base.generate_prompt("  System\tmessage  ", "  User   message  ")
    instruct_prompt = instruct.generate_prompt(" System ", " User ", who_you_are="  testing   tools ")

    assert base_prompt.startswith(base_prefix)
    assert "System message" in base_prompt
    assert "User message" in base_prompt
    assert instruct_marker in instruct_prompt
    assert "testing tools" in instruct_prompt


def test_instruct_prompt_requires_role_description():
    gen = Llama3Point1PromptGenerator(ModelTypeEnum.INSTRUCT)

    with pytest.raises(ValueError, match="who_you_are"):
        gen.generate_prompt("system", "user")


def test_prompt_generator_rejects_unknown_model_type():
    with pytest.raises(ValueError, match="Model type"):
        Llama3Point1PromptGenerator("BASE")


def test_string_utils_normalize_text_without_destroying_words():
    assert StringUtils.trim_string("  hello  ") == "hello"
    assert StringUtils.remove_trailing_period("hello.") == "hello"
    assert StringUtils.remove_trailing_period("hello") == "hello"
    assert StringUtils.remove_spaces("  a\t\tb   c  ") == "a b c"
    assert StringUtils.to_lowercase("MiXeD") == "mixed"
