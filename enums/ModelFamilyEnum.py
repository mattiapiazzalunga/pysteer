"""Supported prompt-template model family identifiers."""

from enum import Enum


class ModelFamilyEnum(Enum):
    """Identify model families with dedicated prompt templates."""
    LLAMA3_1 = "LLAMA3_1"
    GEMMA3 = "GEMMA3"
    QWEN2_5 = "QWEN2_5"
    MISTRALV0_3 = "MISTRALV0_3"
    OLMO2 = "OLMO2"
