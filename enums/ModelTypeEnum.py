"""Supported model interaction styles for prompt templates."""

from enum import Enum


class ModelTypeEnum(Enum):
    """Identify base versus instruction-tuned model prompt formats."""
    BASE = "BASE"
    INSTRUCT = "INSTRUCT"
