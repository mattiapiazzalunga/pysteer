"""Application start policies for runtime steering."""

from enum import Enum


class ApplyFromModeEnum(Enum):
    """Select how the first steered token is computed."""
    PROMPT_END = "PROMPT_END"
    FIXED = "FIXED"
