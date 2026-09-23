"""Training task modes supported by the executor."""

from enum import Enum


class TaskTypeEnum(Enum):
    """Identify supported executor training task modes."""
    UNSUPERVISED = "UNSUPERVISED"
