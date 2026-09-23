"""Utility helpers for model inspection and string normalization."""

from utils.StringUtils import StringUtils

__all__ = [
    "ModelUtils",
    "StringUtils",
]


def __getattr__(name: str):
    """Lazily import utility classes to keep lightweight helpers lightweight."""
    if name == "ModelUtils":
        from utils.ModelUtils import ModelUtils

        return ModelUtils
    raise AttributeError(f"module 'utils' has no attribute {name!r}")
