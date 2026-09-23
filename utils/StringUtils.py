"""String cleanup helpers used by prompt generation."""

import re


class StringUtils:
    """Namespace for prompt text cleanup helpers."""
    @staticmethod
    def trim_string(s):
        """Strip leading and trailing whitespace."""
        return s.strip()

    @staticmethod
    def remove_trailing_period(s):
        """Remove a final period when present."""
        return s[:-1] if s.endswith('.') else s

    @staticmethod
    def remove_spaces(s: str) -> str:
        """Collapse repeated spaces and tabs."""
        s = StringUtils.trim_string(s)
        s = re.sub(r"[ \t]+", " ", s)
        return s.strip()

    @staticmethod
    def to_lowercase(s):
        """Return the lowercase form of a string."""
        return s.lower()