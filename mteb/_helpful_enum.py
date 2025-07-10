from __future__ import annotations

from enum import Enum


class HelpfulStrEnum(str, Enum):
    """StrEnum that provides a method to create an instance from a string value, which allows for more user-friendly error messages."""

    @classmethod
    def from_str(cls, value: str) -> HelpfulStrEnum:
        try:
            return cls(value)
        except ValueError:
            raise ValueError(
                f"'{value}' is not a valid {cls.__name__} value, must be one of {[e.value for e in cls]}"
            )
