from __future__ import annotations

from collections.abc import Mapping
from typing import Union

from ._result import HFSubset

ISO_LANGUAGE_SCRIPT = str  # a 3-letter ISO 639-3 language code followed by a 4-letter ISO 15924 script code (e.g. "eng-Latn")
ISO_LANGUAGE = str  # a 3-letter ISO 639-3 language code
ISO_SCRIPT = str  # a 4-letter ISO 15924 script code

LANGUAGES = Union[
    list[ISO_LANGUAGE_SCRIPT], Mapping[HFSubset, list[ISO_LANGUAGE_SCRIPT]]
]
