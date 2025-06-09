from __future__ import annotations

from collections.abc import Mapping
from typing import Union

from ._result import HFSubset

ISOLanguageScript = str  # a 3-letter ISO 639-3 language code followed by a 4-letter ISO 15924 script code (e.g. "eng-Latn")
ISOLanguage = str  # a 3-letter ISO 639-3 language code
ISOScript = str  # a 4-letter ISO 15924 script code

Languages = Union[list[ISOLanguageScript], Mapping[HFSubset, list[ISOLanguageScript]]]
