from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, Union

from ._result import HFSubset

## LANGUAGE TYPES ##
ISOLanguageScript = str  # a 3-letter ISO 639-3 language code followed by a 4-letter ISO 15924 script code (e.g. "eng-Latn")
ISOLanguage = str  # a 3-letter ISO 639-3 language code
ISOScript = str  # a 4-letter ISO 15924 script code

Languages = Union[list[ISOLanguageScript], Mapping[HFSubset, list[ISOLanguageScript]]]

## LICENSE TYPES ##
Licenses = (  # this list can be extended as needed
    Literal[  # we use lowercase for the licenses similar to the huggingface datasets
        "not specified",  # or none found
        "mit",
        "cc-by-2.0",
        "cc-by-3.0",
        "cc-by-4.0",
        "cc-by-sa-3.0",
        "cc-by-sa-4.0",
        "cc-by-nc-3.0",
        "cc-by-nc-4.0",
        "cc-by-nc-sa-3.0",
        "cc-by-nc-sa-4.0",
        "cc-by-nc-nd-4.0",
        "cc-by-nd-4.0",
        "openrail",
        "openrail++",
        "odc-by",
        "afl-3.0",
        "apache-2.0",
        "cc-by-nd-2.1-jp",
        "cc0-1.0",
        "bsd-3-clause",
        "gpl-3.0",
        "lgpl",
        "lgpl-3.0",
        "cdla-sharing-1.0",
        "mpl-2.0",
        "msr-la-nc",
        "multiple",
    ]
)


## MODEL TYPES ##
ModelName = str
Revision = str


## MODALITY TYPES ##
Modalities = Literal[
    "text",
    "image",
]
