from collections.abc import Mapping
from typing import Literal

from ._result import HFSubset

# LANGUAGE TYPES
ISOLanguageScript = str
"""A string representing the language and script. Language is denoted as a 3-letter [ISO 639-3](https://en.wikipedia.org/wiki/ISO_639-3) language code and the script is denoted by a 4-letter [ISO 15924](https://en.wikipedia.org/wiki/ISO_15924) script code (e.g. "eng-Latn")."""

ISOLanguage = str
"""A string representing the language. Language is denoted as a 3-letter [ISO 639-3](https://en.wikipedia.org/wiki/ISO_639-3) language code (e.g. "eng")."""

ISOScript = str
"""A string representing the script. The script is denoted by a 4-letter [ISO 15924](https://en.wikipedia.org/wiki/ISO_15924) script code (e.g. "Latn")."""

Languages = list[ISOLanguageScript] | Mapping[HFSubset, list[ISOLanguageScript]]
"""A list of languages or a mapping from HFSubset to a list of languages. E.g. ["eng-Latn", "deu-Latn"] or {"en-de": ["eng-Latn", "deu-Latn"], "fr-it": ["fra-Latn", "ita-Latn"]}."""

# LICENSE TYPES
Licenses = (
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
        "gemma",
    ]
)
"""The different licenses that a dataset or model can have. This list can be extended as needed."""

# MODEL TYPES
ModelName = str
"""The name of a model, typically as found on HuggingFace e.g. `sentence-transformers/all-MiniLM-L6-v2`."""
Revision = str
"""The revision of a model, typically a git commit hash. For APIs this can be a version string e.g. `1`."""


# MODALITY TYPES
Modalities = Literal[
    "text",
    "image",
]
"""The different modalities that a model can support."""
