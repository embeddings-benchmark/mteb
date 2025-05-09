from __future__ import annotations

from datetime import date
from typing import Annotated

from pydantic import AnyUrl, BeforeValidator, TypeAdapter
from typing_extensions import Literal

MODALITIES = Literal[
    "text",
    "image",
]

http_url_adapter = TypeAdapter(AnyUrl)
STR_URL = Annotated[
    str, BeforeValidator(lambda value: str(http_url_adapter.validate_python(value)))
]  # Allows the type to be a string, but ensures that the string is a URL

LICENSES = (  # this list can be extended as needed
    Literal[  # we use lowercase for the licenses similar to the huggingface datasets
        "not specified",  # or none found
        "mit",
        "cc-by-2.0",
        "cc-by-3.0",
        "cc-by-4.0",
        "cc-by-sa-3.0",
        "cc-by-sa-4.0",
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
        "openrail",
    ]
)

pastdate_adapter = TypeAdapter(date)
STR_DATE = Annotated[
    str, BeforeValidator(lambda value: str(pastdate_adapter.validate_python(value)))
]  # Allows the type to be a string, but ensures that the string is a valid date
