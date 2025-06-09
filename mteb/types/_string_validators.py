from __future__ import annotations

from datetime import date
from typing import Annotated

from pydantic import AnyUrl, BeforeValidator, TypeAdapter

pastdate_adapter = TypeAdapter(date)
StrDate = Annotated[
    str, BeforeValidator(lambda value: str(pastdate_adapter.validate_python(value)))
]  # Allows the type to be a string, but ensures that the string is a valid date

http_url_adapter = TypeAdapter(AnyUrl)
StrURL = Annotated[
    str, BeforeValidator(lambda value: str(http_url_adapter.validate_python(value)))
]  # Allows the type to be a string, but ensures that the string is a URL
