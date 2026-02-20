from datetime import date
from typing import Annotated

from pydantic import AnyUrl, BeforeValidator, TypeAdapter

pastdate_adapter = TypeAdapter(date)
StrDate = Annotated[
    str, BeforeValidator(lambda value: str(pastdate_adapter.validate_python(value)))
]
"""A string that is a valid date in the past, e.g. formatted as YYYY-MM-DD."""


http_url_adapter = TypeAdapter(AnyUrl)
StrURL = Annotated[
    str, BeforeValidator(lambda value: str(http_url_adapter.validate_python(value)))
]
"""A string that is a valid URL."""
