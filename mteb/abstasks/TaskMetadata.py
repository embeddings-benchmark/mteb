from __future__ import annotations

from datetime import date
from typing import Annotated, Literal

from pydantic import (
    AnyUrl,
    BaseModel,
    BeforeValidator,
    TypeAdapter,
)

TASK_SUBTYPES = Literal[
    "Article retrieval",
    "Dialect pairing",
    "Dialog Systems",
    "Discourse coherence",
    "Language identification",
    "Linguistic acceptability",
    "Political",
    "Question answering",
    "Sentiment/Hate speech",
    "Thematic clustering",
]

TASK_DOMAINS = Literal[
    "Academic",
    "Blog",
    "Encyclopaedic",
    "Fiction",
    "Government",
    "Legal",
    "Medical",
    "News",
    "Non-fiction",
    "Poetry",
    "Religious",
    "Reviews",
    "Social",
    "Spoken",
    "Web",
]

TEXT_CREATION_METHOD = Literal[
    "found",
    "created",
    "machine-translated",
    "human-translated and localized",
    "machine-translated and verified",
    "machine-translated and localized",
]

SOCIOECONOMIC_STATUS = Literal[
    "high",
    "medium",
    "low",
]

ANNOTATOR_TYPE = Literal["expert-annotated", "human-annotated", "derived"]

http_url_adapter = TypeAdapter(AnyUrl)
STR_URL = Annotated[
    str, BeforeValidator(lambda value: str(http_url_adapter.validate_python(value)))
]  # Allows the type to be a string, but ensures that the string is a URL

pastdate_adapter = TypeAdapter(date)
STR_DATE = Annotated[
    str, BeforeValidator(lambda value: str(pastdate_adapter.validate_python(value)))
]  # Allows the type to be a string, but ensures that the string is a valid date


class TaskMetadata(BaseModel):
    # Meta: We can annotate the requirements here, and then link to it in the docs. This would move the documentation closer to the code, which I think is a good idea.

    hf_hub_name: str
    revision: str

    name: str
    description: str
    type: str  # No equivalent in mmteb
    category: str  # No equivalent in mmteb
    reference: STR_URL  # URL to documentation, e.g. published paper

    eval_splits: list[str]
    eval_langs: list[str]  # Might want to have a literal of langs when #251 is resolved
    main_score: str  # Might want a literal here

    date: tuple[STR_DATE, STR_DATE] | None  # When the data was collected
    form: list[Literal["spoken", "written"]] | None

    domains: list[TASK_DOMAINS] | None

    task_subtypes: list[TASK_SUBTYPES] | None
    license: str

    socioeconomic_status: SOCIOECONOMIC_STATUS | None
    annotations_creators: ANNOTATOR_TYPE | None
    dialect: list[str] | None

    text_creation: TEXT_CREATION_METHOD | None
    bibtex_citation: str | None
