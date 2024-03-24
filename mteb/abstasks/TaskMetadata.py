from __future__ import annotations

from datetime import date

from pydantic import (
    AnyUrl,
    BaseModel,
    BeforeValidator,
    TypeAdapter,
)
from typing_extensions import Annotated, Literal

TASK_SUBTYPE = Literal[
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
    "Scientific Reranking",
]

TASK_DOMAIN = Literal[
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
    "mixed",
]

TASK_TYPE = Literal[
    "BitextMining",
    "Classification",
    "Clustering",
    "PairClassification",
    "Reranking",
    "Retrieval",
    "STS",
    "Summarization",
]

TASK_CATEGORY = Literal[
    "s2s",  # Sentence-to-sentence
    "s2p",  # Sentence-to-paragraph
    "p2p",  # Paragraph-to-paragraph
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
    """
    Metadata for a task.

    Args:
        hf_hub_name: The name of the dataset for the task on the Hugging Face Hub.
        revision: The revision of the dataset for the task on the Hugging Face Hub.
        name: The name of the task.
        description: A description of the task.
        type: The type of the task. These includes "Classification", "Summarization", "STS", "Retrieval", "Reranking", "Clustering",
            "PairClassification", "BitextMining". The type should match the abstask type.
        category: The category of the task. E.g. includes "s2s", "s2p", "p2p" (s=sentence, p=paragraph).
        reference: A URL to the documentation of the task. E.g. a published paper.
        eval_splits: The splits of the dataset used for evaluation.
        eval_langs: The languages of the dataset used for evaluation.
        main_score: The main score used for evaluation.
        date: The date when the data was collected. Specified as a tuple of two dates.
        form: The form of the data. Either "spoken", "written".
        domains: The domains of the data. These includes "Non-fiction", "Social", "Fiction", "News", "Academic", "Blog", "Encyclopaedic",
            "Government", "Legal", "Medical", "Poetry", "Religious", "Reviews", "Web", "Spoken". A dataset can belong to multiple domains.
        task_subtypes: The subtypes of the task. E.g. includes "Sentiment/Hate speech", "Thematic Clustering". Feel free to update the list as needed.
        license: The license of the data.
        socioeconomic_status: The socioeconomic status of the data. Includes "high", "medium", "low", "mixed".
        annotations_creators: The type of the annotators. Includes "expert-annotated" (annotated by experts), "human-annotated" (annotated e.g. by
            mturkers), "derived" (derived from structure in the data).
        dialect: The dialect of the data, if applicable. Ideally specified as a BCP-47 language tag.
        text_creation: The method of text creation. Includes "found", "created", "machine-translated", "machine-translated and verified", and
            "machine-translated and localized".
        bibtex_citation: The BibTeX citation for the dataset.
    """

    hf_hub_name: str
    revision: str

    name: str
    description: str
    type: TASK_TYPE
    category: TASK_CATEGORY
    reference: STR_URL | None  # URL to documentation, e.g. published paper

    eval_splits: list[str]
    eval_langs: list[str]  # Might want to have a literal of langs when #251 is resolved
    main_score: str  # Might want a literal here

    date: tuple[STR_DATE, STR_DATE] | None  # When the data was collected
    form: list[Literal["spoken", "written"]] | None
    domains: list[TASK_DOMAIN] | None
    task_subtypes: list[TASK_SUBTYPE] | None
    license: str | None

    socioeconomic_status: SOCIOECONOMIC_STATUS | None
    annotations_creators: ANNOTATOR_TYPE | None
    dialect: list[str] | None

    text_creation: TEXT_CREATION_METHOD | None
    bibtex_citation: str | None
