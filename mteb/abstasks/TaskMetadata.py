from __future__ import annotations

import logging
from datetime import date
from typing import List, Mapping, Union

from pydantic import AnyUrl, BaseModel, BeforeValidator, TypeAdapter, field_validator
from typing_extensions import Annotated, Literal

from .languages import (
    ISO_TO_LANGUAGE,
    ISO_TO_SCRIPT,
    path_to_lang_codes,
    path_to_lang_scripts,
)

TASK_SUBTYPE = Literal[
    "Article retrieval",
    "Dialect pairing",
    "Dialog Systems",
    "Discourse coherence",
    "Language identification",
    "Linguistic acceptability",
    "Political classification",
    "Question answering",
    "Sentiment/Hate speech",
    "Thematic clustering",
    "Scientific Reranking",
    "Claim verification",
    "Topic classification",
    "Code retrieval",
    "False Friends",
    "Cross-Lingual Semantic Discrimination",
    "Textual Entailment",
]

TASK_DOMAIN = Literal[
    "Academic",
    "Blog",
    "Constructed",
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
    "Subtitles",
    "Web",
    "Programming",
]

TEXT_CREATION_METHOD = Literal[
    "found",
    "created",
    "machine-translated",
    "human-translated and localized",
    "machine-translated and verified",
    "machine-translated and localized",
    "LM-generated and verified",
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
    "MultilabelClassification",
    "Clustering",
    "PairClassification",
    "Reranking",
    "Retrieval",
    "STS",
    "Summarization",
    "InstructionRetrieval",
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

SPLIT_NAME = str
# a 3-letter ISO 639-3 language code followed by a 4-letter ISO 15924 script code (e.g. "eng-Latn")
ISO_LANGUAGE_SCRIPT = str
LANGUAGES = Union[List[ISO_LANGUAGE_SCRIPT], Mapping[str, List[ISO_LANGUAGE_SCRIPT]]]

PROGRAMMING_LANGS = [
    "python",
    "javascript",
    "typescript",
    "go",
    "ruby",
    "java",
    "php",
    "c",
    "c++",
    "rust",
    "swift",
    "scala",
    "shell",
]

logger = logging.getLogger(__name__)


class TaskMetadata(BaseModel):
    """Metadata for a task.

    Args:
        dataset: All arguments to pass to datasets.load_dataset to load the dataset for the task. Refer to https://huggingface.co/docs/datasets/v2.18.0/en/package_reference/loading_methods#datasets.load_dataset
        name: The name of the task.
        description: A description of the task.
        type: The type of the task. These includes "Classification", "Summarization", "STS", "Retrieval", "Reranking", "Clustering",
            "PairClassification", "BitextMining". The type should match the abstask type.
        category: The category of the task. E.g. includes "s2s", "s2p", "p2p" (s=sentence, p=paragraph).
        reference: A URL to the documentation of the task. E.g. a published paper.
        eval_splits: The splits of the dataset used for evaluation.
        eval_langs: The languages of the dataset used for evaluation. Langauges follows a ETF BCP 47 standard consisting of "{language}-{script}"
            tag (e.g. "eng-Latn"). Where language is specified as a list of ISO 639-3 language codes (e.g. "eng") followed by ISO 15924 script codes
            (e.g. "Latn"). Can be either a list of languages or a dictionary mapping huggingface subsets to lists of languages (e.g. if a the
            huggingface dataset contain different languages).
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
        dialect: The dialect of the data, if applicable. Ideally specified as a BCP-47 language tag. Empty list if no dialects are present.
        text_creation: The method of text creation. Includes "found", "created", "machine-translated", "machine-translated and verified", and
            "machine-translated and localized".
        bibtex_citation: The BibTeX citation for the dataset. Should be an empty string if no citation is available.
        n_samples: The number of samples in the dataset. This should only be for the splits evaluated on. For retrieval tasks, this should be the
            number of query-document pairs.
        avg_character_length: The average character length of the samples in the dataset. This should only be for the splits evaluated on. For
            retrieval tasks, this should be the average character length of the query-document pairs.
    """

    dataset: dict

    name: str
    description: str
    type: TASK_TYPE
    category: TASK_CATEGORY
    reference: STR_URL | None  # URL to documentation, e.g. published paper

    eval_splits: list[str]
    eval_langs: LANGUAGES
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

    n_samples: dict[SPLIT_NAME, int] | None
    avg_character_length: dict[SPLIT_NAME, float] | None

    @field_validator("dataset")
    def _check_dataset_path_is_specified(cls, dataset):
        """This method checks that the dataset path is specified."""
        if "path" not in dataset or dataset["path"] is None:
            raise ValueError(
                "You must specify the path to the dataset in the dataset dictionary. "
                "See https://huggingface.co/docs/datasets/main/en/package_reference/loading_methods#datasets.load_dataset"
            )
        return dataset

    @field_validator("dataset")
    def _check_dataset_revision_is_specified(cls, dataset):
        if "revision" not in dataset:
            raise ValueError(
                "You must explicitly specify a revision for the dataset (either a SHA or None)."
            )
        if dataset["revision"] is None:
            logger.warning(
                "Revision missing for the dataset %s. It is encourage to specify a dataset revision for reproducability.",
                dataset["path"],
            )
        return dataset

    @field_validator("eval_langs")
    def _check_eval_langs(cls, eval_langs):
        """This method checks that the eval_langs are specified as a list of languages."""
        if isinstance(eval_langs, dict):
            for langs in eval_langs.values():
                for code in langs:
                    cls._check_language_code(code)
        else:
            for code in eval_langs:
                cls._check_language_code(code)
        return eval_langs

    @staticmethod
    def _check_language_code(code):
        """This method checks that the language code (e.g. "eng-Latn") is valid."""
        lang, script = code.split("-")
        if script == "Code":
            if lang in PROGRAMMING_LANGS:
                return  # override for code
            else:
                raise ValueError(
                    f"Programming language {lang} is not a valid programming language."
                )
        if lang not in ISO_TO_LANGUAGE:
            raise ValueError(
                f"Invalid language code: {lang}, you can find valid ISO 639-3 codes in {path_to_lang_codes}"
            )
        if script not in ISO_TO_SCRIPT:
            raise ValueError(
                f"Invalid script code: {script}, you can find valid ISO 15924 codes in {path_to_lang_scripts}"
            )

    @property
    def languages(self) -> list[str]:
        """Return the languages of the dataset as iso639-3 codes."""

        def get_lang(lang: str) -> str:
            return lang.split("-")[0]

        if isinstance(self.eval_langs, dict):
            return sorted(
                set(
                    get_lang(lang)
                    for langs in self.eval_langs.values()
                    for lang in langs
                )
            )
        return sorted(set([get_lang(lang) for lang in self.eval_langs]))

    @property
    def scripts(self) -> set[str]:
        """Return the scripts of the dataset as iso15924 codes."""

        def get_script(lang: str) -> str:
            return lang.split("-")[1]

        if isinstance(self.eval_langs, dict):
            return set(
                get_script(lang) for langs in self.eval_langs.values() for lang in langs
            )
        return set(get_script(lang) for lang in self.eval_langs)

    def is_filled(self) -> bool:
        """Check if all the metadata fields are filled."""
        return all(
            getattr(self, field_name) is not None for field_name in self.model_fields
        )
