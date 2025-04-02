from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Union

from pydantic import (
    BaseModel,
    field_validator,
)
from typing_extensions import Literal, TypedDict

from ..custom_validators import LICENSES, MODALITIES, STR_DATE, STR_URL
from ..encoder_interface import PromptType
from ..languages import (
    ISO_LANGUAGE_SCRIPT,
    ISO_TO_LANGUAGE,
    ISO_TO_SCRIPT,
    path_to_lang_codes,
    path_to_lang_scripts,
)

TASK_SUBTYPE = Literal[
    "Article retrieval",
    "Conversational retrieval",
    "Dialect pairing",
    "Dialog Systems",
    "Discourse coherence",
    "Duplicate Image Retrieval",
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
    "Counterfactual Detection",
    "Emotion classification",
    "Reasoning as Retrieval",
    "Rendered Texts Understanding",
    "Image Text Retrieval",
    "Object recognition",
    "Scene recognition",
    "Caption Pairing",
    "Emotion recognition",
    "Textures recognition",
    "Activity recognition",
    "Tumor detection",
    "Duplicate Detection",
    "Rendered semantic textual similarity",
]

TASK_DOMAIN = Literal[
    "Academic",
    "Blog",
    "Constructed",
    "Encyclopaedic",
    "Engineering",
    "Fiction",
    "Government",
    "Legal",
    "Medical",
    "News",
    "Non-fiction",
    "Poetry",
    "Religious",
    "Reviews",
    "Scene",
    "Social",
    "Spoken",
    "Subtitles",
    "Web",
    "Written",
    "Programming",
    "Chemistry",
    "Financial",
    "Chemistry",
    "Financial",
]

SAMPLE_CREATION_METHOD = Literal[
    "found",
    "created",
    "human-translated and localized",
    "human-translated",
    "machine-translated",
    "machine-translated and verified",
    "machine-translated and localized",
    "LM-generated and verified",
    "rendered",
    "multiple",
]

MIEB_TASK_TYPE = (
    "Any2AnyMultiChoice",
    "Any2AnyRetrieval",
    "Any2AnyMultilingualRetrieval",
    "VisionCentricQA",
    "ImageClustering",
    "ImageClassification",
    "ImageMultilabelClassification",
    "DocumentUnderstanding",
    "VisualSTS(eng)",
    "VisualSTS(multi)",
    "ZeroShotClassification",
    "Compositionality",
)

TASK_TYPE = (
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
    "Speed",
) + MIEB_TASK_TYPE

TASK_TYPE = Literal[TASK_TYPE]


TASK_CATEGORY = Literal[
    "s2s",  # Sentence-to-sentence
    "s2p",  # Sentence-to-paragraph
    "p2p",  # Paragraph-to-paragraph
    "t2t",  # specifically for text-only tasks in mieb
    "i2i",  # image-to-image
    "i2t",  # image-to-text
    "t2i",  # text-to-image
    "it2t",  # image+text-to-text
    "it2i",  # image+text-to-image
    "i2it",  # image-to-image+text
    "t2it",  # text-to-image+text
    "it2it",  # image+text-to-image+text
]

ANNOTATOR_TYPE = Literal[
    "expert-annotated",
    "human-annotated",
    "derived",
    "LM-generated",
    "LM-generated and reviewed",  # reviewed by humans
]

SPLIT_NAME = str
HFSubset = str
LANGUAGES = Union[
    list[ISO_LANGUAGE_SCRIPT], Mapping[HFSubset, list[ISO_LANGUAGE_SCRIPT]]
]

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
    "sql",
]

METRIC_NAME = str
METRIC_VALUE = Union[int, float, dict[str, Any]]


class PromptDict(TypedDict, total=False):
    """A dictionary containing the prompt used for the task.

    Args:
        query: The prompt used for the queries in the task.
        passage: The prompt used for the passages in the task.
    """

    query: str
    passage: str


class DescriptiveStatistics(TypedDict):
    """Class for descriptive statistics."""

    pass


METRIC_VALUE = Union[int, float, dict[str, Any]]


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
        domains: The domains of the data. These includes "Non-fiction", "Social", "Fiction", "News", "Academic", "Blog", "Encyclopaedic",
            "Government", "Legal", "Medical", "Poetry", "Religious", "Reviews", "Web", "Spoken", "Written". A dataset can belong to multiple domains.
        task_subtypes: The subtypes of the task. E.g. includes "Sentiment/Hate speech", "Thematic Clustering". Feel free to update the list as needed.
        license: The license of the data specified as lowercase, e.g. "cc-by-nc-4.0". If the license is not specified, use "not specified". For custom licenses a URL is used.
        annotations_creators: The type of the annotators. Includes "expert-annotated" (annotated by experts), "human-annotated" (annotated e.g. by
            mturkers), "derived" (derived from structure in the data).
        dialect: The dialect of the data, if applicable. Ideally specified as a BCP-47 language tag. Empty list if no dialects are present.
        sample_creation: The method of text creation. Includes "found", "created", "machine-translated", "machine-translated and verified", and
            "machine-translated and localized".
        prompt: The prompt used for the task. Can be a string or a dictionary containing the query and passage prompts.
        bibtex_citation: The BibTeX citation for the dataset. Should be an empty string if no citation is available.
        adapted_from: Datasets adapted (translated, sampled from, etc.) from other datasets.
    """

    dataset: dict[str, Any]

    name: str
    description: str
    prompt: str | PromptDict | None = None
    type: TASK_TYPE
    modalities: list[MODALITIES] = ["text"]
    category: TASK_CATEGORY | None = None
    reference: STR_URL | None = None

    eval_splits: list[str] = ["test"]
    eval_langs: LANGUAGES
    main_score: str

    date: tuple[STR_DATE, STR_DATE] | None = None
    domains: list[TASK_DOMAIN] | None = None
    task_subtypes: list[TASK_SUBTYPE] | None = None
    license: LICENSES | STR_URL | None = None

    annotations_creators: ANNOTATOR_TYPE | None = None
    dialect: list[str] | None = None

    sample_creation: SAMPLE_CREATION_METHOD | None = None
    bibtex_citation: str | None = None
    adapted_from: list[str] | None = None

    def validate_metadata(self) -> None:
        self.dataset_path_is_specified(self.dataset)
        self.dataset_revision_is_specified(self.dataset)
        self.eval_langs_are_valid(self.eval_langs)

    @field_validator("dataset")
    def _check_dataset_path_is_specified(
        cls, dataset: dict[str, Any]
    ) -> dict[str, Any]:
        cls.dataset_path_is_specified(dataset)
        return dataset

    @field_validator("dataset")
    def _check_dataset_revision_is_specified(
        cls, dataset: dict[str, Any]
    ) -> dict[str, Any]:
        cls.dataset_revision_is_specified(dataset)
        return dataset

    @field_validator("prompt")
    def _check_prompt_is_valid(
        cls, prompt: str | PromptDict | None
    ) -> str | PromptDict | None:
        if isinstance(prompt, dict):
            for key in prompt:
                if key not in [e.value for e in PromptType]:
                    raise ValueError(
                        "The prompt dictionary should only contain the keys 'query' and 'passage'."
                    )
        return prompt

    @staticmethod
    def dataset_path_is_specified(dataset: dict[str, Any]) -> None:
        """This method checks that the dataset path is specified."""
        if "path" not in dataset or dataset["path"] is None:
            raise ValueError(
                "You must specify the path to the dataset in the dataset dictionary. "
                + "See https://huggingface.co/docs/datasets/main/en/package_reference/loading_methods#datasets.load_dataset"
            )

    @staticmethod
    def dataset_revision_is_specified(dataset: dict[str, Any]) -> None:
        if "revision" not in dataset:
            raise ValueError(
                "You must explicitly specify a revision for the dataset (either a SHA or None)."
            )
        if dataset["revision"] is None:
            logger.warning(
                "Revision missing for the dataset %s. It is encourage to specify a dataset revision for reproducability.",
                dataset["path"],
            )

    def eval_langs_are_valid(self, eval_langs: LANGUAGES) -> None:
        """This method checks that the eval_langs are specified as a list of languages."""
        if isinstance(eval_langs, dict):
            for langs in eval_langs.values():
                for code in langs:
                    self._check_language_code(code)
        else:
            for code in eval_langs:
                self._check_language_code(code)

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
    def bcp47_codes(self) -> list[ISO_LANGUAGE_SCRIPT]:
        """Return the languages and script codes of the dataset formatting in accordance with the BCP-47 standard."""
        if isinstance(self.eval_langs, dict):
            return sorted(
                {lang for langs in self.eval_langs.values() for lang in langs}
            )
        return sorted(set(self.eval_langs))

    @property
    def languages(self) -> list[str]:
        """Return the languages of the dataset as iso639-3 codes."""

        def get_lang(lang: str) -> str:
            return lang.split("-")[0]

        if isinstance(self.eval_langs, dict):
            return sorted(
                {get_lang(lang) for langs in self.eval_langs.values() for lang in langs}
            )
        return sorted({get_lang(lang) for lang in self.eval_langs})

    @property
    def scripts(self) -> set[str]:
        """Return the scripts of the dataset as iso15924 codes."""

        def get_script(lang: str) -> str:
            return lang.split("-")[1]

        if isinstance(self.eval_langs, dict):
            return {
                get_script(lang) for langs in self.eval_langs.values() for lang in langs
            }
        return {get_script(lang) for lang in self.eval_langs}

    def is_filled(self) -> bool:
        """Check if all the metadata fields are filled."""
        return all(
            getattr(self, field_name) is not None
            for field_name in self.model_fields
            if field_name not in ["prompt", "adapted_from"]
        )

    @property
    def hf_subsets_to_langscripts(self) -> dict[HFSubset, list[ISO_LANGUAGE_SCRIPT]]:
        """Return a dictionary mapping huggingface subsets to languages."""
        if isinstance(self.eval_langs, dict):
            return self.eval_langs
        return {"default": self.eval_langs}  # type: ignore

    @property
    def intext_citation(self, include_cite: bool = True) -> str:
        """Create an in-text citation for the dataset."""
        cite = ""
        if self.bibtex_citation:
            cite = f"{self.bibtex_citation.split(',')[0].split('{')[1]}"
        if include_cite and cite:
            # check for whitespace in the citation
            if " " in cite:
                logger.warning(
                    "Citation contains whitespace. Please ensure that the citation is correctly formatted."
                )
            return f"\\cite{{{cite}}}"
        return cite

    @property
    def descriptive_stats(self) -> dict[str, DescriptiveStatistics] | None:
        """Return the descriptive statistics for the dataset."""
        if self.descriptive_stat_path.exists():
            with self.descriptive_stat_path.open("r") as f:
                return json.load(f)
        return None

    @property
    def descriptive_stat_path(self) -> Path:
        """Return the path to the descriptive statistics file."""
        descriptive_stat_base_dir = Path(__file__).parent.parent / "descriptive_stats"
        if self.type in MIEB_TASK_TYPE:
            descriptive_stat_base_dir = descriptive_stat_base_dir / "Image"
        task_type_dir = descriptive_stat_base_dir / self.type
        if not descriptive_stat_base_dir.exists():
            descriptive_stat_base_dir.mkdir()
        if not task_type_dir.exists():
            task_type_dir.mkdir()
        return task_type_dir / f"{self.name}.json"

    @property
    def n_samples(self) -> dict[str, int] | None:
        """Returns the number of samples in the dataset"""
        stats = self.descriptive_stats
        if not stats:
            return None

        n_samples = {}
        for subset, subset_value in stats.items():
            if subset == "hf_subset_descriptive_stats":
                continue
            n_samples[subset] = subset_value["num_samples"]  # type: ignore
        return n_samples

    def __hash__(self) -> int:
        return hash(self.model_dump_json())

    @property
    def revision(self) -> str:
        return self.dataset["revision"]
