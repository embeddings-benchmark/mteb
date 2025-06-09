from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from huggingface_hub import (
    DatasetCard,
    DatasetCardData,
    constants,
    file_exists,
    repo_exists,
)
from pydantic import (
    BaseModel,
    ConfigDict,
    field_validator,
)
from typing_extensions import Literal, TypedDict

import mteb
from mteb.languages import check_language_code
from mteb.types import (
    ISO_LANGUAGE_SCRIPT,
    LANGUAGES,
    LICENSES,
    MODALITIES,
    STR_DATE,
    STR_URL,
    PromptType,
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
    "Intent classification",
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
    "Entertainment",
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

_TASK_TYPE = (
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
    "InstructionReranking",
) + MIEB_TASK_TYPE

TASK_TYPE = Literal[_TASK_TYPE]


TASK_CATEGORY = Literal[
    "t2t",
    "t2c",  # text-to-category
    "i2i",  # image-to-image
    "i2c",  # image-to-category
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

HFSubset = str


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


class TextStatistics(TypedDict):
    """Class for descriptive statistics for texts.

    Attributes:
        min_text_length: Minimum length of text
        average_text_length: Average length of text
        max_text_length: Maximum length of text
        unique_texts: Number of unique texts
    """

    min_text_length: int
    average_text_length: float
    max_text_length: int
    unique_texts: int


class ImageStatistics(TypedDict):
    """Class for descriptive statistics for images.

    Attributes:
        min_image_width: Minimum width of images
        average_image_width: Average width of images
        max_image_width: Maximum width of images

        min_image_height: Minimum height of images
        average_image_height: Average height of images
        max_image_height: Maximum height of images
    """

    min_image_width: float
    average_image_width: float
    max_image_width: float

    min_image_height: float
    average_image_height: float
    max_image_height: float


class LabelStatistics(TypedDict):
    """Class for descriptive statistics for texts.

    Attributes:
        min_labels_per_text: Minimum number of labels per text
        average_label_per_text: Average number of labels per text
        max_labels_per_text: Maximum number of labels per text

        unique_labels: Number of unique labels
        labels: dict of label frequencies
    """

    min_labels_per_text: int
    average_label_per_text: float
    max_labels_per_text: int

    unique_labels: int
    labels: dict[str, dict[str, int]]


class ScoreStatistics(TypedDict):
    """Class for descriptive statistics for texts.

    Attributes:
        min_labels_per_text: Minimum number of labels per text
        average_label_per_text: Average number of labels per text
        max_labels_per_text: Maximum number of labels per text

        unique_labels: Number of unique labels
        labels: dict of label frequencies
    """

    min_score: int
    avg_score: float
    max_score: int


logger = logging.getLogger(__name__)


class MetadataDatasetDict(TypedDict, total=False):
    """A dictionary containing the dataset path and revision.

    Args:
        path: The path to the dataset.
        revision: The revision of the dataset.
        name: The name the dataset config.
        split: The split of the dataset.
        trust_remote_code: Whether to trust the remote code.
    """

    path: str
    revision: str
    name: str
    split: str
    trust_remote_code: bool


class TaskMetadata(BaseModel):
    """Metadata for a task.

    Args:
        dataset: All arguments to pass to [datasets.load_dataset](https://huggingface.co/docs/datasets/v2.18.0/en/package_reference/loading_methods#datasets.load_dataset) to load the dataset for the task.
        name: The name of the task.
        description: A description of the task.
        type: The type of the task. These includes "Classification", "Summarization", "STS", "Retrieval", "Reranking", "Clustering",
            "PairClassification", "BitextMining". The type should match the abstask type.
        category: The category of the task. E.g. includes "t2t" (text to text), "t2i" (text to image).
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

    model_config = ConfigDict(extra="forbid")

    dataset: MetadataDatasetDict

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
                    check_language_code(code)
        else:
            for code in eval_langs:
                check_language_code(code)

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

    @property
    def hf_subsets(self) -> list[str]:
        """Return the huggingface subsets."""
        return list(self.hf_subsets_to_langscripts.keys())

    @property
    def is_multilingual(self) -> bool:
        """Check if the task is multilingual."""
        return isinstance(self.eval_langs, dict)

    def __hash__(self) -> int:
        return hash(self.model_dump_json())

    @property
    def revision(self) -> str:
        return self.dataset["revision"]

    def create_dataset_card_data(
        self, existing_dataset_card_data: DatasetCardData | None = None
    ) -> tuple[DatasetCardData, dict[str, str]]:
        """Create a DatasetCardData object from the task metadata.

        Args:
            existing_dataset_card_data: The existing DatasetCardData object to update. If None, a new object will be created.

        Returns:
            A DatasetCardData object with the metadata for the task with kwargs to card
        """
        # todo figure out datasets with multiple types. E. g. one dataset as classification and ZeroShotClassification
        # to get full list of task_types execute:
        # requests.post("https://huggingface.co/api/validate-yaml", json={
        #     "content": "---\ntask_categories: ['test']\n---", "repoType": "dataset"
        # }).json()
        # or look at https://huggingface.co/tasks
        mteb_task_type_to_datasets = {
            # Text
            "BitextMining": ["translation"],
            "Classification": ["text-classification"],
            "MultilabelClassification": ["text-classification"],
            "Clustering": ["text-classification"],
            "PairClassification": ["text-classification"],
            "Reranking": ["text-ranking"],
            "Retrieval": ["text-retrieval"],
            "STS": ["sentence-similarity"],
            "Summarization": ["summarization"],
            "InstructionRetrieval": ["text-retrieval"],
            "InstructionReranking": ["text-ranking"],
            # Image
            "Any2AnyMultiChoice": ["visual-question-answering"],
            "Any2AnyRetrieval": ["visual-document-retrieval"],
            "Any2AnyMultilingualRetrieval": ["visual-document-retrieval"],
            "VisionCentricQA": ["visual-question-answering"],
            "ImageClustering": ["image-clustering"],
            "ImageClassification": ["image-classification"],
            "ImageMultilabelClassification": ["image-classification"],
            "DocumentUnderstanding": ["visual-document-retrieval"],
            "VisualSTS(eng)": ["other"],
            "VisualSTS(multi)": ["other"],
            "ZeroShotClassification": ["zero-shot-image-classification"],
            "Compositionality": ["other"],
        }

        dataset_type = mteb_task_type_to_datasets[self.type]

        if self.category in ["i2i", "it2i", "i2it", "it2it"]:
            dataset_type.append("image-to-image")
        if self.category in ["i2t", "t2i", "it2t", "it2i", "t2it", "i2it", "it2it"]:
            dataset_type.extend(["image-to-text", "text-to-image"])

        if self.is_multilingual:
            languages: list[str] = []
            for val in list(self.eval_langs.values()):
                languages.extend(val)
        else:
            languages: list[str] = self.eval_langs

        languages = sorted({lang.split("-")[0] for lang in languages})

        multilinguality = "multilingual" if self.is_multilingual else "monolingual"
        if self.sample_creation and "translated" in self.sample_creation:
            multilinguality = "translated"

        if self.adapted_from is not None:
            source_datasets = [
                task.metadata.dataset["path"]
                for task in mteb.get_tasks(self.adapted_from)
            ]
        else:
            source_datasets = None

        tags = ["mteb"] + self.modalities

        descriptive_stats = self.descriptive_stats
        if descriptive_stats is not None:
            for split, split_stat in descriptive_stats.items():
                if len(split_stat.get("hf_subset_descriptive_stats", {})) > 10:
                    split_stat.pop("hf_subset_descriptive_stats", {})
            descriptive_stats = json.dumps(descriptive_stats, indent=4)

        if existing_dataset_card_data is None:
            existing_dataset_card_data = DatasetCardData()

        dataset_license = self.license
        if dataset_license:
            license_mapping = {"not specified": "unknown", "msr-la-nc": "other"}
            dataset_license = license_mapping.get(
                dataset_license,
                "other" if dataset_license.startswith("http") else dataset_license,
            )

        dataset_card_data_params = existing_dataset_card_data.to_dict()
        # override the existing values
        dataset_card_data_params.update(
            dict(
                language=languages,
                license=dataset_license,
                annotations_creators=[self.annotations_creators]
                if self.annotations_creators
                else None,
                multilinguality=multilinguality,
                source_datasets=source_datasets,
                task_categories=dataset_type,
                task_ids=self._map_subtypes_to_hf(),
                tags=tags,
            )
        )

        return (
            DatasetCardData(**dataset_card_data_params),
            # parameters for readme generation
            dict(
                citation=self.bibtex_citation,
                dataset_description=self.description,
                dataset_reference=self.reference,
                descritptive_stats=descriptive_stats,
                dataset_task_name=self.name,
                category=self.category,
                domains=", ".join(self.domains) if self.domains else None,
            ),
        )

    def generate_dataset_card(
        self, existing_dataset_card: DatasetCard | None = None
    ) -> DatasetCard:
        """Generates a dataset card for the task.

        Args:
            existing_dataset_card: The existing dataset card to update. If None, a new dataset card will be created.

        Returns:
            DatasetCard: The dataset card for the task.
        """
        path = Path(__file__).parent / "dataset_card_template.md"
        existing_dataset_card_data = (
            existing_dataset_card.data if existing_dataset_card else None
        )
        dataset_card_data, template_kwargs = self.create_dataset_card_data(
            existing_dataset_card_data
        )
        dataset_card = DatasetCard.from_template(
            card_data=dataset_card_data,
            template_path=str(path),
            **template_kwargs,
        )
        return dataset_card

    def push_dataset_card_to_hub(self, repo_name: str) -> None:
        """Pushes the dataset card to the huggingface hub.

        Args:
            repo_name: The name of the repository to push the dataset card to.
        """
        dataset_card = None
        if repo_exists(
            repo_name, repo_type=constants.REPO_TYPE_DATASET
        ) and file_exists(
            repo_name, constants.REPOCARD_NAME, repo_type=constants.REPO_TYPE_DATASET
        ):
            dataset_card = DatasetCard.load(repo_name)
        dataset_card = self.generate_dataset_card(dataset_card)
        dataset_card.push_to_hub(repo_name, commit_message="Add dataset card")

    def _map_subtypes_to_hf(self) -> list[str]:
        # to get full list of available task_ids execute
        # requests.post("https://huggingface.co/api/validate-yaml", json={
        #   "content": "---\ntask_ids: 'test'\n---",
        #   "repoType": "dataset"
        # })
        mteb_to_hf_subtype = {
            "Article retrieval": ["document-retrieval"],
            "Conversational retrieval": ["conversational", "utterance-retrieval"],
            "Dialect pairing": [],
            "Dialog Systems": ["dialogue-modeling", "dialogue-generation"],
            "Discourse coherence": [],
            "Duplicate Image Retrieval": [],
            "Language identification": ["language-identification"],
            "Linguistic acceptability": ["acceptability-classification"],
            "Political classification": [],
            "Question answering": [
                "multiple-choice-qa",
            ],
            "Sentiment/Hate speech": [
                "sentiment-analysis",
                "sentiment-scoring",
                "sentiment-classification",
                "hate-speech-detection",
            ],
            "Thematic clustering": [],
            "Scientific Reranking": [],
            "Claim verification": ["fact-checking", "fact-checking-retrieval"],
            "Topic classification": ["topic-classification"],
            "Code retrieval": [],
            "False Friends": [],
            "Cross-Lingual Semantic Discrimination": [],
            "Textual Entailment": ["natural-language-inference"],
            "Counterfactual Detection": [],
            "Emotion classification": [],
            "Reasoning as Retrieval": [],
            "Rendered Texts Understanding": [],
            "Image Text Retrieval": [],
            "Object recognition": [],
            "Scene recognition": [],
            "Caption Pairing": ["image-captioning"],
            "Emotion recognition": [],
            "Textures recognition": [],
            "Activity recognition": [],
            "Tumor detection": [],
            "Duplicate Detection": [],
            "Rendered semantic textual similarity": [
                "semantic-similarity-scoring",
                "rendered semantic textual similarity",
            ],
            "Intent classification": [
                "intent-classification",
            ],
        }
        task_types_to_task_ids = {
            # Text
            "BitextMining": [],
            "Classification": [],
            "MultilabelClassification": ["multi-label-classification"],
            "Clustering": [],
            "PairClassification": ["semantic-similarity-classification"],
            "Reranking": [],
            "Retrieval": [],
            "STS": ["semantic-similarity-scoring"],
            "Summarization": [],
            "InstructionRetrieval": [],
            "InstructionReranking": [],
            # Image
            "Any2AnyMultiChoice": [],
            "Any2AnyRetrieval": [],
            "Any2AnyMultilingualRetrieval": [],
            "VisionCentricQA": ["visual-question-answering"],
            "ImageClustering": [],
            "ImageClassification": [],
            "ImageMultilabelClassification": [],
            "DocumentUnderstanding": [],
            "VisualSTS(eng)": [],
            "VisualSTS(multi)": [],
            "ZeroShotClassification": [],
            "Compositionality": [],
        }
        subtypes = task_types_to_task_ids.get(self.type, [])
        if self.task_subtypes:
            for subtype in self.task_subtypes:
                subtypes.extend(mteb_to_hf_subtype.get(subtype, []))
        return subtypes
