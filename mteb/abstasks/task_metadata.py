import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

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
from typing_extensions import Required, TypedDict

import mteb
from mteb.languages import check_language_code
from mteb.types import (
    HFSubset,
    ISOLanguageScript,
    Languages,
    Licenses,
    Modalities,
    PromptType,
    StrDate,
    StrURL,
)
from mteb.types.statistics import DescriptiveStatistics

logger = logging.getLogger(__name__)

TaskSubtype = Literal[
    "Article retrieval",
    "Patent retrieval",
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
"""The subtypes of the task. E.g. includes "Sentiment/Hate speech", "Thematic Clustering". This list can be updated as needed."""

TaskDomain = Literal[
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
"""
The domains follow the categories used in the [Universal Dependencies project](https://universaldependencies.org), though
 we updated them where deemed appropriate. These do not have to be mutually exclusive.
"""

SampleCreationMethod = Literal[
    "found",
    "created",
    "created and machine-translated",
    "human-translated and localized",
    "human-translated",
    "machine-translated",
    "machine-translated and verified",
    "machine-translated and localized",
    "LM-generated and verified",
    "machine-translated and LM verified",
    "rendered",
    "multiple",
]
"""How the text was created. It can be an important factor for understanding the quality of a dataset. E.g. used to filter out machine-translated datasets."""

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
    "Regression",
    "Reranking",
    "Retrieval",
    "STS",
    "Summarization",
    "InstructionRetrieval",
    "InstructionReranking",
) + MIEB_TASK_TYPE

TaskType = Literal[_TASK_TYPE]
"""The type of the task. E.g. includes "Classification", "Retrieval" and "Clustering"."""


TaskCategory = Literal[
    "t2t",
    "t2c",
    "i2i",
    "i2c",
    "i2t",
    "t2i",
    "it2t",
    "it2i",
    "i2it",
    "t2it",
    "it2it",
]
"""The category of the task.

1. t2t: text to text
2. t2c: text to category
3. i2i: image to image
4. i2c: image to category
5. i2t: image to text
6. t2i: text to image
7. it2t: image+text to text
8. it2i: image+text to image
9. i2it: image to image+text
10. t2it: text to image+text
11. it2it: image+text to image+text
"""

AnnotatorType = Literal[
    "expert-annotated",
    "human-annotated",
    "derived",
    "LM-generated",
    "LM-generated and reviewed",  # reviewed by humans
]
"""The type of the annotators. Is often important for understanding the quality of a dataset."""


PromptDict = TypedDict(
    "PromptDict", {prompt_type.value: str for prompt_type in PromptType}, total=False
)
"""A dictionary containing the prompt used for the task.

Attributes:
    query: The prompt used for the queries in the task.
    document: The prompt used for the passages in the task.
"""


class MetadataDatasetDict(TypedDict, total=False):
    """A dictionary containing the dataset path and revision.

    Attributes:
        path: The path to the dataset.
        revision: The revision of the dataset.
        name: The name the dataset config.
        split: The split of the dataset.
        trust_remote_code: Whether to use `trust_remote_code`. Datasets shouldn't use this since,
         because datasets `v4` doesn't support this. This parameter is left for compatibility with forks/external usage.
    """

    path: Required[str]
    revision: Required[str]
    name: str
    split: str
    trust_remote_code: bool


class TaskMetadata(BaseModel):
    """Metadata for a task.

    Attributes:
        dataset: All arguments to pass to [datasets.load_dataset](https://huggingface.co/docs/datasets/v2.18.0/en/package_reference/loading_methods#datasets.load_dataset) to load the dataset for the task.
        name: The name of the task.
        description: A description of the task.
        type: The type of the task. This includes "Classification", "Summarization", "STS", "Retrieval", "Reranking", "Clustering",
            "PairClassification", "BitextMining". The type should match the abstask type.
        category: The category of the task. E.g. includes "t2t" (text to text), "t2i" (text to image).
        reference: A URL to the documentation of the task. E.g. a published paper.
        eval_splits: The splits of the dataset used for evaluation.
        eval_langs: The languages of the dataset used for evaluation. Languages follows a ETF BCP 47 standard consisting of "{language}-{script}"
            tag (e.g. "eng-Latn"). Where language is specified as a list of ISO 639-3 language codes (e.g. "eng") followed by ISO 15924 script codes
            (e.g. "Latn"). Can be either a list of languages or a dictionary mapping huggingface subsets to lists of languages (e.g. if a the
            huggingface dataset contain different languages).
        main_score: The main score used for evaluation.
        date: The date when the data was collected. Specified as a tuple of two dates.
        domains: The domains of the data. This includes "Non-fiction", "Social", "Fiction", "News", "Academic", "Blog", "Encyclopaedic",
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
        is_public: Whether the dataset is publicly available. If False (closed/private), a HuggingFace token is required to run the datasets.
        superseded_by: Denotes the task that this task is superseded by. Used to issue warning to users of outdated datasets, while maintaining
            reproducibility of existing benchmarks.
    """

    model_config = ConfigDict(extra="forbid")

    dataset: MetadataDatasetDict

    name: str
    description: str
    prompt: str | PromptDict | None = None
    type: TaskType
    modalities: list[Modalities] = ["text"]
    category: TaskCategory | None = None
    reference: StrURL | None = None

    eval_splits: list[str] = ["test"]
    eval_langs: Languages
    main_score: str

    date: tuple[StrDate, StrDate] | None = None
    domains: list[TaskDomain] | None = None
    task_subtypes: list[TaskSubtype] | None = None
    license: Licenses | StrURL | None = None

    annotations_creators: AnnotatorType | None = None
    dialect: list[str] | None = None

    sample_creation: SampleCreationMethod | None = None
    bibtex_citation: str | None = None
    adapted_from: Sequence[str] | None = None
    is_public: bool = True
    superseded_by: str | None = None

    def _validate_metadata(self) -> None:
        self._eval_langs_are_valid(self.eval_langs)

    @field_validator("prompt")
    @classmethod
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

    def _eval_langs_are_valid(self, eval_langs: Languages) -> None:
        """This method checks that the eval_langs are specified as a list of languages."""
        if isinstance(eval_langs, dict):
            for langs in eval_langs.values():
                for code in langs:
                    check_language_code(code)
        else:
            for code in eval_langs:
                check_language_code(code)

    @property
    def bcp47_codes(self) -> list[ISOLanguageScript]:
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
        """Check if all the metadata fields are filled.

        Returns:
            True if all the metadata fields are filled, False otherwise.
        """
        return all(
            getattr(self, field_name) is not None
            for field_name in self.model_fields
            if field_name not in ["prompt", "adapted_from", "superseded_by"]
        )

    @property
    def hf_subsets_to_langscripts(self) -> dict[HFSubset, list[ISOLanguageScript]]:
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
        """Return the dataset revision."""
        return self.dataset["revision"]

    def get_modalities(self, prompt_type: PromptType | None = None) -> list[Modalities]:
        """Get the modalities for the task based category if prompt_type provided.

        Args:
            prompt_type: The prompt type to get the modalities for.

        Returns:
            A list of modalities for the task.

        Raises:
            ValueError: If the prompt type is not recognized.
        """
        if prompt_type is None:
            return self.modalities
        query_modalities, doc_modalities = self.category.split("2")
        category_to_modality: dict[str, Modalities] = {
            "t": "text",
            "i": "image",
        }
        if prompt_type == PromptType.query:
            return [
                category_to_modality[query_modality]
                for query_modality in query_modalities
            ]
        if prompt_type == PromptType.document:
            return [
                category_to_modality[doc_modality] for doc_modality in doc_modalities
            ]
        raise ValueError(f"Unknown prompt type: {prompt_type}")

    def _create_dataset_card_data(
        self,
        existing_dataset_card_data: DatasetCardData | None = None,
    ) -> tuple[DatasetCardData, dict[str, Any]]:
        """Create a DatasetCardData object from the task metadata.

        Args:
            existing_dataset_card_data: The existing DatasetCardData object to update. If None, a new object will be created.

        Returns:
            A DatasetCardData object with the metadata for the task with kwargs to card
        """
        if existing_dataset_card_data is None:
            existing_dataset_card_data = DatasetCardData()

        dataset_type = [
            *self._hf_task_type(),
            *self._hf_task_category(),
            *self._hf_subtypes(),
        ]
        languages = self._hf_languages()

        multilinguality = "monolingual" if len(languages) == 1 else "multilingual"
        if self.sample_creation and "translated" in self.sample_creation:
            multilinguality = "translated"

        if self.adapted_from is not None:
            source_datasets = [
                task.metadata.dataset["path"]
                for task in mteb.get_tasks(self.adapted_from)
            ]
            source_datasets.append(self.dataset["path"])
        else:
            source_datasets = None if not self.dataset else [self.dataset["path"]]

        tags = ["mteb"] + self.modalities

        descriptive_stats = self.descriptive_stats
        if descriptive_stats is not None:
            for split, split_stat in descriptive_stats.items():
                if len(split_stat.get("hf_subset_descriptive_stats", {})) > 10:
                    split_stat.pop("hf_subset_descriptive_stats", {})
            descriptive_stats = json.dumps(descriptive_stats, indent=4)

        dataset_card_data_params = existing_dataset_card_data.to_dict()
        # override the existing values
        dataset_card_data_params.update(
            dict(
                language=languages,
                license=self._hf_license(),
                annotations_creators=[self.annotations_creators]
                if self.annotations_creators
                else None,
                multilinguality=multilinguality,
                source_datasets=source_datasets,
                task_categories=dataset_type,
                task_ids=self._hf_subtypes(),
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
                descriptive_stats=descriptive_stats,
                dataset_task_name=self.name,
                category=self.category,
                domains=", ".join(self.domains) if self.domains else None,
            ),
        )

    def generate_dataset_card(
        self,
        existing_dataset_card: DatasetCard | None = None,
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
        dataset_card_data, template_kwargs = self._create_dataset_card_data(
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

    def _hf_subtypes(self) -> list[str]:
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
                "question-answering",
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
        subtypes = []
        if self.task_subtypes:
            for subtype in self.task_subtypes:
                subtypes.extend(mteb_to_hf_subtype.get(subtype, []))
        return subtypes

    def _hf_task_type(self) -> list[str]:
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
            "ZeroShotClassification": ["zero-shot-classification"],
            "Compositionality": ["other"],
        }
        if self.type == "ZeroShotClassification":
            if self.modalities == ["image"]:
                return ["zero-shot-image-classification"]
            return ["zero-shot-classification"]

        return mteb_task_type_to_datasets[self.type]

    def _hf_task_category(self) -> list[str]:
        dataset_type = []
        if self.category in ["i2i", "it2i", "i2it", "it2it"]:
            dataset_type.append("image-to-image")
        if self.category in ["i2t", "t2i", "it2t", "it2i", "t2it", "i2it", "it2it"]:
            dataset_type.extend(["image-to-text", "text-to-image"])
        if self.category in ["it2t", "it2i", "t2it", "i2it", "it2it"]:
            dataset_type.extend(["image-text-to-text"])
        return dataset_type

    def _hf_languages(self) -> list[str]:
        languages: list[str] = []
        if self.is_multilingual:
            for val in list(self.eval_langs.values()):
                languages.extend(val)
        else:
            languages = self.eval_langs
        # value "python" is not valid. It must be an ISO 639-1, 639-2 or 639-3 code (two/three letters),
        # or a special value like "code", "multilingual".
        readme_langs = []
        for lang in languages:
            lang_name, family = lang.split("-")
            if family == "Code":
                readme_langs.append("code")
            else:
                readme_langs.append(lang_name)
        return sorted(set(readme_langs))

    def _hf_license(self) -> str:
        dataset_license = self.license
        if dataset_license:
            license_mapping = {
                "not specified": "unknown",
                "msr-la-nc": "other",
                "cc-by-nd-2.1-jp": "other",
            }
            dataset_license = license_mapping.get(
                dataset_license,
                "other" if dataset_license.startswith("http") else dataset_license,
            )
        return dataset_license
