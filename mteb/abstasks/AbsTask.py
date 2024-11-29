from __future__ import annotations

import json
import logging
import random
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import datasets
import numpy as np
import torch
import tqdm
from datasets import Dataset, DatasetDict
from sklearn.preprocessing import MultiLabelBinarizer

from mteb.abstasks.stratification import _iterative_train_test_split
from mteb.abstasks.TaskMetadata import DescriptiveStatistics, HFSubset, TaskMetadata
from mteb.encoder_interface import Encoder
from mteb.languages import LanguageScripts

logger = logging.getLogger(__name__)

ScoresDict = dict[str, Any]
# ^ e.g {'main_score': 0.5, 'hf_subset': 'en-de', 'languages': ['eng-Latn', 'deu-Latn']}


def _multilabel_subsampling(
    dataset_dict: DatasetDict,
    seed: int,
    splits: list[str] = ["test"],
    label: str = "label",
    n_samples: int = 2048,
) -> DatasetDict:
    """Multilabel subsampling the dataset with stratification by the supplied label.
    Returns a DatasetDict object.

    Args:
        dataset_dict: the DatasetDict object.
        seed: the random seed.
        splits: the splits of the dataset.
        label: the label with which the stratified sampling is based on.
        n_samples: Optional, number of samples to subsample. Default is max_n_samples.
    """
    for split in splits:
        n_split = len(dataset_dict[split])
        X_np = np.arange(n_split).reshape((-1, 1))
        binarizer = MultiLabelBinarizer()
        labels_np = binarizer.fit_transform(dataset_dict[split][label])
        _, test_idx = _iterative_train_test_split(
            X_np, labels_np, test_size=n_samples / n_split, random_state=seed
        )
        dataset_dict.update({split: Dataset.from_dict(dataset_dict[split][test_idx])})
    return dataset_dict


class AbsTask(ABC):
    metadata: TaskMetadata
    abstask_prompt: str | None = None
    _eval_splits: list[str] | None = None
    superseded_by: None | str = None
    dataset: dict[HFSubset, DatasetDict] | None = None  # type: ignore
    data_loaded: bool = False
    is_multilingual: bool = False

    def __init__(self, seed: int = 42, **kwargs: Any):
        self.save_suffix = kwargs.get("save_suffix", "")

        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

    def check_if_dataset_is_superseded(self):
        """Check if the dataset is superseded by a newer version"""
        if self.superseded_by:
            logger.warning(
                f"Dataset '{self.metadata.name}' is superseded by '{self.superseded_by}', you might consider using the newer version of the dataset."
            )

    def dataset_transform(self):
        """Transform operations applied to the dataset after loading.
        Override this method if your dataset requires any transformation.
        """
        pass

    def evaluate(
        self,
        model: Encoder,
        split: str = "test",
        *,
        encode_kwargs: dict[str, Any] = {},
        **kwargs: Any,
    ) -> dict[HFSubset, ScoresDict]:
        """Evaluates a Sentence Embedding Model on the task.
        Returns a dict (that can be serialized to json).

        Args:
            model: Sentence embedding method. Implements a encode(sentences) method, that encodes sentences and returns a numpy matrix with the
                sentence embeddings
            split: Which datasplit to be used.
            encode_kwargs: Additional keyword arguments that are passed to the model's `encode` method.
            kwargs: Additional keyword arguments that are passed to the _evaluate_subset method.
        """
        if not self.data_loaded:
            self.load_data()

        self.dataset: dict[HFSubset, DatasetDict]

        scores = {}
        hf_subsets = list(self.dataset.keys()) if self.is_multilingual else ["default"]

        for hf_subset in hf_subsets:
            logger.info(
                f"\nTask: {self.metadata_dict['name']}, split: {split}, subset: {hf_subset}. Running..."
            )
            if hf_subset not in self.dataset and hf_subset == "default":
                data_split = self.dataset[split]
            else:
                data_split = self.dataset[hf_subset][split]
            scores[hf_subset] = self._evaluate_subset(
                model, data_split, encode_kwargs=encode_kwargs, **kwargs
            )
        return scores

    @abstractmethod
    def _evaluate_subset(
        self,
        model: Encoder,
        data_split: DatasetDict | Dataset,
        encode_kwargs: dict[str, Any],
        **kwargs: Any,
    ) -> ScoresDict:
        raise NotImplementedError(
            "If you are using the default evaluate method, you must implement _evaluate_subset method."
        )

    @staticmethod
    def stratified_subsampling(
        dataset_dict: datasets.DatasetDict,
        seed: int,
        splits: list[str] = ["test"],
        label: str = "label",
        n_samples: int = 2048,
    ) -> datasets.DatasetDict:
        """Subsamples the dataset with stratification by the supplied label.
        Returns a datasetDict object.

        Args:
            dataset_dict: the DatasetDict object.
            seed: the random seed.
            splits: the splits of the dataset.
            label: the label with which the stratified sampling is based on.
            n_samples: Optional, number of samples to subsample. Default is max_n_samples.
        """
        ## Can only do this if the label column is of ClassLabel.
        if not isinstance(dataset_dict[splits[0]].features[label], datasets.ClassLabel):
            try:
                dataset_dict = dataset_dict.class_encode_column(label)
            except ValueError as e:
                if isinstance(dataset_dict[splits[0]][label][0], Sequence):
                    return _multilabel_subsampling(
                        dataset_dict, seed, splits, label, n_samples
                    )
                else:
                    raise e

        for split in splits:
            if n_samples >= len(dataset_dict[split]):
                logger.debug(
                    f"Subsampling not needed for split {split}, as n_samples is equal or greater than the number of samples."
                )
                continue
            dataset_dict.update(
                {
                    split: dataset_dict[split].train_test_split(
                        test_size=n_samples, seed=seed, stratify_by_column=label
                    )["test"]
                }
            )  ## only take the specified test split.
        return dataset_dict

    def load_data(self, **kwargs):
        """Load dataset from HuggingFace hub"""
        if self.data_loaded:
            return
        self.dataset = datasets.load_dataset(**self.metadata_dict["dataset"])  # type: ignore
        self.dataset_transform()
        self.data_loaded = True

    def calculate_metadata_metrics(
        self, overwrite_results: bool = False
    ) -> dict[str, DescriptiveStatistics | dict[str, DescriptiveStatistics]]:
        if self.metadata.descriptive_stat_path.exists() and not overwrite_results:
            logger.info("Loading metadata descriptive statistics from cache.")
            return self.metadata.descriptive_stats

        self.load_data()

        descriptive_stats = {}
        hf_subset_stat = "hf_subset_descriptive_stats"
        eval_splits = self.metadata.eval_splits
        if self.metadata.type in ["Classification", "MultilabelClassification"]:
            eval_splits += ["train"]

        pbar_split = tqdm.tqdm(eval_splits, desc="Processing Splits...")
        for split in pbar_split:
            pbar_split.set_postfix_str(f"Split: {split}")
            logger.info(f"Processing metadata for split {split}")
            if self.is_multilingual:
                descriptive_stats[split] = self._calculate_metrics_from_split(
                    split, compute_overall=True
                )
                descriptive_stats[split][hf_subset_stat] = {}

                eval_langs = (
                    list(self.metadata.eval_langs.keys())
                    if isinstance(self.metadata.eval_langs, dict)
                    else self.metadata.eval_langs
                )

                pbar_subsets = tqdm.tqdm(eval_langs, desc="Processing Languages...")
                for hf_subset in pbar_subsets:
                    pbar_subsets.set_postfix_str(f"Language: {hf_subset}")
                    logger.info(f"Processing metadata for language {hf_subset}")
                    split_details = self._calculate_metrics_from_split(split, hf_subset)
                    descriptive_stats[split][hf_subset_stat][hf_subset] = split_details
            else:
                split_details = self._calculate_metrics_from_split(split)
                descriptive_stats[split] = split_details

        with self.metadata.descriptive_stat_path.open("w") as f:
            json.dump(descriptive_stats, f, indent=4)

        return descriptive_stats

    @abstractmethod
    def _calculate_metrics_from_split(
        self, split: str, hf_subset: str | None = None, compute_overall: bool = False
    ) -> DescriptiveStatistics:
        raise NotImplementedError

    @property
    def metadata_dict(self) -> dict[str, Any]:
        return dict(self.metadata)

    @property
    def languages(self) -> list[str]:
        """Returns the languages of the task"""
        # check if self.hf_subsets is set
        if self.is_multilingual and hasattr(self, "hf_subsets"):
            assert isinstance(
                self.metadata.eval_langs, dict
            ), "eval_langs must be dict for multilingual tasks"
            eval_langs = self.metadata.eval_langs
            languages = []

            for lang in self.hf_subsets:
                for langscript in eval_langs[lang]:
                    iso_lang, script = langscript.split("-")
                    languages.append(iso_lang)

            return sorted(set(languages))

        return self.metadata.languages

    def filter_eval_splits(self, eval_splits: list[str] | None) -> AbsTask:
        """Filter the evaluation splits of the task."""
        self._eval_splits = eval_splits
        return self

    def filter_languages(
        self, languages: list[str] | None, script: list[str] | None = None
    ) -> AbsTask:
        """Filter the languages of the task.

        Args:
            languages: list of languages to filter the task by can be either a 3-letter langauge code (e.g. "eng") or also include the script
                (e.g. "eng-Latn")
            script: list of scripts to filter the task by. Will be ignored if language code specified the script. If None, all scripts are included.
                If the language code does not specify the script the intersection of the language and script will be used.
        """
        lang_scripts = LanguageScripts.from_languages_and_scripts(languages, script)

        subsets_to_keep = []

        if not isinstance(self.metadata.eval_langs, dict):
            self.hf_subsets = self.metadata.eval_langs
            return self

        for hf_subset, langs in self.metadata.eval_langs.items():
            for langscript in langs:
                if lang_scripts.contains_language(
                    langscript
                ) or lang_scripts.contains_script(langscript):
                    subsets_to_keep.append(hf_subset)
                    break

        self.hf_subsets = subsets_to_keep
        return self

    @property
    def eval_splits(self) -> list[str]:
        if self._eval_splits:
            return self._eval_splits
        return self.metadata.eval_splits

    def __repr__(self) -> str:
        """Format the representation of the task such that it appears as:

        TaskObjectName(name='{name}', languages={lang1, lang2, ...})
        """
        langs = self.languages
        if len(langs) > 3:
            langs = langs[:3]
            langs.append("...")
        return (
            f"{self.__class__.__name__}(name='{self.metadata.name}', languages={langs})"
        )

    def __hash__(self) -> int:
        return hash(self.metadata)
