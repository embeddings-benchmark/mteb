from __future__ import annotations

import logging
import random
from abc import ABC, abstractmethod
from typing import Any

import datasets
import numpy as np
import torch
from datasets import DatasetDict

from mteb.abstasks.languages import LanguageScripts
from mteb.abstasks.TaskMetadata import TaskMetadata
from mteb.encoder_interface import Encoder, EncoderWithQueryCorpusEncode

from .MTEBResults import HFSubset, ScoresDict

logger = logging.getLogger(__name__)


class AbsTask(ABC):
    metadata: TaskMetadata
    superseeded_by: None | str = None

    def __init__(self, seed: int = 42, **kwargs: Any):
        self.dataset = None
        self.data_loaded = False
        self.is_multilingual = False
        self.is_crosslingual = False
        self.save_suffix = kwargs.get("save_suffix", "")

        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

    def check_if_dataset_is_superseeded(self):
        """Check if the dataset is superseeded by a newer version"""
        if self.superseeded_by:
            logger.warning(
                f"Dataset '{self.metadata.name}' is superseeded by '{self.superseeded_by}', you might consider using the newer version of the dataset."
            )

    def dataset_transform(self):
        """Transform operations applied to the dataset after loading.
        Override this method if your dataset requires any transformation.
        """
        pass

    def evaluate(
        self,
        model: Encoder | EncoderWithQueryCorpusEncode,
        split: str = "test",
        **kwargs: Any,
    ) -> dict[HFSubset, ScoresDict]:
        """Evaluates a Sentence Embedding Model on the task.
        Returns a dict (that can be serialized to json).

        Args:
            model: Sentence embedding method. Implements a encode(sentences) method, that encodes sentences and returns a numpy matrix with the
                sentence embeddings
            split: Which datasplit to be used.
            kwargs: Additional keyword arguments that are passed to the _evaluate_subset method.
        """
        if not self.data_loaded:
            self.load_data()

        self.dataset: dict[HFSubset, DatasetDict]

        scores = {}
        hf_subsets = (
            [l for l in self.dataset.keys()]
            if self.is_crosslingual or self.is_multilingual
            else ["default"]
        )

        for hf_subset in hf_subsets:
            logger.info(
                f"\nTask: {self.metadata_dict['name']}, split: {split}, subset: {hf_subset}. Running..."
            )
            if hf_subset not in self.dataset and hf_subset == "default":
                data_split = self.dataset[split]
            else:
                data_split = self.dataset[hf_subset][split]
            scores[hf_subset] = self._evaluate_subset(model, data_split, **kwargs)
        return scores

    @abstractmethod
    def _evaluate_subset(self, model, data_split, **kwargs) -> ScoresDict:
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
            dataset_dict = dataset_dict.class_encode_column(label)

        for split in splits:
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

    @property
    def metadata_dict(self) -> dict[str, Any]:
        return dict(self.metadata)

    @property
    def languages(self) -> list[str]:
        """Returns the languages of the task"""
        # check if self.hf_subsets is set
        has_lang_splits = self.is_crosslingual or self.is_multilingual
        if has_lang_splits and hasattr(self, "hf_subsets"):
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
