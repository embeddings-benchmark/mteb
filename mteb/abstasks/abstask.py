import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from copy import copy
from pathlib import Path
from typing import Any, cast

import numpy as np
from datasets import ClassLabel, Dataset, DatasetDict, load_dataset
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm.auto import tqdm
from typing_extensions import Self

from mteb._set_seed import _set_seed
from mteb.abstasks.task_metadata import TaskMetadata
from mteb.languages import LanguageScripts
from mteb.models import (
    CrossEncoderProtocol,
    EncoderProtocol,
    MTEBModels,
    SearchProtocol,
)
from mteb.types import HFSubset, Modalities, ScoresDict
from mteb.types.statistics import DescriptiveStatistics, SplitDescriptiveStatistics

logger = logging.getLogger(__name__)


def _multilabel_subsampling(
    dataset_dict: DatasetDict,
    seed: int,
    splits: list[str] = ["test"],
    label: str = "label",
    n_samples: int = 2048,
) -> DatasetDict:
    """Multilabel subsampling the dataset with stratification by the supplied label.

    Args:
        dataset_dict: the DatasetDict object.
        seed: the random seed.
        splits: the splits of the dataset.
        label: the label with which the stratified sampling is based on.
        n_samples: Optional, number of samples to subsample. Default is max_n_samples.

    Returns:
        A subsampled DatasetDict object.
    """
    from ._stratification import _iterative_train_test_split

    for split in splits:
        n_split = len(dataset_dict[split])
        x_np = np.arange(n_split).reshape((-1, 1))
        binarizer = MultiLabelBinarizer()
        labels_np = binarizer.fit_transform(dataset_dict[split][label])
        _, test_idx = _iterative_train_test_split(
            x_np, labels_np, test_size=n_samples / n_split, random_state=seed
        )
        dataset_dict.update({split: Dataset.from_dict(dataset_dict[split][test_idx])})
    return dataset_dict


class AbsTask(ABC):
    """The abstract class for the tasks

    Attributes:
        metadata: The metadata describing the task
        dataset: The dataset represented as a dictionary on the form {"hf subset": {"split": Dataset}} where "split" is the dataset split (e.g. "test")
            and Dataset is a datasets.Dataset object. "hf subset" is the data subset on Huggingface typically used to denote the language e.g.
            datasets.load_dataset("data", "en"). If the dataset does not have a subset this is simply "default".
        seed: The random seed used for reproducibility.
        hf_subsets: The list of Huggingface subsets to use.
        data_loaded: Denotes if the dataset is loaded or not. This is used to avoid loading the dataset multiple times.
        abstask_prompt: Prompt to use for the task for instruction model if not prompt is provided in TaskMetadata.prompt.
        fast_loading: **Deprecated**. Denotes if the task should be loaded using the fast loading method.
            This is only possible if the dataset have a "default" config. We don't recommend to use this method, and suggest to use different subsets for loading datasets.
            This was used only for historical reasons and will be removed in the future.
    """

    metadata: TaskMetadata
    abstask_prompt: str | None = None
    _eval_splits: list[str] | None = None
    dataset: dict[HFSubset, DatasetDict] | None = None
    data_loaded: bool = False
    hf_subsets: list[HFSubset]
    fast_loading: bool = False

    _support_cross_encoder: bool = False
    _support_search: bool = False

    def __init__(self, seed: int = 42, **kwargs: Any) -> None:
        """The init function. This is called primarily to set the seed.

        Args:
            seed: An integer seed.
            kwargs: arguments passed to subclasses.
        """
        self.seed = seed
        self.rng_state, self.np_rng = _set_seed(seed)
        self.hf_subsets = self.metadata.hf_subsets

    def check_if_dataset_is_superseded(self) -> None:
        """Check if the dataset is superseded by a newer version."""
        if self.superseded_by:
            logger.warning(
                f"Dataset '{self.metadata.name}' is superseded by '{self.superseded_by}', you might consider using the newer version of the dataset."
            )

    def dataset_transform(self):
        """A transform operations applied to the dataset after loading.

        This method is useful when the dataset from Huggingface is not in an `mteb` compatible format.
        Override this method if your dataset requires additional transformation.
        """
        pass

    def evaluate(
        self,
        model: MTEBModels,
        split: str = "test",
        subsets_to_run: list[HFSubset] | None = None,
        *,
        encode_kwargs: dict[str, Any],
        prediction_folder: Path | None = None,
        **kwargs: Any,
    ) -> dict[HFSubset, ScoresDict]:
        """Evaluates an MTEB compatible model on the task.

        Args:
            model: MTEB compatible model. Implements a encode(sentences) method, that encodes sentences and returns an array of embeddings
            split: Which split (e.g. *"test"*) to be used.
            subsets_to_run: List of huggingface subsets (HFSubsets) to evaluate. If None, all subsets are evaluated.
            encode_kwargs: Additional keyword arguments that are passed to the model's `encode` method.
            prediction_folder: Folder to save model predictions
            kwargs: Additional keyword arguments that are passed to the _evaluate_subset method.

        Returns:
            A dictionary with the scores for each subset.

        Raises:
            TypeError: If the model is a CrossEncoder and the task does not support CrossEncoders.
            TypeError: If the model is a SearchProtocol and the task does not support Search.
        """
        if isinstance(model, CrossEncoderProtocol) and not self._support_cross_encoder:
            raise TypeError(
                f"Model {model} is a CrossEncoder, but this task {self.metadata.name} does not support CrossEncoders. "
                "Please use a Encoder model instead."
            )

        # encoders might implement search protocols
        if (
            isinstance(model, SearchProtocol)
            and not isinstance(model, EncoderProtocol)
            and not self._support_search
        ):
            raise TypeError(
                f"Model {model} is a SearchProtocol, but this task {self.metadata.name} does not support Search. "
                "Please use a Encoder model instead."
            )

        if not self.data_loaded:
            self.load_data()

        self.dataset = cast(dict[HFSubset, DatasetDict], self.dataset)

        scores = {}
        if self.hf_subsets is None:
            hf_subsets = list(self.dataset.keys())
        else:
            hf_subsets = copy(self.hf_subsets)

        if subsets_to_run is not None:  # allow overwrites of pre-filtering
            hf_subsets = [s for s in hf_subsets if s in subsets_to_run]

        for hf_subset in hf_subsets:
            logger.info(
                f"Running task {self.metadata.name} ({split=}, {hf_subset=})..."
            )
            if hf_subset not in self.dataset and hf_subset == "default":
                data_split = self.dataset[split]
            else:
                data_split = self.dataset[hf_subset][split]
            scores[hf_subset] = self._evaluate_subset(
                model,
                data_split,
                hf_split=split,
                hf_subset=hf_subset,
                encode_kwargs=encode_kwargs,
                prediction_folder=prediction_folder,
                **kwargs,
            )
            self._add_main_score(scores[hf_subset])
        return scores

    @abstractmethod
    def _evaluate_subset(
        self,
        model: EncoderProtocol,
        data_split: Dataset,
        *,
        encode_kwargs: dict[str, Any],
        hf_split: str,
        hf_subset: str,
        prediction_folder: Path | None = None,
        **kwargs: Any,
    ) -> ScoresDict:
        raise NotImplementedError(
            "If you are using the default evaluate method, you must implement _evaluate_subset method."
        )

    def _save_task_predictions(
        self,
        predictions: dict[str, Any] | list[Any],
        model: MTEBModels,
        prediction_folder: Path,
        hf_split: str,
        hf_subset: str,
    ) -> None:
        """Saves the predictions of the model on the task to a json file.

        Args:
            predictions: Dictionary containing the predictions.
            model: The model used to generate the predictions.
            prediction_folder: The folder to save the predictions to.
            hf_split: The split of the dataset (e.g. "test").
            hf_subset: The subset of the dataset (e.g. "en").
        """
        predictions_path = self._predictions_path(prediction_folder)
        existing_results = {
            "mteb_model_meta": {
                "model_name": model.mteb_model_meta.name,
                "revision": model.mteb_model_meta.revision,
            }
        }
        if predictions_path.exists():
            with predictions_path.open("r") as predictions_file:
                existing_results = json.load(predictions_file)

        if hf_subset not in existing_results:
            existing_results[hf_subset] = {}

        existing_results[hf_subset][hf_split] = predictions
        with predictions_path.open("w") as predictions_file:
            json.dump(existing_results, predictions_file)

    def _predictions_path(
        self,
        output_folder: Path | str,
    ) -> Path:
        if isinstance(output_folder, str):
            output_folder = Path(output_folder)

        if not output_folder.exists():
            output_folder.mkdir(parents=True, exist_ok=True)
        return output_folder / self.prediction_file_name

    @property
    def prediction_file_name(self) -> str:
        """The name of the prediction file in format {task_name}_predictions.json"""
        return f"{self.metadata.name}_predictions.json"

    @staticmethod
    def stratified_subsampling(
        dataset_dict: DatasetDict,
        seed: int,
        splits: list[str] = ["test"],
        label: str = "label",
        n_samples: int = 2048,
    ) -> DatasetDict:
        """Subsamples the dataset with stratification by the supplied label.

        Args:
            dataset_dict: the DatasetDict object.
            seed: the random seed.
            splits: the splits of the dataset.
            label: the label with which the stratified sampling is based on.
            n_samples: Optional, number of samples to subsample. Default is max_n_samples.

        Returns:
            A subsampled DatasetDict object.
        """
        # Can only do this if the label column is of ClassLabel.
        if not isinstance(dataset_dict[splits[0]].features[label], ClassLabel):
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
            )  # only take the specified test split.
        return dataset_dict

    def load_data(self) -> None:
        """Loads dataset from HuggingFace hub

        This is the main loading function for Task. Do not overwrite this, instead we recommend using `dataset_transform`, which is called after the
        dataset is loaded using `datasets.load_dataset`.
        """
        if self.data_loaded:
            return
        if self.metadata.is_multilingual:
            if self.fast_loading:
                self.fast_load()
            else:
                self.dataset = {}
                for hf_subset in self.hf_subsets:
                    self.dataset[hf_subset] = load_dataset(
                        name=hf_subset,
                        **self.metadata.dataset,
                    )
        else:
            # some of monolingual datasets explicitly adding the split name to the dataset name
            self.dataset = load_dataset(**self.metadata.dataset)  # type: ignore
        self.dataset_transform()
        self.data_loaded = True

    def fast_load(self) -> None:
        """**Deprecated**. Load all subsets at once, then group by language. Using fast loading has two requirements:

        - Each row in the dataset should have a 'lang' feature giving the corresponding language/language pair
        - The datasets must have a 'default' config that loads all the subsets of the dataset (see more [here](https://huggingface.co/docs/datasets/en/repository_structure#configurations))
        """
        self.dataset = {}
        merged_dataset = load_dataset(**self.metadata.dataset)  # load "default" subset
        for split in merged_dataset.keys():
            df_split = merged_dataset[split].to_polars()
            df_grouped = dict(df_split.group_by(["lang"]))
            for lang in set(df_split["lang"].unique()) & set(self.hf_subsets):
                self.dataset.setdefault(lang, {})
                self.dataset[lang][split] = Dataset.from_polars(
                    df_grouped[(lang,)].drop("lang")
                )  # Remove lang column and convert back to HF datasets, not strictly necessary but better for compatibility
        for lang, subset in self.dataset.items():
            self.dataset[lang] = DatasetDict(subset)

    def calculate_descriptive_statistics(
        self, overwrite_results: bool = False
    ) -> dict[str, DescriptiveStatistics]:
        """Calculates descriptive statistics from the dataset.

        Args:
            overwrite_results: Whether to overwrite existing results. If False and results already exist, the existing results will be loaded from cache.

        Returns:
            A dictionary containing descriptive statistics for each split.
        """
        from mteb.abstasks import AbsTaskClassification

        if self.metadata.descriptive_stat_path.exists() and not overwrite_results:
            logger.info("Loading metadata descriptive statistics from cache.")
            return self.metadata.descriptive_stats

        if not self.data_loaded:
            self.load_data()

        descriptive_stats: dict[str, DescriptiveStatistics] = {}
        hf_subset_stat = "hf_subset_descriptive_stats"
        eval_splits = self.metadata.eval_splits
        if isinstance(self, AbsTaskClassification):
            eval_splits.append(self.train_split)

        pbar_split = tqdm(eval_splits, desc="Processing Splits...")
        for split in pbar_split:
            pbar_split.set_postfix_str(f"Split: {split}")
            logger.info(f"Processing metadata for split {split}")
            if self.metadata.is_multilingual:
                descriptive_stats[split] = (
                    self._calculate_descriptive_statistics_from_split(
                        split, compute_overall=True
                    )
                )
                descriptive_stats[split][hf_subset_stat] = {}

                pbar_subsets = tqdm(
                    self.metadata.hf_subsets,
                    desc="Processing Languages...",
                )
                for hf_subset in pbar_subsets:
                    pbar_subsets.set_postfix_str(f"Huggingface subset: {hf_subset}")
                    logger.info(f"Processing metadata for subset {hf_subset}")
                    split_details = self._calculate_descriptive_statistics_from_split(
                        split, hf_subset
                    )
                    descriptive_stats[split][hf_subset_stat][hf_subset] = split_details
            else:
                split_details = self._calculate_descriptive_statistics_from_split(split)
                descriptive_stats[split] = split_details

        with self.metadata.descriptive_stat_path.open("w") as f:
            json.dump(descriptive_stats, f, indent=4)

        return descriptive_stats

    def calculate_metadata_metrics(
        self, overwrite_results: bool = False
    ) -> dict[str, DescriptiveStatistics]:
        """Old name of `calculate_descriptive_statistics`, kept for backward compatibility."""
        return self.calculate_descriptive_statistics(
            overwrite_results=overwrite_results
        )

    @abstractmethod
    def _calculate_descriptive_statistics_from_split(
        self, split: str, hf_subset: str | None = None, compute_overall: bool = False
    ) -> SplitDescriptiveStatistics:
        raise NotImplementedError

    @property
    def languages(self) -> list[str]:
        """Returns the languages of the task."""
        if self.hf_subsets:
            eval_langs = self.metadata.hf_subsets_to_langscripts
            languages = []

            for lang in self.hf_subsets:
                for langscript in eval_langs[lang]:
                    iso_lang, script = langscript.split("-")
                    languages.append(iso_lang)

            return sorted(set(languages))

        return self.metadata.languages

    def filter_eval_splits(self, eval_splits: list[str] | None) -> Self:
        """Filter the evaluation splits of the task.

        Args:
            eval_splits: A list of evaluation splits to keep. If None, all splits are kept.

        Returns:
            The filtered task
        """
        self._eval_splits = eval_splits
        return self

    def filter_languages(
        self,
        languages: list[str] | None,
        script: list[str] | None = None,
        hf_subsets: list[HFSubset] | None = None,
        exclusive_language_filter: bool = False,
    ) -> Self:
        """Filter the languages of the task.

        Args:
            languages: list of languages to filter the task by can be either a 3-letter language code (e.g. "eng") or also include the script
                (e.g. "eng-Latn")
            script: A list of scripts to filter the task by. Will be ignored if language code specified the script. If None, all scripts are included.
                If the language code does not specify the script the intersection of the language and script will be used.
            hf_subsets: A list of huggingface subsets to filter on. This is useful if a dataset have multiple subsets containing the desired language,
                but you only want to test on one. An example is STS22 which e.g. have both "en" and "de-en" which both contains English.
            exclusive_language_filter: Some datasets contains more than one language e.g. for STS22 the subset "de-en" contain eng and deu. If
                exclusive_language_filter is set to False both of these will be kept, but if set to True only those that contains all the languages
                specified will be kept.

        Returns:
            The filtered task
        """
        lang_scripts = LanguageScripts.from_languages_and_scripts(languages, script)

        subsets_to_keep = []

        for hf_subset, langs in self.metadata.hf_subsets_to_langscripts.items():
            if (hf_subsets is not None) and (hf_subset not in hf_subsets):
                continue
            if exclusive_language_filter is False:
                for langscript in langs:
                    if lang_scripts.contains_language(
                        langscript
                    ) or lang_scripts.contains_script(langscript):
                        subsets_to_keep.append(hf_subset)
                        break

            if exclusive_language_filter is True and languages:
                if lang_scripts.contains_languages(langs):
                    subsets_to_keep.append(hf_subset)

        if len(subsets_to_keep) == 0:
            raise ValueError(
                f"No subsets were found for {self.metadata.name} with filters: language code {languages}, script {script}, hf subsets {hf_subsets}."
            )

        self.hf_subsets = subsets_to_keep
        return self

    def _add_main_score(self, scores: dict[HFSubset, ScoresDict]) -> None:
        scores["main_score"] = scores[self.metadata.main_score]

    def _upload_dataset_to_hub(
        self, repo_name: str, fields: list[str] | dict[str, str]
    ) -> None:
        if self.metadata.is_multilingual:
            for config in self.metadata.eval_langs:
                logger.info(f"Converting {config} of {self.metadata.name}")
                sentences = {}
                for split in self.dataset[config]:
                    if isinstance(fields, dict):
                        sentences[split] = Dataset.from_dict(
                            {
                                mapped_name: self.dataset[config][split][original_name]
                                for original_name, mapped_name in fields.items()
                            }
                        )
                    else:
                        sentences[split] = Dataset.from_dict(
                            {
                                field: self.dataset[config][split][field]
                                for field in fields
                            }
                        )
                sentences = DatasetDict(sentences)
                sentences.push_to_hub(
                    repo_name, config, commit_message=f"Add {config} dataset"
                )
        else:
            sentences = {}
            for split in self.dataset:
                if isinstance(fields, dict):
                    sentences[split] = Dataset.from_dict(
                        {
                            mapped_name: self.dataset[split][original_name]
                            for original_name, mapped_name in fields.items()
                        }
                    )
                else:
                    sentences[split] = Dataset.from_dict(
                        {field: self.dataset[split][field] for field in fields}
                    )
            sentences = DatasetDict(sentences)
            sentences.push_to_hub(repo_name, commit_message="Add dataset")

    def _push_dataset_to_hub(self, repo_name: str) -> None:
        raise NotImplementedError

    def push_dataset_to_hub(self, repo_name: str) -> None:
        """Push the dataset to the HuggingFace Hub.

        Args:
            repo_name: The name of the repository to push the dataset to.

        Examples:
            >>> import mteb
            >>> task = mteb.get_task("Caltech101")
            >>> repo_name = f"myorg/{task.metadata.name}"
            >>> # Push the dataset to the Hub
            >>> task.push_dataset_to_hub(repo_name)
        """
        if not self.data_loaded:
            self.load_data()

        self._push_dataset_to_hub(repo_name)
        # dataset repo not creating when pushing card
        self.metadata.push_dataset_card_to_hub(repo_name)

    @property
    def is_aggregate(self) -> bool:
        """Whether the task is an aggregate of multiple tasks."""
        return False

    @property
    def eval_splits(self) -> list[str]:
        """Returns the evaluation splits of the task."""
        if self._eval_splits:
            return self._eval_splits
        return self.metadata.eval_splits

    @property
    def modalities(self) -> list[Modalities]:
        """Returns the modalities of the task."""
        return self.metadata.modalities

    def __repr__(self) -> str:
        # Format the representation of the task such that it appears as:
        # TaskObjectName(name='{name}', languages={lang1, lang2, ...})

        langs = self.languages
        if len(langs) > 3:
            langs = langs[:3]
            langs.append("...")
        return (
            f"{self.__class__.__name__}(name='{self.metadata.name}', languages={langs})"
        )

    def __hash__(self) -> int:
        return hash(self.metadata)

    def unload_data(self) -> None:
        """Unloads the dataset from memory"""
        if self.data_loaded:
            self.dataset = None
            self.data_loaded = False
            logger.info(f"Unloaded dataset {self.metadata.name} from memory.")
        else:
            logger.warning(
                f"Dataset {self.metadata.name} is not loaded, cannot unload it."
            )

    @property
    def superseded_by(self) -> str | None:
        """If the dataset is superseded by another dataset, return the name of the new dataset."""
        return self.metadata.superseded_by
