from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING, Any, TypedDict

import numpy as np
from datasets import Dataset, DatasetDict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import KFold

from mteb._create_dataloaders import create_dataloader
from mteb._evaluators.sklearn_evaluator import SklearnEvaluator
from mteb.models import EncoderProtocol
from mteb.types.statistics import (
    SplitDescriptiveStatistics,
)

from ._statistics_calculation import (
    _compute_modality_hashes,
    _count_samples_in_train,
    calculate_label_statistics,
    calculate_single_input_modality_statistics,
)
from .abstask import AbsTask

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from numpy.typing import NDArray

    from mteb._evaluators.sklearn_evaluator import SklearnModelProtocol
    from mteb.models import MTEBModels
    from mteb.types import Array, EncodeKwargs, HFSubset, Modalities, ScoresDict
    from mteb.types.statistics import (
        AudioStatistics,
        ImageStatistics,
        LabelStatistics,
        TextStatistics,
        VideoStatistics,
    )

logger = logging.getLogger(__name__)


class ClassificationDescriptiveStatistics(SplitDescriptiveStatistics):
    """Descriptive statistics for Classification

    Attributes:
        num_samples: number of samples in the dataset.
        samples_in_train: Number of unique test samples (across all input modalities)
            that also appear in the train split. None when evaluated on the train split itself.

        text_statistics: Statistics for text
        image_statistics: Statistics for images
        audio_statistics: Statistics for audio
        video_statistics: Statistics for video
        label_statistics: Statistics for labels
    """

    num_samples: int
    samples_in_train: int | None

    text_statistics: TextStatistics | None
    image_statistics: ImageStatistics | None
    audio_statistics: AudioStatistics | None
    video_statistics: VideoStatistics | None
    label_statistics: LabelStatistics


class ClassificationMetrics(TypedDict):
    """Scores for classification tasks

    Attributes:
        accuracy: Accuracy score.
        f1: F1 score (macro).
        f1_weighted: Weighted F1 score.
        precision: Precision score (macro).
        precision_weighted: Weighted precision score.
        recall: Recall score (macro).
        recall_weighted: Weighted recall score.
        ap: Average precision score (macro) for binary classification.
        ap_weighted: Weighted average precision score for binary classification.
    """

    accuracy: float
    f1: float
    f1_weighted: float
    precision: float
    precision_weighted: float
    recall: float
    recall_weighted: float
    ap: float | None
    ap_weighted: float | None


class FullClassificationMetrics(ClassificationMetrics):
    """Full classification metrics including scores per experiment. In main scores, the average over all experiments is reported.

    Attributes:
        scores_per_experiment: List of ClassificationMetrics for each experiment.
    """

    scores_per_experiment: list[ClassificationMetrics]


class AbsTaskClassification(AbsTask):
    """The class which classification tasks inherit from.

    A classification task consists of a dataset with input data and corresponding labels. The task is to predict the label for each input.
    The task works by training a sklearn compatible model on samples drawn from the training split of the dataset,
    where the input data is encoded using the provided model.
    The trained model is then evaluated on the evaluation split of the dataset. This process is repeated for `n_experiments` times, and both average and
    individual scores for each experiment are reported.

    Attributes:
        dataset: Hugging Face dataset containing the data for the task. Should have train split (split name can be changed by train_split. Must contain the following columns:
            text: str (for text) or PIL.Image (for image). Column name can be changed via `input_column_name` attribute.
            label: int. Column name can be changed via `label_column_name` attribute.
        evaluator_model: The model to use for evaluation. Can be any sklearn compatible model. Default is `LogisticRegression`.
        samples_per_label: Number of samples per label to use for training the evaluator model. Default is 8.
        n_experiments: Number of experiments to run. Default is 10.
        train_split: Name of the split to use for training the evaluator model. Default is "train".
        label_column_name: Name of the column containing the labels. Default is "label".
        input_column_name: Name of the column(s) containing the input data. Default is "text".
            Can be a string for single-column tasks or a list of strings for multimodal tasks (e.g. ["video", "audio"]).
            When specified as a list, values must be the default column names as defined in the encoder I/O types
            (see https://embeddings-benchmark.github.io/mteb/api/types/#mteb.types._encoder_io).
        abstask_prompt: Prompt to use for the task for instruction model if not prompt is provided in TaskMetadata.prompt.
        is_cross_validation: Is task cross validation
        n_splits: Number of splits for cross-validation
    """

    evaluator: type[SklearnEvaluator] = SklearnEvaluator
    evaluator_model: SklearnModelProtocol = LogisticRegression(
        max_iter=100,
    )

    samples_per_label: int = 8
    n_experiments: int = 10
    train_split: str = "train"
    label_column_name: str = "label"
    input_column_name: str | Sequence[Modalities] = "text"
    abstask_prompt = "Classify user passages."
    is_cross_validation: bool = False
    n_splits = 5

    def evaluate(
        self,
        model: MTEBModels,
        split: str = "test",
        subsets_to_run: list[HFSubset] | None = None,
        *,
        encode_kwargs: EncodeKwargs,
        prediction_folder: Path | None = None,
        num_proc: int | None = None,
        **kwargs: Any,
    ) -> dict[HFSubset, ScoresDict]:
        """Evaluate a model on the classification task.

        Differs from other tasks as it requires train split.
        """
        if not isinstance(model, EncoderProtocol):
            raise TypeError(
                f"Model {model} is a SearchProtocol, but this task {self.metadata.name} does not support Search. "
                "Please use a Encoder model instead."
            )

        if not self.data_loaded:
            self.load_data(num_proc=num_proc)

        if self.dataset is None:
            raise RuntimeError("Dataset not loaded.")

        if "random_state" in self.evaluator_model.get_params():
            self.evaluator_model = self.evaluator_model.set_params(
                random_state=self.seed
            )
        scores = {}
        hf_subsets = self.hf_subsets
        if subsets_to_run is not None:
            hf_subsets = [s for s in hf_subsets if s in subsets_to_run]

        for hf_subset in hf_subsets:
            logger.info(
                f"Task: {self.metadata.name}, split: {split}, subset: {hf_subset}. Running..."
            )

            if hf_subset not in self.dataset and hf_subset == "default":
                ds = self.dataset
            else:
                ds = self.dataset[hf_subset]

            if isinstance(ds, Dataset | DatasetDict):
                # Keep label and input columns
                input_cols = (
                    [self.input_column_name]
                    if isinstance(self.input_column_name, str)
                    else list(self.input_column_name)
                )
                columns_to_keep = set(input_cols) | {self.label_column_name}

                if isinstance(ds, DatasetDict):
                    available = set(next(iter(ds.values())).column_names)
                else:
                    available = set(ds.column_names)

                missing_columns = columns_to_keep - available
                if missing_columns:
                    raise ValueError(
                        f"Missing required columns in dataset: {missing_columns}. "
                        f"Available columns: {available}"
                    )
                ds = ds.select_columns(list(columns_to_keep))
            eval_function = (
                self._evaluate_subset
                if not self.is_cross_validation
                else self._evaluate_subset_cross_validation
            )
            scores[hf_subset] = eval_function(
                model,
                ds,
                hf_split=split,
                hf_subset=hf_subset,
                encode_kwargs=encode_kwargs,
                prediction_folder=prediction_folder,
                num_proc=num_proc,
                **kwargs,
            )
            self._add_main_score(scores[hf_subset])

        return scores  # type: ignore[return-value]

    def _evaluate_subset(
        self,
        model: MTEBModels,
        data_split: DatasetDict,
        *,
        encode_kwargs: EncodeKwargs,
        hf_split: str,
        hf_subset: str,
        prediction_folder: Path | None = None,
        num_proc: int | None = None,
        **kwargs: Any,
    ) -> FullClassificationMetrics:
        if not isinstance(model, EncoderProtocol):
            raise TypeError("Expected model to be an instance of EncoderProtocol")

        train_split = data_split[self.train_split]
        eval_split = data_split[hf_split]

        scores = []
        # we store idxs to make the shuffling reproducible
        test_cache, idxs = None, None

        all_predictions = []
        for i in range(self.n_experiments):
            logger.info(f"Running experiment ({i}/{self.n_experiments})")
            scores_exp, predictions, idxs, test_cache = self._run_experiment(
                model,
                train_split,
                eval_split,
                experiment_num=i,
                idxs=idxs,
                test_cache=test_cache,
                encode_kwargs=encode_kwargs,
                hf_split=hf_split,
                hf_subset=hf_subset,
                num_proc=num_proc,
            )

            if prediction_folder:
                all_predictions.append(predictions)
            scores.append(scores_exp)

        if prediction_folder:
            self._save_task_predictions(
                all_predictions,
                model,
                prediction_folder,
                hf_subset=hf_subset,
                hf_split=hf_split,
            )

        return self._calculate_avg_scores(scores)

    def _evaluate_subset_cross_validation(
        self,
        model: EncoderProtocol,
        data_split: DatasetDict,
        *,
        encode_kwargs: EncodeKwargs,
        hf_split: str,
        hf_subset: str,
        prediction_folder: Path | None = None,
        num_proc: int | None = None,
        **kwargs: Any,
    ) -> FullClassificationMetrics:
        if self.train_split != hf_split:
            raise ValueError(
                f"Performing {self.n_splits}-fold cross validation, but the dataset has a train (`{self.train_split}`) and test split (`{hf_split}`)! Set `is_cross_validation` to False, and retry."
            )
        logger.info(
            f"Performing {self.n_splits}-fold cross-validation on the entire dataset!"
        )

        ds = data_split[self.train_split]
        num_samples = len(ds)

        scores = []
        idxs = None
        cross_validation_splitter = KFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.seed
        )
        all_predictions = []
        dataloader_train = create_dataloader(
            ds,
            task_metadata=self.metadata,
            input_column=self.input_column_name,
            num_proc=num_proc,
            **encode_kwargs,
        )
        logger.info("Running cross-validation - Encoding samples...")
        # precompute all embeddings for cross-validation to not recomupute them in different k-folds
        dataset_embeddings = model.encode(
            dataloader_train,
            task_metadata=self.metadata,
            hf_split=hf_split,
            hf_subset=hf_subset,
            **encode_kwargs,
        )
        for i, (train_idx, val_idx) in enumerate(
            cross_validation_splitter.split(range(num_samples))
        ):
            train_split = ds.select(train_idx)
            eval_split = ds.select(val_idx)
            train_cache = dataset_embeddings[train_idx]
            test_cache = dataset_embeddings[val_idx]
            logger.info(f"Running experiment ({i}/{self.n_experiments})")
            scores_exp, predictions, idxs, _ = self._run_experiment(
                model,
                train_split,
                eval_split,
                experiment_num=i,
                idxs=idxs,
                encode_kwargs=encode_kwargs,
                hf_split=hf_split,
                hf_subset=hf_subset,
                test_cache=test_cache,
                train_cache=train_cache,
                num_proc=num_proc,
            )

            if prediction_folder:
                all_predictions.append(predictions)
            scores.append(scores_exp)

        if prediction_folder:
            self._save_task_predictions(
                all_predictions,
                model,
                prediction_folder,
                hf_subset=hf_subset,
                hf_split=hf_split,
            )
        return self._calculate_avg_scores(scores)

    def _run_experiment(  # noqa: PLR0913
        self,
        model: EncoderProtocol,
        train_split: Dataset,
        eval_split: Dataset,
        *,
        experiment_num: int,
        idxs: list[int] | None,
        test_cache: Array | None,
        encode_kwargs: EncodeKwargs,
        hf_split: str,
        hf_subset: str,
        train_cache: Array | None = None,
        num_proc: int | None = None,
    ) -> tuple[ClassificationMetrics, list[float], list[int], Array]:
        train_dataset, idxs, selected_idx = self._undersample_data(
            train_split,
            experiment_num,
            idxs,
        )
        sub_train_cache = None
        if train_cache is not None:
            sub_train_cache = train_cache[selected_idx]

        evaluator = self.evaluator(
            train_dataset,
            eval_split,
            values_column_name=self.input_column_name,
            label_column_name=self.label_column_name,
            task_metadata=self.metadata,
            hf_split=hf_split,
            hf_subset=hf_subset,
            evaluator_model=self.evaluator_model,
        )
        y_pred, test_cache = evaluator(
            model,
            encode_kwargs=encode_kwargs,
            test_cache=test_cache,
            train_cache=sub_train_cache,
            num_proc=num_proc,
        )
        y_test = eval_split[self.label_column_name]
        return self._calculate_scores(y_test, y_pred), y_pred.tolist(), idxs, test_cache

    def _calculate_avg_scores(
        self, scores: list[ClassificationMetrics]
    ) -> FullClassificationMetrics:
        avg_scores: dict[str, Any] = {
            # ap will be none for non binary classification tasks
            k: (
                float(np.mean(values))
                if (values := [s[k] for s in scores if s[k] is not None])  # type: ignore[literal-required]
                else np.nan
            )
            for k in scores[0].keys()
        }
        logger.info(f"Running {self.metadata.name} - Finished.")
        return FullClassificationMetrics(
            scores_per_experiment=scores,
            **avg_scores,  # type: ignore[typeddict-item]
        )

    def _calculate_scores(  # noqa: PLR6301
        self,
        y_test: NDArray[np.integer] | list[int],
        y_pred: NDArray[np.integer | np.floating] | list[int],
    ) -> ClassificationMetrics:
        scores = ClassificationMetrics(
            accuracy=accuracy_score(y_test, y_pred),
            f1=f1_score(y_test, y_pred, average="macro"),
            f1_weighted=f1_score(y_test, y_pred, average="weighted"),
            precision=precision_score(y_test, y_pred, average="macro"),
            precision_weighted=precision_score(y_test, y_pred, average="weighted"),
            recall=recall_score(y_test, y_pred, average="macro"),
            recall_weighted=recall_score(y_test, y_pred, average="weighted"),
            ap=None,
            ap_weighted=None,
        )

        # if binary classification
        if len(np.unique(y_test)) == 2:
            scores["ap"] = average_precision_score(y_test, y_pred, average="macro")
            scores["ap_weighted"] = average_precision_score(
                y_test, y_pred, average="weighted"
            )
        return scores

    def _undersample_data(
        self, dataset: Dataset, experiment_num: int, idxs: list[int] | None = None
    ) -> tuple[Dataset, list[int], list[int]]:
        """Undersample data to have `samples_per_label` samples of each label.

        Args:
            dataset: Hugging Face `datasets.Dataset` containing "text" and "label".
            experiment_num: Experiment number, used to set the random seed.
            idxs: Optional indices to shuffle and sample from.

        Returns:
            Tuple of:
            - A new Dataset containing undersampled examples.
            - The shuffled indices used for sampling.
            - Selected indexes
        """
        if idxs is None:
            idxs = list(range(len(dataset)))

        # using RandomState for backward compatibility with `v1`
        rng_state = np.random.RandomState(self.seed)
        rng_state.shuffle(idxs)

        label_counter: dict[str, int] = defaultdict(int)
        sampled_idxs = []

        for i in idxs:
            label = dataset[i][self.label_column_name]
            if label_counter[label] < self.samples_per_label:
                sampled_idxs.append(i)
                label_counter[label] += 1

        return dataset.select(sampled_idxs), idxs, sampled_idxs

    def _load_statistics_col_inputs_and_hashes(
        self, split: str, hf_subset: str | None, compute_overall: bool
    ) -> tuple[
        dict[Modalities, list[Any]],
        list,
        dict[str, list[str]],
        dict[str, list[str]] | None,
    ]:
        """Load input columns, label data, and pre-computed hashes for a split."""
        if isinstance(self.input_column_name, str):
            col_map = {self.metadata.modalities[0]: self.input_column_name}
        else:
            col_map = {col: col for col in self.input_column_name}

        if hf_subset:
            ds = self.dataset[hf_subset][split]
            col_inputs = {mod: ds[col] for mod, col in col_map.items()}
            label = ds[self.label_column_name]
            train_inputs = (
                {
                    mod: self.dataset[hf_subset][self.train_split][col]
                    for mod, col in col_map.items()
                }
                if split != self.train_split
                else None
            )
        elif compute_overall:
            col_inputs = {mod: [] for mod in col_map}
            label = []
            train_inputs = (
                {mod: [] for mod in col_map} if split != self.train_split else None
            )
            for subset in self.metadata.eval_langs:
                ds = self.dataset[subset][split]
                for mod, col in col_map.items():
                    col_inputs[mod].extend(ds[col])
                label.extend(ds[self.label_column_name])
                if train_inputs is not None:
                    for mod, col in col_map.items():
                        train_inputs[mod].extend(
                            self.dataset[subset][self.train_split][col]
                        )
        else:
            ds = self.dataset[split]
            col_inputs = {mod: ds[col] for mod, col in col_map.items()}
            label = ds[self.label_column_name]
            train_inputs = (
                {
                    mod: self.dataset[self.train_split][col]
                    for mod, col in col_map.items()
                }
                if split != self.train_split
                else None
            )

        # Compute hashes once; reuse for both statistics (uniqueness counts) and
        # train/test intersection — avoids decoding expensive media (e.g. video frames) twice.
        test_hashes = _compute_modality_hashes(col_inputs)
        train_hashes = (
            _compute_modality_hashes(train_inputs) if train_inputs is not None else None
        )
        return col_inputs, label, test_hashes, train_hashes

    def _calculate_descriptive_statistics_from_split(
        self, split: str, hf_subset: str | None = None, compute_overall: bool = False
    ) -> ClassificationDescriptiveStatistics:
        col_inputs, label, test_hashes, train_hashes = (
            self._load_statistics_col_inputs_and_hashes(
                split, hf_subset, compute_overall
            )
        )
        modality_stats = calculate_single_input_modality_statistics(
            col_inputs, test_hashes
        )
        return ClassificationDescriptiveStatistics(
            num_samples=len(label),
            samples_in_train=_count_samples_in_train(test_hashes, train_hashes),
            **modality_stats,
            label_statistics=calculate_label_statistics(label),
        )

    def _push_dataset_to_hub(
        self,
        repo_name: str,
        num_proc: int | None = None,
    ) -> None:
        input_cols = (
            [self.input_column_name]
            if isinstance(self.input_column_name, str)
            else list(self.input_column_name)
        )
        self._upload_dataset_to_hub(
            repo_name,
            input_cols + [self.label_column_name],
            num_proc=num_proc,
        )
