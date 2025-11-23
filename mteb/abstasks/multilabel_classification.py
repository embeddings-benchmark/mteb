import itertools
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
from datasets import DatasetDict
from sklearn.base import clone
from sklearn.metrics import f1_score, label_ranking_average_precision_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from typing_extensions import override

from mteb._create_dataloaders import create_dataloader
from mteb._evaluators.sklearn_evaluator import SklearnModelProtocol
from mteb.models import EncoderProtocol

from .classification import AbsTaskClassification

logger = logging.getLogger(__name__)


def _evaluate_classifier(
    embeddings_train: np.ndarray,
    y_train: np.ndarray,
    embeddings_test: np.ndarray,
    classifier: SklearnModelProtocol,
) -> tuple[np.ndarray, SklearnModelProtocol]:
    classifier: SklearnModelProtocol = clone(classifier)
    classifier.fit(embeddings_train, y_train)
    return classifier.predict(embeddings_test), classifier


class MultilabelClassificationMetrics(TypedDict):
    """Scores for multilabel classification tasks.

    Attributes:
        accuracy: Accuracy of the classifier.
        lrap: Label Ranking Average Precision (LRAP) score.
        f1: Macro F1 score.
    """

    accuracy: float
    lrap: float
    f1: float


class FullMultilabelClassificationMetrics(MultilabelClassificationMetrics):
    """Extended scores for multilabel classification tasks.

    Attributes:
        scores_per_experiment: List of individual experiment scores.
    """

    scores_per_experiment: list[MultilabelClassificationMetrics]


class AbsTaskMultilabelClassification(AbsTaskClassification):
    """Abstract class for multioutput classification tasks

    Attributes:
        dataset: Huggingface dataset containing the data for the task. Dataset must contain columns specified by input_column_name and label_column_name.
            Input column must contain the text or image to be classified, and label column must contain a list of labels for each example.
        input_column_name: Name of the column containing the input text.
        label_column_name: Name of the column containing the labels.
        samples_per_label: Number of samples to use pr. label. These samples are embedded and a classifier is fit using the labels and samples.
        evaluator: Classifier to use for evaluation. Must implement the SklearnModelProtocol.
    """

    evaluator: SklearnModelProtocol = KNeighborsClassifier(n_neighbors=5)
    input_column_name: str = "text"
    label_column_name: str = "label"

    @override
    def _evaluate_subset(
        self,
        model: EncoderProtocol,
        data_split: DatasetDict,
        *,
        encode_kwargs: dict[str, Any],
        hf_split: str,
        hf_subset: str,
        prediction_folder: Path | None = None,
        **kwargs: Any,
    ) -> FullMultilabelClassificationMetrics:
        if isinstance(data_split, DatasetDict):
            data_split = data_split.select_columns(
                [self.input_column_name, self.label_column_name]
            )
        train_split = data_split[self.train_split]
        eval_split = data_split[hf_split]

        logger.info(
            "Running multilabel classification task - Sampling training data..."
        )
        scores = []
        # Bootstrap sample indices from training set for each experiment
        train_samples = []
        for _ in range(self.n_experiments):
            sample_indices, _ = self._undersample_data_indices(
                train_split[self.label_column_name], self.samples_per_label, None
            )
            train_samples.append(sample_indices)
        # Encode all unique sentences at the indices
        unique_train_indices = list(set(itertools.chain.from_iterable(train_samples)))
        unique_train_dataset = train_split.select(unique_train_indices).select_columns(
            self.input_column_name
        )
        dataloader_train = create_dataloader(
            unique_train_dataset,
            self.metadata,
            input_column=self.input_column_name,
            **encode_kwargs,
        )

        logger.info("Running multilabel classification - Encoding training set...")
        _unique_train_embeddings = model.encode(
            dataloader_train,
            task_metadata=self.metadata,
            hf_split=self.train_split,
            hf_subset=hf_subset,
            **encode_kwargs,
        )
        unique_train_embeddings = dict(
            zip(unique_train_indices, _unique_train_embeddings)
        )
        # Stratified subsampling of test set to 2000 examples.
        test_dataset = eval_split
        try:
            if len(test_dataset) > 2000:
                split_dataset = eval_split.train_test_split(
                    test_size=2000, seed=42, stratify_by_column="label"
                )
                test_dataset = split_dataset["test"]
        except ValueError:
            logger.warning("Couldn't subsample, continuing with the entire test set.")

        dataloader_test = create_dataloader(
            test_dataset.select_columns(self.input_column_name),
            self.metadata,
            input_column=self.input_column_name,
            **encode_kwargs,
        )

        logger.info("Running multilabel classification - Encoding test set...")
        X_test = model.encode(
            dataloader_test,
            task_metadata=self.metadata,
            hf_split=hf_split,
            hf_subset=hf_subset,
            **encode_kwargs,
        )
        binarizer = MultiLabelBinarizer()
        y_test = binarizer.fit_transform(test_dataset[self.label_column_name])

        logger.info("Running multilabel classification - Evaluating classifiers...")
        all_predictions = []
        for i_experiment, sample_indices in enumerate(train_samples):
            X_train = np.stack([unique_train_embeddings[idx] for idx in sample_indices])
            y_train = train_split.select(sample_indices)[self.label_column_name]
            y_train = binarizer.transform(y_train)
            y_pred, current_classifier = _evaluate_classifier(
                X_train, y_train, X_test, self.evaluator
            )
            if prediction_folder:
                all_predictions.append(y_pred.tolist())

            scores_exp = self._calculate_scores(
                y_test, y_pred, X_test, current_classifier
            )
            scores.append(scores_exp)

        if prediction_folder:
            self._save_task_predictions(
                all_predictions,
                model,
                prediction_folder,
                hf_subset=hf_subset,
                hf_split=hf_split,
            )

        avg_scores: dict[str, Any] = {
            k: np.mean([s[k] for s in scores]) for k in scores[0].keys()
        }
        logger.info("Running multilabel classification - Finished.")
        return FullMultilabelClassificationMetrics(
            scores_per_experiment=scores,
            **avg_scores,
        )

    def _calculate_scores(
        self,
        y_test: np.ndarray,
        y_pred: np.ndarray,
        x_test_embedding: np.ndarray,
        current_classifier: SklearnModelProtocol,
    ) -> MultilabelClassificationMetrics:
        accuracy = current_classifier.score(x_test_embedding, y_test)
        if isinstance(current_classifier, MultiOutputClassifier):
            predictions = current_classifier.predict_proba(x_test_embedding)
            all_probs = [emb[:, 1] for emb in predictions]

            y_score = np.stack(all_probs, axis=1)  # shape: (n_samples, n_labels)
            lrap = label_ranking_average_precision_score(y_test, y_score)
        else:
            lrap = label_ranking_average_precision_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")
        return MultilabelClassificationMetrics(
            accuracy=accuracy,
            lrap=lrap,
            f1=f1,
        )

    def _undersample_data_indices(
        self, y: list[list[int]], samples_per_label: int, idxs: list[int] | None = None
    ) -> tuple[list[int], list[int]]:
        """Undersample data to have samples_per_label samples of each label.

        Returns:
            A tuple containing:
                - List of sampled indices.
                - List of all indices after shuffling.
        """
        sample_indices = []
        if idxs is None:
            idxs = np.arange(len(y))
        self.np_rng.shuffle(idxs)
        idxs = idxs.tolist()
        label_counter = defaultdict(int)
        for i in idxs:
            if any((label_counter[label] < samples_per_label) for label in y[i]):
                sample_indices.append(i)
                for label in y[i]:
                    label_counter[label] += 1
        return sample_indices, idxs
