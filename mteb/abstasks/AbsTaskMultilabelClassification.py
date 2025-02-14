from __future__ import annotations

import itertools
import logging
import warnings
from collections import Counter, defaultdict
from typing import Any

import numpy as np
from sklearn.base import ClassifierMixin, clone
from sklearn.metrics import f1_score, label_ranking_average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from mteb.encoder_interface import Encoder

from ..load_results.task_results import HFSubset, ScoresDict
from .AbsTask import AbsTask
from .TaskMetadata import DescriptiveStatistics

logger = logging.getLogger(__name__)


def evaluate_classifier(
    embeddings_train: np.ndarray,
    y_train: np.ndarray,
    embeddings_test: np.ndarray,
    y_test: np.ndarray,
    classifier: ClassifierMixin,
):
    scores = {}
    classifier = clone(classifier)
    classifier.fit(embeddings_train, y_train)
    y_pred = classifier.predict(embeddings_test)
    accuracy = classifier.score(embeddings_test, y_test)
    f1 = f1_score(y_test, y_pred, average="macro")
    scores["accuracy"] = accuracy
    scores["f1"] = f1
    lrap = label_ranking_average_precision_score(y_test, y_pred)
    scores["lrap"] = lrap
    return scores


class MultilabelClassificationDescriptiveStatistics(DescriptiveStatistics):
    """Descriptive statistics for MultilabelClassification

    Attributes:
        num_samples: number of samples in the dataset.
        number_of_characters: Total number of symbols in the dataset.
        number_texts_in_train: Number of texts in the train split

        min_text_length: Minimum length of text
        average_text_length: Average length of text
        max_text_length: Maximum length of text
        unique_texts: Number of unique texts

        min_labels_per_text: Minimum number of labels per text
        average_label_per_text: Average number of labels per text
        max_labels_per_text: Maximum number of labels per text
        unique_labels: Number of unique labels
        labels: dict of label frequencies
    """

    num_samples: int
    number_of_characters: int
    number_texts_in_train: int | None

    min_text_length: int
    average_text_length: float
    max_text_length: int
    unique_texts: int

    min_labels_per_text: int
    average_label_per_text: float
    max_labels_per_text: int
    unique_labels: int
    labels: dict[str, dict[str, int]]


class AbsTaskMultilabelClassification(AbsTask):
    """Abstract class for multioutput classification tasks
    The similarity is computed between pairs and the results are ranked.

    self.load_data() must generate a huggingface dataset with a split matching self.metadata_dict["eval_splits"], and assign it to self.dataset. It must contain the following columns:
        text: str
        label: list[list[int]]

    Attributes:
       samples_per_label: Number of samples to use pr. label. These samples are embedded and a classifier is fit using the labels and samples.

    """

    classifier = KNeighborsClassifier(n_neighbors=5)
    abstask_prompt = "Classify user passages."
    samples_per_label: int = 8

    def __init__(
        self,
        n_experiments=None,
        batch_size=32,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.batch_size = batch_size

        if n_experiments:
            warnings.warn(
                "Passing `n_experiments` to AbsTaskMultilabelClassification is deprecated and will be removed in v2.0.0.",
                DeprecationWarning,
            )
        # Bootstrap parameters
        self.n_experiments = n_experiments or getattr(self, "n_experiments", 10)

        # Run metadata validation by instantiating addressing the attribute
        # This is quite hacky. Ideally, this would be done in the constructor of
        # each concrete task, but then we have to duplicate the __init__ method's
        # interface.
        if hasattr(self, "metadata"):
            self.metadata

    def _add_main_score(self, scores):
        scores["main_score"] = scores[self.metadata.main_score]

    def evaluate(
        self,
        model: Encoder,
        eval_split: str = "test",
        train_split: str = "train",
        subsets_to_run: list[HFSubset] | None = None,
        *,
        encode_kwargs: dict[str, Any] = {},
        **kwargs: Any,
    ) -> dict[HFSubset, ScoresDict]:
        if not self.data_loaded:
            self.load_data()

        if train_split != "train":
            warnings.warn(
                "Passing `train_split` to AbsTaskClassification.evaluate is deprecated and will be removed in v2.0.0.",
                DeprecationWarning,
            )

        scores = {}
        hf_subsets = list(self.dataset) if self.is_multilingual else ["default"]
        # If subsets_to_run is specified, filter the hf_subsets accordingly
        if subsets_to_run is not None:
            hf_subsets = [s for s in hf_subsets if s in subsets_to_run]

        for hf_subset in hf_subsets:
            logger.info(
                f"\nTask: {self.metadata.name}, split: {eval_split}, subset: {hf_subset}. Running..."
            )

            if hf_subset not in self.dataset and hf_subset == "default":
                ds = self.dataset
            else:
                ds = self.dataset[hf_subset]
            scores[hf_subset] = self._evaluate_subset(
                model,
                ds,
                eval_split,
                train_split,
                encode_kwargs=encode_kwargs,
                **kwargs,
            )
            self._add_main_score(scores[hf_subset])

        return scores

    def _evaluate_subset(
        self,
        model: Encoder,
        dataset,
        eval_split: str = "test",
        train_split: str = "train",
        *,
        encode_kwargs: dict[str, Any] = {},
        **kwargs: Any,
    ) -> ScoresDict:
        train_split = dataset[train_split]
        eval_split = dataset[eval_split]
        params = {
            "classifier_type": type(self.classifier).__name__,
            "classifier_params": self.classifier.get_params(),
            "batch_size": self.batch_size,
        }
        params.update(kwargs)

        scores = []
        # Bootstrap sample indices from training set for each experiment
        train_samples = []
        for _ in range(self.n_experiments):
            sample_indices, _ = self._undersample_data_indices(
                train_split["label"], self.samples_per_label, None
            )
            train_samples.append(sample_indices)
        # Encode all unique sentences at the indices
        unique_train_indices = list(set(itertools.chain.from_iterable(train_samples)))
        unique_train_sentences = train_split.select(unique_train_indices)["text"]

        _unique_train_embeddings = model.encode(
            unique_train_sentences,
            task_name=self.metadata.name,
            **encode_kwargs,
        )
        unique_train_embeddings = dict(
            zip(unique_train_indices, _unique_train_embeddings)
        )
        test_text = eval_split["text"]
        binarizer = MultiLabelBinarizer()
        y_test = binarizer.fit_transform(eval_split["label"])
        # Stratified subsampling of test set to 2000 examples.
        try:
            if len(test_text) > 2000:
                test_text, _, y_test, _ = train_test_split(
                    test_text, y_test, stratify=y_test, train_size=2000
                )
        except ValueError:
            logger.warning("Couldn't subsample, continuing with the entire test set.")

        X_test = model.encode(
            test_text,
            task_name=self.metadata.name,
            **encode_kwargs,
        )
        for i_experiment, sample_indices in enumerate(train_samples):
            logger.info(
                "=" * 10
                + f" Experiment {i_experiment + 1}/{self.n_experiments} "
                + "=" * 10
            )
            X_train = np.stack([unique_train_embeddings[idx] for idx in sample_indices])
            y_train = train_split.select(sample_indices)["label"]
            y_train = binarizer.transform(y_train)
            scores_exp = evaluate_classifier(
                X_train, y_train, X_test, y_test, self.classifier
            )
            scores.append(scores_exp)

        avg_scores: dict[str, Any] = {
            k: np.mean([s[k] for s in scores]) for k in scores[0].keys()
        }
        avg_scores["scores_per_experiment"] = scores

        return avg_scores

    def _undersample_data_indices(self, y, samples_per_label, idxs=None):
        """Undersample data to have samples_per_label samples of each label"""
        sample_indices = []
        if idxs is None:
            idxs = np.arange(len(y))
        np.random.shuffle(idxs)
        label_counter = defaultdict(int)
        for i in idxs:
            if any((label_counter[label] < samples_per_label) for label in y[i]):
                sample_indices.append(i)
                for label in y[i]:
                    label_counter[label] += 1
        return sample_indices, idxs

    def _calculate_metrics_from_split(
        self, split: str, hf_subset: str | None = None, compute_overall: bool = False
    ) -> MultilabelClassificationDescriptiveStatistics:
        train_text = []
        if hf_subset:
            text = self.dataset[hf_subset][split]["text"]
            label = self.dataset[hf_subset][split]["label"]
            if split != "train":
                train_text = self.dataset[hf_subset]["train"]["text"]
        elif compute_overall:
            text = []
            label = []
            for hf_subset in self.metadata.eval_langs:
                text.extend(self.dataset[hf_subset][split]["text"])
                label.extend(self.dataset[hf_subset][split]["label"])
                if split != "train":
                    train_text.extend(self.dataset[hf_subset]["train"]["text"])
        else:
            text = self.dataset[split]["text"]
            label = self.dataset[split]["label"]
            if split != "train":
                train_text = self.dataset["train"]["text"]

        text_len = [len(t) for t in text]
        total_text_len = sum(text_len)
        label_len = [len(l) for l in label]
        total_label_len = sum(label_len)
        total_labels = []
        for l in label:
            total_labels.extend(l if len(l) > 0 else [None])
        label_count = Counter(total_labels)
        num_texts_in_train = (
            len(set(text) & set(train_text)) if split != "train" else None
        )
        return MultilabelClassificationDescriptiveStatistics(
            num_samples=len(text),
            number_of_characters=total_text_len,
            number_texts_in_train=num_texts_in_train,
            min_text_length=min(text_len),
            average_text_length=total_text_len / len(text),
            max_text_length=max(text_len),
            unique_texts=len(set(text)),
            min_labels_per_text=min(label_len),
            average_label_per_text=total_label_len / len(label),
            max_labels_per_text=max(label_len),
            unique_labels=len(label_count),
            labels={
                str(label): {
                    "count": value,
                }
                for label, value in label_count.items()
            },
        )
