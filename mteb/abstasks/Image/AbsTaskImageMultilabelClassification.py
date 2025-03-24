from __future__ import annotations

import itertools
import logging
from collections import Counter, defaultdict
from typing import Any

import numpy as np
from sklearn.base import ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, label_ranking_average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from mteb.abstasks.TaskMetadata import HFSubset

from ...encoder_interface import Encoder
from ..AbsTask import AbsTask, ScoresDict
from ..TaskMetadata import DescriptiveStatistics

logger = logging.getLogger(__name__)


class ImageMultilabelClassificationDescriptiveStatistics(DescriptiveStatistics):
    """Descriptive statistics for ImageMultilabelClassification

    Attributes:
        num_samples: number of samples in the dataset.

        min_image_width: Minimum width of images
        average_image_width: Average width of images
        max_image_width: Maximum width of images

        min_image_height: Minimum height of images
        average_image_height: Average height of images
        max_image_height: Maximum height of images

        min_labels_per_sample: Minimum number of labels per sample
        average_label_per_sample: Average number of labels per sample
        max_labels_per_sample: Maximum number of labels per sample
        unique_labels: Number of unique labels
        labels: dict of label frequencies
    """

    num_samples: int

    min_image_width: float
    average_image_width: float
    max_image_width: float

    min_image_height: float
    average_image_height: float
    max_image_height: float

    min_labels_per_sample: int
    average_label_per_sample: float
    max_labels_per_sample: int
    unique_num_labels: int
    labels: dict[str, dict[str, int]]


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
    all_probs = []
    for estimator in classifier.estimators_:
        probs = estimator.predict_proba(embeddings_test)[:, 1]
        all_probs.append(probs)

    y_score = np.stack(all_probs, axis=1)  # shape: (n_samples, n_labels)
    lrap = label_ranking_average_precision_score(y_test, y_score)
    scores["lrap"] = lrap
    return scores


class AbsTaskImageMultilabelClassification(AbsTask):
    """Abstract class for image multioutput classification tasks
    The similarity is computed between pairs and the results are ranked.

    self.load_data() must generate a huggingface dataset with a split matching self.metadata_dict["eval_splits"], and assign it to self.dataset. It must contain the following columns:
        image: list[PIL.Image]
        labels: list[Hashable]
    """

    image_column_name: str = "image"
    label_column_name: str = "labels"

    classifier = MultiOutputClassifier(estimator=LogisticRegression())

    def __init__(
        self,
        n_experiments=None,
        samples_per_label=None,
        batch_size=32,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.batch_size = batch_size

        # Bootstrap parameters
        self.n_experiments = n_experiments or getattr(self, "n_experiments", 10)
        self.samples_per_label = samples_per_label or getattr(
            self, "samples_per_label", 8
        )
        # Run metadata validation by instantiating addressing the attribute
        # This is quite hacky. Ideally, this would be done in the constructor of
        # each concrete task, but then we have to duplicate the __init__ method's
        # interface.
        if hasattr(self, "metadata"):
            self.metadata

    def _add_main_score(self, scores):
        scores["main_score"] = scores[self.metadata.main_score]

    def _calculate_metrics_from_split(
        self, split: str, hf_subset: str | None = None, compute_overall: bool = False
    ) -> ImageMultilabelClassificationDescriptiveStatistics:
        if hf_subset:
            imgs = self.dataset[hf_subset][split][self.image_column_name]
            labels = self.dataset[hf_subset][split][self.label_column_name]
        elif compute_overall:
            imgs, labels = [], []
            for hf_subset in self.metadata.eval_langs:
                imgs.extend(self.dataset[hf_subset][split][self.image_column_name])
                labels.extend(self.dataset[hf_subset][split][self.label_column_name])
        else:
            imgs = self.dataset[split][self.image_column_name]
            labels = self.dataset[split][self.label_column_name]

        num_samples = len(labels)

        label_len = [len(l) for l in labels]
        total_label_len = sum(label_len)
        total_labels = []
        for l in labels:
            total_labels.extend(l if len(l) > 0 else [None])
        label_count = Counter(total_labels)

        img_widths, img_heights = [], []
        for img in imgs:
            width, height = img.size
            img_heights.append(height)
            img_widths.append(width)

        return ImageMultilabelClassificationDescriptiveStatistics(
            num_samples=num_samples,
            min_image_width=min(img_widths),
            average_image_width=sum(img_widths) / len(img_widths),
            max_image_width=max(img_widths),
            min_image_height=min(img_heights),
            average_image_height=sum(img_heights) / len(img_heights),
            max_image_height=max(img_heights),
            min_labels_per_sample=min(label_len),
            average_label_per_sample=total_label_len / len(labels),
            max_labels_per_sample=max(label_len),
            unique_num_labels=len(label_count),
            labels={
                str(label): {"count": count} for label, count in label_count.items()
            },
        )

    def evaluate(
        self,
        model: Encoder,
        eval_split: str = "test",
        train_split: str = "train",
        *,
        encode_kwargs: dict[str, Any] = {},
        **kwargs: Any,
    ) -> dict[HFSubset, ScoresDict]:
        if not self.data_loaded:
            self.load_data()

        scores = {}
        hf_subsets = list(self.dataset) if self.is_multilingual else ["default"]

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
                train_split[self.label_column_name], self.samples_per_label, None
            )
            train_samples.append(sample_indices)
        # Encode all unique images at the indices
        unique_train_indices = list(set(itertools.chain.from_iterable(train_samples)))
        unique_train_images = train_split.select(unique_train_indices)[
            self.image_column_name
        ]

        _unique_train_embeddings = model.get_image_embeddings(
            unique_train_images,
            **encode_kwargs,
        )
        unique_train_embeddings = dict(
            zip(unique_train_indices, _unique_train_embeddings)
        )
        test_images = eval_split[self.image_column_name]
        binarizer = MultiLabelBinarizer()
        y_test = binarizer.fit_transform(eval_split[self.label_column_name])
        # Stratified subsampling of test set to 2000 examples.
        try:
            if len(test_images) > 2000:
                test_images, _, y_test, _ = train_test_split(
                    test_images, y_test, stratify=y_test, train_size=2000
                )
        except ValueError:
            logger.warning("Couldn't subsample, continuing with the entire test set.")

        X_test = model.get_image_embeddings(test_images, **encode_kwargs)
        for i_experiment, sample_indices in enumerate(train_samples):
            logger.info(
                "=" * 10
                + f" Experiment {i_experiment + 1}/{self.n_experiments} "
                + "=" * 10
            )
            X_train = np.stack([unique_train_embeddings[idx] for idx in sample_indices])
            y_train = train_split.select(sample_indices)[self.label_column_name]
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
