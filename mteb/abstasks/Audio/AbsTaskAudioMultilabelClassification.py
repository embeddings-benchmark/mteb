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

from ...encoder_interface import AudioEncoder
from ..AbsTask import AbsTask, ScoresDict
from ..TaskMetadata import DescriptiveStatistics

logger = logging.getLogger(__name__)


class AudioMultilabelClassificationDescriptiveStatistics(DescriptiveStatistics):
    """Descriptive statistics for AudioMultilabelClassification

    Attributes:
        num_samples: Number of audio samples
        total_duration: Total audio duration in seconds
        min_duration: Minimum audio clip duration
        avg_duration: Average audio clip duration
        max_duration: Maximum audio clip duration
        sample_rate: Audio sample rate
        min_labels_per_sample: Minimum number of labels per sample
        avg_labels_per_sample: Average number of labels
        max_labels_per_sample: Maximum number of labels
        unique_labels: Number of unique labels
        labels: dict of label frequencies
    """

    num_samples: int

    total_duration: float
    min_duration: float
    avg_duration: float
    max_duration: float
    sample_rate: int

    min_labels_per_sample: int
    avg_labels_per_sample: float
    max_labels_per_sample: int
    unique_labels: int
    labels: dict[str, dict[str, int]]


def evaluate_classifier(
    embeddings_train: np.ndarray,
    y_train: np.ndarray,
    embeddings_test: np.ndarray,
    y_test: np.ndarray,
    classifier: ClassifierMixin,
) -> dict[str, Any]:
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


class AbsTaskAudioMultilabelClassification(AbsTask):
    """Abstract class for audio multioutput classification tasks
    The similarity is computed between pairs and the results are ranked.

    self.load_data() must generate a huggingface dataset with a split matching self.metadata_dict["eval_splits"], and assign it to self.dataset. It must contain the following columns:
        audio: List[datasets.Audio]
        labels: list[Hashable]

    Attributes:
       samples_per_label: Number of samples to use pr. label. These samples are embedded and a classifier is fit using the labels and samples.
    """

    audio_column_name: str = "audio"
    label_column_name: str = "labels"
    samples_per_label: int = 8
    n_experiments: int = 10
    batch_size: int = 32
    train_split: str = "train"

    classifier = MultiOutputClassifier(estimator=LogisticRegression())

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _add_main_score(self, scores):
        scores["main_score"] = scores[self.metadata.main_score]

    def _calculate_metrics_from_split(
        self, split: str, hf_subset: str | None = None, compute_overall: bool = False
    ) -> AudioMultilabelClassificationDescriptiveStatistics:
        if hf_subset:
            audio = self.dataset[hf_subset][split][self.audio_column_name]
            labels = self.dataset[hf_subset][split][self.label_column_name]
        elif compute_overall:
            audio = []
            labels = []
            for hf_subset in self.metadata.eval_langs:
                audio.extend(self.dataset[hf_subset][split][self.audio_column_name])
                labels.extend(self.dataset[hf_subset][split][self.label_column_name])
        else:
            audio = self.dataset[split][self.audio_column_name]
            labels = self.dataset[split][self.label_column_name]

        durations = [
            len(arr) / sr for arr, sr in zip(audio["array"], audio["sample_rate"])
        ]
        label_counts = [len(l) for l in labels]

        flat_labels = [label for sublist in labels for label in sublist]

        return AudioMultilabelClassificationDescriptiveStatistics(
            num_samples=len(labels),
            total_duration=sum(durations),
            min_duration=min(durations),
            avg_duration=np.mean(durations),
            max_duration=max(durations),
            sample_rate=audio["sample_rate"][0],
            min_labels_per_sample=min(label_counts),
            avg_labels_per_sample=np.mean(label_counts),
            max_labels_per_sample=max(label_counts),
            unique_labels=len(set(flat_labels)),
            label_distribution=dict(Counter(flat_labels)),
        )

    def evaluate(
        self,
        model: AudioEncoder,
        eval_split: str = "test",
        train_split: str = "train",
        *,
        encode_kwargs: dict[str, Any] = {},
        **kwargs: Any,
    ) -> dict[HFSubset, ScoresDict]:
        if not self.data_loaded:
            self.load_data()

        scores = {}
        hf_subsets = self.hf_subsets

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
        model: AudioEncoder,
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

        # Bootstrap sample indices from training set for each experiment
        train_samples = []
        for _ in range(self.n_experiments):
            sample_indices, _ = self._undersample_data_indices(
                train_split[self.label_column_name], self.samples_per_label, None
            )
            train_samples.append(sample_indices)

        # Get unique training embeddings
        unique_indices = list(set(itertools.chain.from_iterable(train_samples)))
        unique_audio = train_split.select(unique_indices)[self.audio_column_name]
        _unique_embeddings = model.get_audio_embeddings(unique_audio, **kwargs)
        unique_train_embeddings = dict(zip(unique_indices, _unique_embeddings))
        test_audio = eval_split[self.audio_column_name]
        binarizer = MultiLabelBinarizer()
        y_test = binarizer.fit_transform(eval_split[self.label_column_name])

        # Subsample large test sets
        try:
            if len(test_audio) > 2000:
                test_audio, _, y_test, _ = train_test_split(
                    test_audio, y_test, train_size=2000, stratify=y_test
                )
        except ValueError:
            logger.warning("Could not stratify test set. Using all samples.")

        X_test = model.get_audio_embeddings(
            test_audio,
            **kwargs,
        )

        all_scores = []
        for exp_idx, sample_indices in enumerate(train_samples):
            logger.info(
                "=" * 10 + f" Experiment {exp_idx + 1}/{self.n_experiments} " + "=" * 10
            )
            X_train = np.stack([unique_train_embeddings[i] for i in sample_indices])
            y_train = binarizer.transform(
                train_split.select(sample_indices)[self.label_column_name]
            )
            scores_exp = evaluate_classifier(
                X_train, y_train, X_test, y_test, self.classifier
            )
            all_scores.append(scores_exp)

        avg_scores: dict[str, Any] = {
            k: np.mean([s[k] for s in all_scores]) for k in all_scores[0].keys()
        }
        avg_scores["scores_per_experiment"] = all_scores

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
