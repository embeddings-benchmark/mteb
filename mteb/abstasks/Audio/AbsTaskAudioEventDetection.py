from __future__ import annotations

import itertools
import logging
from collections import Counter, defaultdict
from typing import Any

import numpy as np

from mteb.abstasks.TaskMetadata import HFSubset

from ...encoder_interface import AudioEncoder
from ...evaluation.evaluators import EventDetector, onset_f_measure
from ..AbsTask import AbsTask, ScoresDict
from ..TaskMetadata import DescriptiveStatistics

logger = logging.getLogger(__name__)


class AudioEventDetectionDescriptiveStatistics(DescriptiveStatistics):
    """Descriptive statistics for AudioEventDetection task"""

    def __init__(
        self,
        num_samples: int,
        total_duration: float,
        min_duration: float,
        avg_duration: float,
        max_duration: float,
        sample_rate: int,
        min_events_per_sample: int,
        avg_events_per_sample: float,
        max_events_per_sample: int,
        unique_event_labels: int,
        event_label_distribution: dict[str, int],
        min_event_duration: float,
        avg_event_duration: float,
        max_event_duration: float,
    ):
        self.num_samples = num_samples
        self.total_duration = total_duration
        self.min_duration = min_duration
        self.avg_duration = avg_duration
        self.max_duration = max_duration
        self.sample_rate = sample_rate
        self.min_events_per_sample = min_events_per_sample
        self.avg_events_per_sample = avg_events_per_sample
        self.max_events_per_sample = max_events_per_sample
        self.unique_event_labels = unique_event_labels
        self.event_label_distribution = event_label_distribution
        self.min_event_duration = min_event_duration
        self.avg_event_duration = avg_event_duration
        self.max_event_duration = max_event_duration


class AbstractTaskAudioEventsDetection(AbsTask):
    """Abstract task for audio event detection with onset F-measure evaluation
    self.load_data() must generate a huggingface dataset with a split matching self.metadata_dict["eval_splits"], and assign it to self.dataset. It must contain the following columns:
        audio: List[datasets.Audio]
        events: list[list[dict[str, Any]]] -> List of samples, each sample is a list of events. Each event is a dict with keys: "label", "start", "end"

    Attributes:
       samples_per_label: Number of samples to use pr. label. These samples are embedded and a classifier is fit using the labels and samples.
    """

    audio_column_name: str = "audio"
    event_column_name: str = "events"
    samples_per_label: int = 8
    n_experiments: int = 10
    batch_size: int = 32
    train_split: str = "train"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _add_main_score(self, scores):
        scores["main_score"] = scores[self.metadata.main_score]

    def _calculate_metrics_from_split(
        self, split: str, hf_subset: str | None = None, compute_overall: bool = False
    ) -> AudioEventDetectionDescriptiveStatistics:
        if hf_subset:
            audio = self.dataset[hf_subset][split][self.audio_column_name]
            events = self.dataset[hf_subset][split][self.event_column_name]
        elif compute_overall:
            audio = []
            events = []
            for hf_subset in self.metadata.eval_langs:
                audio.extend(self.dataset[hf_subset][split][self.audio_column_name])
                events.extend(self.dataset[hf_subset][split][self.event_column_name])
        else:
            audio = self.dataset[split][self.audio_column_name]
            events = self.dataset[split][self.event_column_name]

        durations = [
            len(arr) / sr for arr, sr in zip(audio["array"], audio["sample_rate"])
        ]

        event_counts = [len(e) for e in events]

        all_event_labels = []
        event_durations = []
        for sample_events in events:
            for event in sample_events:
                all_event_labels.append(event["label"])
                event_durations.append(event["end"] - event["start"])

        return AudioEventDetectionDescriptiveStatistics(
            num_samples=len(events),
            total_duration=sum(durations),
            min_duration=min(durations) if durations else 0,
            avg_duration=np.mean(durations) if durations else 0,
            max_duration=max(durations) if durations else 0,
            sample_rate=audio["sample_rate"][0] if audio and len(audio) > 0 else 0,
            min_events_per_sample=min(event_counts) if event_counts else 0,
            avg_events_per_sample=np.mean(event_counts) if event_counts else 0,
            max_events_per_sample=max(event_counts) if event_counts else 0,
            unique_event_labels=len(set(all_event_labels)),
            event_label_distribution=dict(Counter(all_event_labels)),
            min_event_duration=min(event_durations) if event_durations else 0,
            avg_event_duration=np.mean(event_durations) if event_durations else 0,
            max_event_duration=max(event_durations) if event_durations else 0,
        )

    def evaluate(
        self,
        model: AudioEncoder,
        eval_split: str = "test",
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
        *,
        encode_kwargs: dict[str, Any] = {},
        **kwargs: Any,
    ) -> ScoresDict:
        train_split = dataset[self.train_split]
        eval_dataset = dataset[eval_split]

        params = {
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
        unique_audio = [train_split[self.audio_column_name][i] for i in unique_indices]
        unique_train_embeddings = model.get_audio_embeddings_per_frame(
            unique_audio, **kwargs
        )
        train_embeddings_dict = dict(zip(unique_indices, unique_train_embeddings))

        test_audio = eval_dataset[self.audio_column_name]
        test_events = eval_dataset[self.event_column_name]

        max_test_samples = 2000
        if len(test_audio) > max_test_samples:
            test_indices = np.random.choice(
                len(test_audio), size=max_test_samples, replace=False
            )
            test_audio = [test_audio[i] for i in test_indices]
            test_events = [test_events[i] for i in test_indices]

        X_test = model.get_audio_embeddings_per_frame(test_audio, **kwargs)

        all_scores = []
        for exp_idx, sample_indices in enumerate(train_samples):
            logger.info(
                "=" * 10 + f" Experiment {exp_idx + 1}/{self.n_experiments} " + "=" * 10
            )
            X_train = np.stack([train_embeddings_dict[i] for i in sample_indices])
            y_train = [train_split[self.event_column_name][i] for i in sample_indices]

            event_detector = EventDetector(seed=42 + exp_idx)
            event_detector.fit(X_train, y_train)

            pred_events_list = event_detector.predict(X_test)

            pred_events_dict = self._format_events_for_evaluation(pred_events_list)
            test_events_dict = self._format_events_for_evaluation(
                test_events, [f"sample_{i}" for i in range(len(test_events))]
            )

            scores_200ms = onset_f_measure(
                pred_events_dict, test_events_dict, t_collar=0.2
            )
            scores_50ms = onset_f_measure(
                pred_events_dict, test_events_dict, t_collar=0.05
            )

            scores_exp = {}
            for name, value in scores_200ms:
                scores_exp[f"onset_200ms_{name}"] = value

            for name, value in scores_50ms:
                scores_exp[f"onset_50ms_{name}"] = value

            all_scores.append(scores_exp)

        avg_scores = {}
        for key in all_scores[0].keys():
            avg_scores[key] = np.mean([s[key] for s in all_scores])

        avg_scores["main_score"] = avg_scores["onset_200ms_f_measure"]

        return avg_scores

    def _format_events_for_evaluation(
        self,
        events_list: list[list[dict[str, Any]]],
        file_ids: list[str] | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        result = defaultdict(list)

        for i, sample_events in enumerate(events_list):
            file_id = file_ids[i] if file_ids else f"sample_{i}"

            for event in sample_events:
                event_file_id = event.get("file_id", file_id)

                result[event_file_id].append(
                    {
                        "label": event["label"],
                        "start": event["start"],
                        "end": event["end"],
                    }
                )

        return dict(result)

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
