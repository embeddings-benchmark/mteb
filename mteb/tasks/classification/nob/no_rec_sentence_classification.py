import random
from typing import Any

from datasets import Dataset, DatasetDict, concatenate_datasets

from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class NoRecSentenceClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="NoRecSentenceClassification",
        description="Sentence-level polarity classification of Norwegian sentences from professional reviews across mixed domains (literature, movies, games, restaurants, music, theater and products). The sentence labels are aggregated from the fine-grained annotations of NoReC_fine. This task uses the binary version, where each sentence is positive or negative.",
        reference="https://huggingface.co/datasets/ltg/norec_sentence",
        dataset={
            "path": "ltg/norec_sentence",
            "name": "binary",
            "revision": "5326d626b35e6da3a503e5303e39adf11250ba3d",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["nob-Latn"],
        main_score="accuracy",
        date=("1998-01-01", "2019-12-31"),
        domains=["Written", "Reviews"],
        task_subtypes=["Sentiment/Hate speech"],
        license="cc-by-nc-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{kutuzov-etal-2021-large,
  address = {Reykjavik, Iceland (Online)},
  author = {Kutuzov, Andrey  and
Barnes, Jeremy  and
Velldal, Erik  and
{\O}vrelid, Lilja  and
Oepen, Stephan},
  booktitle = {Proceedings of the 23rd Nordic Conference on Computational Linguistics (NoDaLiDa)},
  title = {Large-Scale Contextualised Language Modelling for {N}orwegian},
  url = {https://aclanthology.org/2021.nodalida-main.4},
  year = {2021},
}

@inproceedings{ovrelid-etal-2020-fine,
  address = {Marseille, France},
  author = {{\O}vrelid, Lilja  and
M{\ae}hlum, Petter  and
Barnes, Jeremy  and
Velldal, Erik},
  booktitle = {Proceedings of the Twelfth Language Resources and Evaluation Conference},
  title = {A Fine-grained Sentiment Dataset for {N}orwegian},
  url = {https://aclanthology.org/2020.lrec-1.618},
  year = {2020},
}
""",
        prompt="Classify Norwegian sentences from reviews by sentiment",
    )

    label_column_name = "sentiment"
    input_column_name = "review"
    samples_per_label = 32

    def dataset_transform(self, num_proc: int | None = None, **kwargs: Any) -> None:
        # the validation split is evaluated as part of the test split, since the
        # balanced test split is otherwise small
        held_out = concatenate_datasets(
            [self.dataset["validation"], self.dataset["test"]]
        )

        def _texts(split: Dataset) -> dict[str, int]:
            # keep each text once, and drop texts with conflicting labels
            labels: dict[str, set[int]] = {}
            for text, label in zip(
                split[self.input_column_name],
                split[self.label_column_name],
                strict=True,
            ):
                labels.setdefault(text, set()).add(label)
            return {
                text: next(iter(text_labels))
                for text, text_labels in labels.items()
                if len(text_labels) == 1
            }

        test = _texts(held_out)
        train = {
            text: label
            for text, label in _texts(self.dataset["train"]).items()
            if text not in test
        }

        # balance the classes and subsample to at most 2048 per split
        rng = random.Random(self.seed)
        splits = {}
        for name, texts in {"train": train, "test": test}.items():
            by_label: dict[int, list[str]] = {}
            for text, label in texts.items():
                by_label.setdefault(label, []).append(text)
            n_per_label = min(
                2048 // len(by_label), *(len(t) for t in by_label.values())
            )
            selected_texts, selected_labels = [], []
            for label, label_texts in by_label.items():
                sampled = rng.sample(label_texts, k=n_per_label)
                selected_texts += sampled
                selected_labels += [label] * n_per_label
            splits[name] = Dataset.from_dict(
                {
                    self.input_column_name: selected_texts,
                    self.label_column_name: selected_labels,
                }
            )

        self.dataset = DatasetDict(splits)
