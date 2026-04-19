from __future__ import annotations

from mteb.abstasks import AbsTaskPairClassification
from mteb.abstasks.task_metadata import TaskMetadata

from ._video_pair_helpers import build_pair_dataset, generate_pairs


class MELDPairClassification(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="MELDPairClassification",
        description=(
            "Pair classification on the MELD dataset: "
            "determining whether two video clips from the Friends TV series "
            "express the same emotion from 7 categories "
            "(anger, disgust, fear, joy, neutral, sadness, surprise)."
        ),
        reference="https://affective-meld.github.io/",
        dataset={
            "path": "mteb/MELD",
            "revision": "6c0bf58845b1acccefc450b131c304378c1e38d5",
        },
        type="VideoPairClassification",
        category="v2v",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="max_ap",
        date=("2019-07-01", "2019-07-01"),
        domains=["Spoken"],
        task_subtypes=["Emotion classification"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video"],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{poria-etal-2019-meld,
  author = {Poria, Soujanya and Hazarika, Devamanyu and Majumder, Navonil and Naik, Gautam and Cambria, Erik and Mihalcea, Rada},
  booktitle = {Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics},
  doi = {10.18653/v1/P19-1050},
  month = {July},
  pages = {527--536},
  publisher = {Association for Computational Linguistics},
  title = {MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversations},
  url = {https://aclanthology.org/P19-1050/},
  year = {2019},
}
""",
        contributed_by="stef41",
        is_beta=True,
    )

    input1_column_name: str = "video1"
    input2_column_name: str = "video2"
    label_column_name: str = "label"

    def dataset_transform(self, num_proc: int | None = None, **kwargs) -> None:
        import random

        rng = random.Random(42)
        for split in self.metadata.eval_splits:
            ds = self.dataset[split]
            pairs = generate_pairs(ds["emotion"], rng)
            self.dataset[split] = build_pair_dataset(ds, pairs)
