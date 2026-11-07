from __future__ import annotations

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

CITATION = r"""
@inproceedings{dong2023simmmdg,
  title = {Sim{MMDG}: A Simple and Effective Framework for Multi-modal Domain Generalization},
  author = {Dong, Hao and Nejjar, Ismail and Sun, Han and Chatzi, Eleni and Fink, Olga},
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2023},
}
"""


class HumanAnimalCartoonZeroShotClassification(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="HumanAnimalCartoonZeroShot",
        description=(
            "Human-Animal-Cartoon (HAC) is a multi-domain action recognition "
            "dataset containing clips of humans, animals, and cartoon figures. "
            "The zero-shot task matches each video clip to a text prompt for "
            "one of seven action labels. "
            "Concatenated the three HAC test-only domain CSVs, capped at 32 per class across 7 action classes (~644 examples)."
        ),
        reference="https://arxiv.org/abs/2310.19795",
        dataset={
            "path": "mteb/HumanAnimalCartoonZeroShot",
            "revision": "689d291cacea3ec66ab397ba7a5d82cc910cc779",
        },
        type="VideoZeroshotClassification",
        category="v2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2023-10-30", "2023-10-30"),
        domains=["Activity", "Animation", "Nature", "Web"],
        task_subtypes=["Activity recognition"],
        license="apache-2.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video", "text"],
        sample_creation="found",
        bibtex_citation=CITATION,
        is_beta=True,
    )


class HumanAnimalCartoonVAZeroShotClassification(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="HumanAnimalCartoonVAZeroShot",
        description=(
            "Human-Animal-Cartoon (HAC) is a multi-domain action recognition "
            "dataset containing clips of humans, animals, and cartoon figures. "
            "The zero-shot task matches each (video, audio) pair to a text "
            "prompt for one of seven action labels. "
            "Concatenated the three HAC test-only domain CSVs, capped at 32 per class across 7 action classes (~644 examples)."
        ),
        reference="https://arxiv.org/abs/2310.19795",
        dataset={
            "path": "mteb/HumanAnimalCartoonVAZeroShot",
            "revision": "c700d9c84128c480bc596bc775c7ff0dce7f6654",
        },
        type="VideoZeroshotClassification",
        category="va2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2023-10-30", "2023-10-30"),
        domains=["Activity", "Animation", "Nature", "Web"],
        task_subtypes=["Activity recognition"],
        license="apache-2.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video", "audio", "text"],
        sample_creation="found",
        bibtex_citation=CITATION,
        is_beta=True,
    )
