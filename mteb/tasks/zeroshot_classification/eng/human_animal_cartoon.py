from __future__ import annotations

from mteb.abstasks.task_metadata import TaskMetadata
from mteb.abstasks.zeroshot_classification import AbsTaskZeroShotClassification

CITATION = r"""
@inproceedings{dong2023simmmdg,
  title = {Sim{MMDG}: A Simple and Effective Framework for Multi-modal Domain Generalization},
  author = {Dong, Hao and Nejjar, Ismail and Sun, Han and Chatzi, Eleni and Fink, Olga},
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2023},
}
"""

LABEL_PROMPTS = {
    "drinking": "a video of drinking",
    "eating": "a video of eating",
    "opening door": "a video of opening a door",
    "running": "a video of running",
    "sleeping": "a video of sleeping",
    "swimming": "a video of swimming",
    "watching tv": "a video of watching TV",
}


class HumanAnimalCartoonZeroShotClassification(AbsTaskZeroShotClassification):
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
            "path": "mteb/Human-Animal-Cartoon",
            "revision": "d38566c4bb055c7325314d3e46610792c2799c4b",
        },
        type="VideoZeroshotClassification",
        category="v2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2023-10-30", "2023-10-30"),
        domains=["Web", "Scene"],
        task_subtypes=["Activity recognition"],
        license="apache-2.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video", "text"],
        sample_creation="found",
        bibtex_citation=CITATION,
        is_beta=True,
    )

    input_column_name: str = "video"
    label_column_name: str = "action"

    def get_candidate_labels(self) -> list[str]:
        return [
            LABEL_PROMPTS.get(name, f"a video of {name}")
            for name in self.dataset["test"].features[self.label_column_name].names
        ]


class HumanAnimalCartoonVAZeroShotClassification(AbsTaskZeroShotClassification):
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
            "path": "mteb/Human-Animal-Cartoon",
            "revision": "d38566c4bb055c7325314d3e46610792c2799c4b",
        },
        type="VideoZeroshotClassification",
        category="va2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2023-10-30", "2023-10-30"),
        domains=["Web", "Scene"],
        task_subtypes=["Activity recognition"],
        license="apache-2.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video", "audio", "text"],
        sample_creation="found",
        bibtex_citation=CITATION,
        is_beta=True,
    )

    input_column_name = ("video", "audio")
    label_column_name: str = "action"

    def get_candidate_labels(self) -> list[str]:
        return [
            LABEL_PROMPTS.get(name, f"a video of {name}")
            for name in self.dataset["test"].features[self.label_column_name].names
        ]
