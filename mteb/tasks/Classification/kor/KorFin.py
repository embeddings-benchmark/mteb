from __future__ import annotations

from mteb.abstasks.AbsTaskAnyClassification import AbsTaskAnyClassification

# type: ignore
from mteb.abstasks.task_metadata import TaskMetadata  # type: ignore


class KorFin(AbsTaskAnyClassification):
    metadata = TaskMetadata.model_construct(
        name="KorFin",
        dataset={
            "path": "amphora/korfin-asc",
            "revision": "07cc4a29341ef26e8614ae1139847f4d4888727d",
        },
        description="The KorFin-ASC is an extension of KorFin-ABSA, which is a financial sentiment analysis dataset including 8818 samples with (aspect, polarity) pairs annotated. The samples were collected from KLUE-TC and analyst reports from Naver Finance.",
        reference="https://huggingface.co/datasets/amphora/korfin-asc",
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs=["kor-Hang"],
        main_score="accuracy",
        date=(
            "2022-01-01",
            "2022-12-31",
        ),  # Assumed date based on the citations in the paper
        domains=["News", "Written", "Financial"],
        task_subtypes=["Sentiment/Hate speech"],
        license="cc-by-sa-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{son2023removing,
  author = {Son, Guijin and Lee, Hanwool and Kang, Nahyeon and Hahm, Moonjeong},
  journal = {arXiv preprint arXiv:2301.03136},
  title = {Removing Non-Stationary Knowledge From Pre-Trained Language Models for Entity-Level Sentiment Classification in Finance},
  year = {2023},
}
""",
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_columns(
            {"SRC": "text", "SENTIMENT": "label"}
        ).remove_columns(["SID", "TYPE", "ASPECT"])
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=self.metadata.eval_splits
        )
