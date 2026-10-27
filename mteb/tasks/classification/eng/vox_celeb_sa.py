from typing import Any

from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class VoxCelebSA(AbsTaskClassification):
    metadata = TaskMetadata(
        name="VoxCelebSA",
        description="VoxCeleb dataset augmented for Sentiment Analysis task",
        reference="https://huggingface.co/datasets/DynamicSuperb/Sentiment_Analysis_SLUE-VoxCeleb",
        dataset={
            "path": "mteb/voxceleb-sentiment",
            "revision": "17fb326752c26b58491de360f0a6a152c9bfe19d",
        },
        type="AudioClassification",
        category="a2c",
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2024-06-27", "2024-06-28"),
        domains=["Spoken"],
        task_subtypes=["Sentiment Analysis"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["audio"],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{shon2022sluenewbenchmarktasks,
  author = {Shon, Suwon and Pasad, Ankita and Wu, Felix and Brusco, Pablo and Artzi, Yoav and Livescu, Karen and Han, Kyu J.},
  booktitle = {ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  doi = {10.1109/icassp43922.2022.9746137},
  month = {May},
  pages = {7927–7931},
  publisher = {IEEE},
  title = {SLUE: New Benchmark Tasks For Spoken Language Understanding Evaluation on Natural Speech},
  url = {https://doi.org/10.1109/icassp43922.2022.9746137},
  year = {2022},
}
""",
    )

    input_column_name: str = "audio"
    label_column_name: str = "label"

    is_cross_validation: bool = True

    def dataset_transform(self, **kwargs: Any):
        # remove disagreement data
        self.dataset = self.dataset.filter(lambda x: x["label"] != "Disagreement")
        self.dataset["train"] = self.dataset.pop("test")
