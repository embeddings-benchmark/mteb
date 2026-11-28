from typing import Any

from mteb.abstasks import AbsTaskClustering
from mteb.abstasks.task_metadata import TaskMetadata


class VoxCelebClustering(AbsTaskClustering):
    label_column_name: str = "label_id"
    metadata = TaskMetadata(
        name="VoxCelebClustering",
        description="Clustering task based on the VoxCeleb dataset for sentiment analysis, clustering by positive/negative sentiment.",
        reference="https://huggingface.co/datasets/DynamicSuperb/Sentiment_Analysis_SLUE-VoxCeleb",
        dataset={
            "path": "mteb/Sentiment_Analysis_SLUE-VoxCeleb",
            "revision": "3155df63aedc31a112c284097fdd7c0316a24ee8",
        },
        type="AudioClustering",
        category="a2a",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="v_measure",
        date=("2024-06-27", "2024-06-28"),
        domains=["Spoken", "Speech"],
        task_subtypes=["Sentiment Clustering"],
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
    max_fraction_of_documents_to_embed = None
    input_column_name: str = "audio"

    def dataset_transform(self, **kwargs: Any) -> None:
        ds = self.dataset
        # Remove 'Disagreement' samples and '<mixed>' samples
        ds = ds.filter(lambda x: x["label"] not in ["Disagreement", "<mixed>"])  # noqa: PLR6201
        # Map string sentiment labels to numeric IDs
        label2id = {"Negative": 0, "Neutral": 1, "Positive": 2}

        def add_label_id(example: dict[str, Any]) -> dict[str, Any]:
            example["label_id"] = label2id[example["label"]]
            return example

        ds = ds.map(add_label_id)
        self.dataset = ds
        self.label_column_name = "label_id"

        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["test"], label=self.label_column_name
        )
