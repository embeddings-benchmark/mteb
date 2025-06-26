from __future__ import annotations

from mteb.abstasks.Audio.AbsTaskAudioClustering import AbsTaskAudioClustering
from mteb.abstasks.TaskMetadata import TaskMetadata


class VoxCelebClustering(AbsTaskAudioClustering):
    label_column_name: str = "label_id"
    metadata = TaskMetadata(
        name="VoxCelebClustering",
        description="Clustering task based on the VoxCeleb dataset for sentiment analysis, clustering by positive/negative sentiment.",
        reference="https://huggingface.co/datasets/DynamicSuperb/Sentiment_Analysis_SLUE-VoxCeleb",
        dataset={
            "path": "DynamicSuperb/Sentiment_Analysis_SLUE-VoxCeleb",
            "revision": "554ad4367e98b7c6f4d4d9756dc6bbdf345e042e",
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
@misc{shon2022sluenewbenchmarktasks,
  archiveprefix = {arXiv},
  author = {Suwon Shon and Ankita Pasad and Felix Wu and Pablo Brusco and Yoav Artzi and Karen Livescu and Kyu J. Han},
  eprint = {2111.10367},
  primaryclass = {cs.CL},
  title = {SLUE: New Benchmark Tasks for Spoken Language Understanding Evaluation on Natural Speech},
  url = {https://arxiv.org/abs/2111.10367},
  year = {2022},
}
""",
    )
    max_fraction_of_documents_to_embed = None

    def dataset_transform(self):
        ds = self.dataset
        # Remove 'Disagreement' samples and '<mixed>' samples
        ds = ds.filter(lambda x: x["label"] not in ["Disagreement", "<mixed>"])
        # Map string sentiment labels to numeric IDs
        label2id = {"Negative": 0, "Neutral": 1, "Positive": 2}

        def add_label_id(example):
            example["label_id"] = label2id[example["label"]]
            return example

        ds = ds.map(add_label_id)
        self.dataset = ds
        self.label_column_name = "label_id"

        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["test"], label=self.label_column_name
        )
