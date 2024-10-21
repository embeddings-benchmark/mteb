from __future__ import annotations

from mteb.abstasks.Image.AbsTaskImageClustering import AbsTaskImageClustering
from mteb.abstasks.TaskMetadata import TaskMetadata


class TinyImageNet(AbsTaskImageClustering):
    metadata = TaskMetadata(
        name="TinyImageNetClustering",
        description="Clustering over 200 classes.",
        reference="https://huggingface.co/datasets/zh-plus/tiny-imagenet/viewer/default/valid",
        dataset={
            "path": "zh-plus/tiny-imagenet",
            "revision": "5a77092c28e51558c5586e9c5eb71a7e17a5e43f",
        },
        type="Clustering",
        category="s2s",
        eval_splits=["valid"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=(
            "2012-01-01",
            "2015-12-31",
        ),  # Estimated range for the collection of reviews
        domains=["Reviews"],
        task_subtypes=["Sentiment/Hate speech"],
        license="Not specified",
        annotations_creators="derived",
        dialect=[],
        modalities=["image"],
        sample_creation="found",
        bibtex_citation="""d""",
        descriptive_stats={
            "n_samples": {"valid": 10000},
            "avg_character_length": {"valid": 431.4},
        },
    )
