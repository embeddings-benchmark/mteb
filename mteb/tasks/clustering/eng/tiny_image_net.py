from mteb.abstasks.clustering_legacy import AbsTaskClusteringLegacy
from mteb.abstasks.task_metadata import TaskMetadata


class TinyImageNet(AbsTaskClusteringLegacy):
    metadata = TaskMetadata(
        name="TinyImageNetClustering",
        description="Clustering over 200 classes.",
        reference="https://huggingface.co/datasets/zh-plus/tiny-imagenet/viewer/default/valid",
        dataset={
            "path": "mteb/tiny-imagenet",
            "revision": "b979e28a93d43cb0d9a3a6f4a2fb38c33960cb40",
        },
        type="ImageClustering",
        category="i2c",
        eval_splits=["valid"],
        eval_langs=["eng-Latn"],
        main_score="nmi",
        date=(
            "2012-01-01",
            "2015-12-31",
        ),  # Estimated range for the collection of reviews
        domains=["Reviews"],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        modalities=["image"],
        sample_creation="found",
        bibtex_citation="""""",
    )
    input_column_name = "image"
    label_column_name = "label"
