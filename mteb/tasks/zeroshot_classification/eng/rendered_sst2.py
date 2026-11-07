from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class RenderedSST2(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="RenderedSST2",
        description="RenderedSST2.",
        reference="https://huggingface.co/datasets/clip-benchmark/wds_renderedsst2",
        dataset={
            "path": "mteb/RenderedSST2",
            "revision": "ea885506086327e7bb57ca654ecbf58ee47088c5",
        },
        type="ZeroShotClassification",
        category="i2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2016-01-01", "2016-12-31"),
        domains=["Reviews"],
        task_subtypes=[],
        license="mit",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="created",
        bibtex_citation="""""",
    )
