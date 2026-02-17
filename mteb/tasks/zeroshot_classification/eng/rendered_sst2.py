from mteb.abstasks.task_metadata import TaskMetadata
from mteb.abstasks.zeroshot_classification import (
    AbsTaskZeroShotClassification,
)


class RenderedSST2(AbsTaskZeroShotClassification):
    metadata = TaskMetadata(
        name="RenderedSST2",
        description="RenderedSST2.",
        reference="https://huggingface.co/datasets/clip-benchmark/wds_renderedsst2",
        dataset={
            "path": "mteb/wds_renderedsst2",
            "revision": "c10537bc389e9741a27f7b14767bd42f2b77476b",
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

    # Override default column names in the subclass
    input_column_name: str = "png"
    label_column_name: str = "cls"

    def get_candidate_labels(self) -> list[str]:
        return ["a negative review of a movie", "a positive review of a movie"]
