from pathlib import Path

from mteb.abstasks.any_zeroshot_classification import (
    AbsTaskAnyZeroShotClassification,
)
from mteb.abstasks.task_metadata import TaskMetadata


class Country211ZeroShotClassification(AbsTaskAnyZeroShotClassification):
    metadata = TaskMetadata(
        name="Country211ZeroShot",
        description="Classifying images of 211 countries.",
        reference="https://huggingface.co/datasets/clip-benchmark/wds_country211",
        dataset={
            "path": "clip-benchmark/wds_country211",
            "revision": "1699f138f0558342a1cbf99f7cf36b4361bb5ebc",
        },
        type="ZeroShotClassification",
        category="i2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=(
            "2020-01-01",
            "2021-02-26",
        ),  # Estimated range for the collection of reviews
        domains=["Scene"],
        task_subtypes=["Scene recognition"],
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["image", "text"],
        sample_creation="created",
        bibtex_citation=r"""
@article{radford2021learning,
  author = {Radford, Alec and Kim, Jong Wook and Hallacy, Chris and Ramesh, Aditya and Goh, Gabriel and Agarwal, Sandhini and Sastry, Girish and Askell, Amanda and Mishkin, Pamela and Clark, Jack and others},
  journal = {arXiv preprint arXiv:2103.00020},
  title = {Learning Transferable Visual Models From Natural Language Supervision},
  year = {2021},
}
""",
    )

    input_column_name: str = "jpg"
    label_column_name: str = "cls"

    def get_candidate_labels(self) -> list[str]:
        path = Path(__file__).parent / "templates" / "Country211_labels.txt"
        with path.open() as f:
            labels = f.readlines()

        return [f"a photo showing the country of {c}." for c in labels]
