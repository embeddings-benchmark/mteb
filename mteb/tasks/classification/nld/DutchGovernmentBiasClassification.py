from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class DutchGovernmentBiasClassification(AbsTaskClassification):
    samples_per_label = 32
    metadata = TaskMetadata(
        name="DutchGovernmentBiasClassification",
        description="The Dutch Government Data for Bias Detection (DGDB) is a dataset sourced from the Dutch House of Representatives and annotated for bias by experts",
        reference="https://dl.acm.org/doi/pdf/10.1145/3696410.3714526",
        dataset={
            "path": "milenamileentje/Dutch-Government-Data-for-Bias-detection",
            "revision": "3c4f325ccedeefbd93e47540adbefe1b19a778e5",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        date=("2019-10-04", "2019-10-04"),
        eval_splits=["test"],
        eval_langs=["nld-Latn"],
        main_score="f1",
        domains=["Written", "Government"],
        task_subtypes=[],
        license="cc-by-nc-sa-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{de2025detecting,
  author = {de Swart, Milena and Den Hengst, Floris and Chen, Jieying},
  booktitle = {Proceedings of the ACM on Web Conference 2025},
  pages = {5034--5044},
  title = {Detecting Linguistic Bias in Government Documents Using Large language Models},
  year = {2025},
}
""",
    )
