from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class BengaliDocumentClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="BengaliDocumentClassification",
        description="Dataset for News Classification, categorized with 13 domains.",
        reference="https://aclanthology.org/2023.eacl-main.4",
        dataset={
            "path": "dialect-ai/shironaam",
            "revision": "1c6e67433da618073295b7c90f1c55fa8e78f35c",
        },
        type="Classification",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["ben-Beng"],
        main_score="accuracy",
        date=("2022-05-01", "2023-05-01"),
        dialect=[],
        domains=["News", "Written"],
        task_subtypes=[],
        license="cc-by-nc-sa-4.0",
        annotations_creators="derived",
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{akash-etal-2023-shironaam,
  address = {Dubrovnik, Croatia},
  author = {Akash, Abu Ubaida  and
Nayeem, Mir Tafseer  and
Shohan, Faisal Tareque  and
Islam, Tanvir},
  booktitle = {Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics},
  month = may,
  pages = {52--67},
  publisher = {Association for Computational Linguistics},
  title = {Shironaam: {B}engali News Headline Generation using Auxiliary Information},
  url = {https://aclanthology.org/2023.eacl-main.4},
  year = {2023},
}
""",
    )

    def dataset_transform(self) -> None:
        self.dataset = self.dataset.rename_columns(
            {"article": "text", "category": "label"}
        )
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["test"]
        )
