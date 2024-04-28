from __future__ import annotations

from mteb.abstasks import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class UkrFormalityClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="UkrFormalityClassification",
        description="""
        This dataset contains Ukrainian Formality Classification dataset obtained by
        trainslating English GYAFC data.
        English data source: https://aclanthology.org/N18-1012/
        Translation into Ukrainian language using model: https://huggingface.co/facebook/nllb-200-distilled-600M
        Additionally, the dataset was balanced, witha labels: 0 - informal, 1 - formal.
        """,
        dataset={
            "path": "ukr-detect/ukr-formality-dataset-translated-gyafc",
            "revision": "671d1e6bbf45a74ef21af351fd4ef7b32b7856f8",
        },
        reference="https://huggingface.co/datasets/ukr-detect/ukr-formality-dataset-translated-gyafc",
        type="Classification",
        category="s2s",
        eval_splits=["train", "test"],
        eval_langs=["ukr-Cyrl"],
        main_score="accuracy",
        date=("2018-04-11", "2018-06-20"),
        form=["written"],
        domains=["News"],
        task_subtypes=["Topic classification"],
        license="openrail++",
        socioeconomic_status="mixed",
        annotations_creators="derived",
        dialect=[],
        text_creation="machine-translated",
        bibtex_citation="""@inproceedings{rao-tetreault-2018-dear,
        title = "Dear Sir or Madam, May {I} Introduce the {GYAFC} Dataset: Corpus, Benchmarks and Metrics for Formality Style Transfer",
        author = "Rao, Sudha  and
        Tetreault, Joel",
        booktitle = "Proceedings of the 2018 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers)",
        month = jun,
        year = "2018",
        publisher = "Association for Computational Linguistics",
        url = "https://aclanthology.org/N18-1012",
        }""",
        n_samples={"train": 2048, "test": 2048},
        avg_character_length={"train": 52.10, "test": 53.07},
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_column("labels", "label")
        self.dataset = self.dataset.class_encode_column("label")
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["train", "test"]
        )
