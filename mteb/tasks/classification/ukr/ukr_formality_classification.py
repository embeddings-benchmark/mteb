from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class UkrFormalityClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="UkrFormalityClassification",
        description="This dataset contains Ukrainian Formality Classification dataset obtained by trainslating English GYAFC data. English data source: https://aclanthology.org/N18-1012/ Translation into Ukrainian language using model: https://huggingface.co/facebook/nllb-200-distilled-600M Additionally, the dataset was balanced, with labels: 0 - informal, 1 - formal.",
        dataset={
            "path": "ukr-detect/ukr-formality-dataset-translated-gyafc",
            "revision": "671d1e6bbf45a74ef21af351fd4ef7b32b7856f8",
        },
        reference="https://huggingface.co/datasets/ukr-detect/ukr-formality-dataset-translated-gyafc",
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["train", "test"],
        eval_langs=["ukr-Cyrl"],
        main_score="accuracy",
        date=("2018-04-11", "2018-06-20"),
        domains=["News", "Written"],
        task_subtypes=["Topic classification"],
        license="openrail++",
        annotations_creators="derived",
        dialect=[],
        sample_creation="machine-translated",
        bibtex_citation=r"""
@inproceedings{rao-tetreault-2018-dear,
  author = {Rao, Sudha  and
Tetreault, Joel},
  booktitle = {Proceedings of the 2018 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers)},
  month = jun,
  publisher = {Association for Computational Linguistics},
  title = {Dear Sir or Madam, May {I} Introduce the {GYAFC} Dataset: Corpus, Benchmarks and Metrics for Formality Style Transfer},
  url = {https://aclanthology.org/N18-1012},
  year = {2018},
}
""",
        superseded_by="UkrFormalityClassification.v2",
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_column("labels", "label")
        self.dataset = self.dataset.class_encode_column("label")
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["train", "test"]
        )


class UkrFormalityClassificationV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="UkrFormalityClassification.v2",
        description="This dataset contains Ukrainian Formality Classification dataset obtained by trainslating English GYAFC data. English data source: https://aclanthology.org/N18-1012/ Translation into Ukrainian language using model: https://huggingface.co/facebook/nllb-200-distilled-600M Additionally, the dataset was balanced, with labels: 0 - informal, 1 - formal. This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)",
        dataset={
            "path": "mteb/ukr_formality",
            "revision": "e0b2dfa57d505f207deb571e58b0bd0b81180bd4",
        },
        reference="https://huggingface.co/datasets/ukr-detect/ukr-formality-dataset-translated-gyafc",
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["ukr-Cyrl"],
        main_score="accuracy",
        date=("2018-04-11", "2018-06-20"),
        domains=["News", "Written"],
        task_subtypes=["Topic classification"],
        license="openrail++",
        annotations_creators="derived",
        dialect=[],
        sample_creation="machine-translated",
        bibtex_citation=r"""
@inproceedings{rao-tetreault-2018-dear,
  author = {Rao, Sudha  and
Tetreault, Joel},
  booktitle = {Proceedings of the 2018 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers)},
  month = jun,
  publisher = {Association for Computational Linguistics},
  title = {Dear Sir or Madam, May {I} Introduce the {GYAFC} Dataset: Corpus, Benchmarks and Metrics for Formality Style Transfer},
  url = {https://aclanthology.org/N18-1012},
  year = {2018},
}
""",
        adapted_from=["UkrFormalityClassification"],
    )

    def dataset_transform(self):
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["train", "test"]
        )
