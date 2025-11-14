from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class DBpediaClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="DBpediaClassification",
        description="DBpedia14 is a dataset of English texts from Wikipedia articles, categorized into 14 non-overlapping classes based on their DBpedia ontology.",
        reference="https://arxiv.org/abs/1509.01626",
        dataset={
            "path": "fancyzhx/dbpedia_14",
            "revision": "9abd46cf7fc8b4c64290f26993c540b92aa145ac",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2022-01-25", "2022-01-25"),
        domains=["Encyclopaedic", "Written"],
        task_subtypes=["Topic classification"],
        license="cc-by-sa-3.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{NIPS2015_250cf8b5,
  author = {Zhang, Xiang and Zhao, Junbo and LeCun, Yann},
  booktitle = {Advances in Neural Information Processing Systems},
  editor = {C. Cortes and N. Lawrence and D. Lee and M. Sugiyama and R. Garnett},
  pages = {},
  publisher = {Curran Associates, Inc.},
  title = {Character-level Convolutional Networks for Text Classification},
  url = {https://proceedings.neurips.cc/paper_files/paper/2015/file/250cf8b51c773f3f8dc8b4be867a9a02-Paper.pdf},
  volume = {28},
  year = {2015},
}
""",
        superseded_by="DBpediaClassification.v2",
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_column("content", "text")
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["train", "test"]
        )


class DBpediaClassificationV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="DBpediaClassification.v2",
        description="DBpedia14 is a dataset of English texts from Wikipedia articles, categorized into 14 non-overlapping classes based on their DBpedia ontology. This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)",
        reference="https://arxiv.org/abs/1509.01626",
        dataset={
            "path": "mteb/d_bpedia",
            "revision": "e45aab5cbb44baba43d8a0640d809d2aa0a0a770",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2022-01-25", "2022-01-25"),
        domains=["Encyclopaedic", "Written"],
        task_subtypes=["Topic classification"],
        license="cc-by-sa-3.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{NIPS2015_250cf8b5,
  author = {Zhang, Xiang and Zhao, Junbo and LeCun, Yann},
  booktitle = {Advances in Neural Information Processing Systems},
  editor = {C. Cortes and N. Lawrence and D. Lee and M. Sugiyama and R. Garnett},
  pages = {},
  publisher = {Curran Associates, Inc.},
  title = {Character-level Convolutional Networks for Text Classification},
  url = {https://proceedings.neurips.cc/paper_files/paper/2015/file/250cf8b51c773f3f8dc8b4be867a9a02-Paper.pdf},
  volume = {28},
  year = {2015},
}
""",
        adapted_from=["DBpediaClassification"],
    )

    def dataset_transform(self):
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["train", "test"]
        )
