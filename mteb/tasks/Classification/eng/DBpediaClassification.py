from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskClassification


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
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2022-01-25", "2022-01-25"),
        form=["written"],
        domains=["Encyclopaedic"],
        task_subtypes=["Topic classification"],
        license="cc-by-sa-3.0",
        socioeconomic_status="low",
        annotations_creators="derived",
        dialect=[],
        text_creation="found",
        bibtex_citation="""
            @inproceedings{NIPS2015_250cf8b5,
            author = {Zhang, Xiang and Zhao, Junbo and LeCun, Yann},
            booktitle = {Advances in Neural Information Processing Systems},
            editor = {C. Cortes and N. Lawrence and D. Lee and M. Sugiyama and R. Garnett},
            pages = {},
            publisher = {Curran Associates, Inc.},
            title = {Character-level Convolutional Networks for Text Classification},
            url = {https://proceedings.neurips.cc/paper_files/paper/2015/file/250cf8b51c773f3f8dc8b4be867a9a02-Paper.pdf},
            volume = {28},
            year = {2015}
            }
        """,
        n_samples={"test": 70000},
        avg_character_length={"test": 281.40},
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_column("content", "text")
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["train", "test"]
        )
