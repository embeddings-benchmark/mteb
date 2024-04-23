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
        date=None,
        form=None,
        domains=["Encyclopaedic"],
        task_subtypes=None,
        license="cc-by-sa-3.0",
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
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
        avg_character_length=None,
    )

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = dict(self.metadata)
        metadata_dict["n_experiments"] = 10
        metadata_dict["samples_per_label"] = 4
        return metadata_dict

    def dataset_transform(self):
        self.dataset = self.dataset.map(
            lambda examples: {"text": examples["content"]}, remove_columns=["title"]
        )
        self.dataset = self.dataset["test"].train_test_split(
            test_size=2048, train_size=2048, seed=42, stratify_by_column="label"
        )
