from mteb.abstasks.pair_classification import AbsTaskPairClassification
from mteb.abstasks.task_metadata import TaskMetadata


class DisCoTexPairClassification(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="DisCoTexPairClassification",
        description="The DisCoTEX dataset aims at assessing discourse coherence in Italian texts. This dataset focuses on Italian real-world texts and provides resources to model coherence in natural language.",
        reference="https://github.com/davidecolla/DisCoTex",
        dataset={
            "path": "MattiaSangermano/DisCoTex-last-sentence",
            "revision": "ab9ea43f8e54c8b24b12cd1b77d6eb462385a30b",
        },
        type="PairClassification",
        category="t2t",
        modalities=["text"],
        date=("2023-01-01", "2023-12-31"),
        eval_splits=["test"],
        eval_langs=["ita-Latn"],
        main_score="max_ap",
        domains=["Social", "Written"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{brunato2023discotex,
  author = {Brunato, Dominique and Colla, Davide and Dell'Orletta, Felice and Dini, Irene and Radicioni, Daniele Paolo and Ravelli, Andrea Amelio and others},
  booktitle = {CEUR WORKSHOP PROCEEDINGS},
  organization = {CEUR},
  pages = {1--8},
  title = {DisCoTex at EVALITA 2023: overview of the assessing discourse coherence in Italian texts task},
  volume = {3473},
  year = {2023},
}
""",
    )

    def dataset_transform(self):
        self.dataset = self.dataset.remove_columns(["id", "source"])
        self.dataset = self.dataset.map(
            lambda x: {
                "prompt": [x["prompt"]],
                "target": [x["target"]],
                "class": [x["class"]],
            },
            batched=True,
            batch_size=len(self.dataset["train"]),
        )
        self.dataset = self.dataset.rename_column("prompt", "sentence1")
        self.dataset = self.dataset.rename_column("target", "sentence2")
        self.dataset = self.dataset.rename_column("class", "labels")
