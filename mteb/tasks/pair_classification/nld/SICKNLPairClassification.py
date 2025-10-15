from mteb.abstasks.task_metadata import TaskMetadata
from mteb.abstasks.text.pair_classification import AbsTaskPairClassification


class SICKNLPairClassification(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="SICKNLPairClassification",
        dataset={
            "path": "clips/mteb-nl-sick",
            "revision": "main",
        },
        description="SICK-NL is a Dutch translation of SICK ",
        reference="https://aclanthology.org/2021.eacl-main.126/",
        type="PairClassification",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["nld-Latn"],
        main_score="max_ap",
        date=("2020-09-01", "2021-01-01"),
        domains=["Web", "Written"],
        task_subtypes=[],
        license="mit",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="machine-translated and verified",
        bibtex_citation=r"""
@inproceedings{wijnholds2021sick,
  author = {Wijnholds, Gijs and Moortgat, Michael},
  booktitle = {Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume},
  pages = {1474--1479},
  title = {SICK-NL: A Dataset for Dutch Natural Language Inference},
  year = {2021},
}
""",
    )

    def dataset_transform(self) -> None:
        _dataset = {}

        for split in self.dataset:
            self.dataset[split] = self.dataset[split].filter(
                lambda ex: ex["label"] != "neutral"
            )
            self.dataset[split] = self.dataset[split].map(
                lambda ex: {"labels": 1 if ex["label"] == "entailment" else 0}
            )

            _dataset[split] = [
                {
                    "sentence1": self.dataset[split]["sentence1"],
                    "sentence2": self.dataset[split]["sentence2"],
                    "labels": self.dataset[split]["labels"],
                }
            ]

        self.dataset = _dataset
