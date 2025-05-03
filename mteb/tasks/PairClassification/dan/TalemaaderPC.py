from __future__ import annotations

from mteb.abstasks.AbsTaskPairClassification import AbsTaskPairClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class TalemaaderPC(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="TalemaaderPC",
        description="""\
The Danish Language and Literature Society has developed a dataset for evaluating language models in Danish.
The dataset contains a total of 1000 Danish idioms and fixed expressions with transferred meanings based on the Danish Dictionary's collection of fixed expressions with associated definitions.
For each of the 1000 idioms and fixed expressions, three false definitions have also been prepared.
The dataset can be used to test the performance of language models in identifying correct definitions for Danish idioms and fixed expressions.
""",
        reference="https://sprogteknologi.dk/dataset/1000-talemader-evalueringsdatasaet",
        dataset={
            "path": "mteb/talemaader_pc",
            "revision": "e714d53c059ca83d56c41d22f800da8245bb87fc",
        },
        type="PairClassification",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["dan-Latn"],
        main_score="max_accuracy",
        date=("2024-11-20", "2024-11-20"),
        domains=["Academic", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="created",
        bibtex_citation=r"""
@misc{DSLDK1000Talemader,
  author = {{Det Danske Sprog- og Litteraturselskab}},
  howpublished = {Sprogteknologi.dk},
  language = {Danish},
  note = {CC-BY licensed dataset of 1000 Danish sayings and expressions},
  publisher = {Digitaliseringsstyrelsen \& Det Danske Sprog- og Litteraturselskab},
  title = {1000 danske talemåder - evalueringsdatasæt},
  url = {https://sprogteknologi.dk/dataset/1000-talemader-evalueringsdatasaet},
  year = {2024},
}
""",
    )

    def dataset_transform(self):
        _dataset = {}
        for split in self.metadata.eval_splits:
            hf_dataset = self.dataset[split]
            _dataset[split] = [
                {
                    "sentence1": hf_dataset["sentence1"],
                    "sentence2": hf_dataset["sentence2"],
                    "labels": hf_dataset["label"],
                }
            ]
        self.dataset = _dataset
