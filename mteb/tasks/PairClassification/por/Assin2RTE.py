from __future__ import annotations

from mteb.abstasks.AbsTaskPairClassification import AbsTaskPairClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class Assin2RTE(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="Assin2RTE",
        dataset={
            "path": "nilc-nlp/assin2",
            "revision": "0ff9c86779e06855536d8775ce5550550e1e5a2d",
        },
        description="Recognizing Textual Entailment part of the ASSIN 2, an evaluation shared task collocated with STIL 2019.",
        reference="https://link.springer.com/chapter/10.1007/978-3-030-41505-1_39",
        type="PairClassification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["por-Latn"],
        main_score="ap",
        date=("2019-01-01", "2019-09-16"),  # best guess
        form=["written"],
        domains=[],
        task_subtypes=["Textual Entailment"],
        license="Not specified",
        socioeconomic_status="mixed",
        annotations_creators="human-annotated",
        dialect=[],
        text_creation="found",
        bibtex_citation="""@inproceedings{real2020assin,
            title={The assin 2 shared task: a quick overview},
            author={Real, Livy and Fonseca, Erick and Oliveira, Hugo Goncalo},
            booktitle={International Conference on Computational Processing of the Portuguese Language},
            pages={406--412},
            year={2020},
            organization={Springer}
        }""",
        n_samples={"test": 2448},
        avg_character_length={"test": 53.55},
    )

    def dataset_transform(self):
        _dataset = {}
        self.dataset = self.stratified_subsampling(
            self.dataset,
            seed=self.seed,
            splits=self.metadata.eval_splits,
            label="entailment_judgment",
        )
        for split in self.metadata.eval_splits:
            _dataset[split] = [
                {
                    "sent1": self.dataset[split]["premise"],
                    "sent2": self.dataset[split]["hypothesis"],
                    "labels": self.dataset[split]["entailment_judgment"],
                }
            ]
        self.dataset = _dataset
