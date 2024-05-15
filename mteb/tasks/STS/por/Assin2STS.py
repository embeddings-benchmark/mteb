from __future__ import annotations

from mteb.abstasks.AbsTaskSTS import AbsTaskSTS
from mteb.abstasks.TaskMetadata import TaskMetadata


class Assin2STS(AbsTaskSTS):
    metadata = TaskMetadata(
        name="assin2-sts",
        dataset={
            "path": "assin2",
            "revision": "0ff9c86779e06855536d8775ce5550550e1e5a2d",
        },
        description="Semantic Textual Similarity part of the ASSIN 2, an evaluation shared task collocated with STIL 2019.",
        reference="https://link.springer.com/chapter/10.1007/978-3-030-41505-1_39",
        type="STS",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["por-Latn"],
        main_score="cosine_spearman",
        date=("2019-01-01", "2019-09-16"),  # best guess
        form=["written"],
        domains=[],
        task_subtypes=["Claim verification"],
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

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = super().metadata_dict
        metadata_dict["min_score"] = 1
        metadata_dict["max_score"] = 5
        return metadata_dict

    def dataset_transform(self) -> None:
        self.dataset = self.dataset.rename_column("premise", "sentence1")
        self.dataset = self.dataset.rename_column("hypothesis", "sentence2")
        self.dataset = self.dataset.rename_column("relatedness_score", "score")
