from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskSTS import AbsTaskSTS


class STS12STS(AbsTaskSTS):
    metadata = TaskMetadata(
        name="STS12",
        dataset={
            "path": "mteb/sts12-sts",
            "revision": "a0d554a64d88156834ff5ae9920b964011b16384",
        },
        description="SemEval-2012 Task 6.",
        reference="https://www.aclweb.org/anthology/S12-1051.pdf",
        type="STS",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="cosine_spearman",
        date=("2005-01-01", "2012-12-31"),
        form=["written"],
        domains=["Encyclopaedic", "News"],
        task_subtypes=[],
        license="Not specified",
        socioeconomic_status="mixed",
        annotations_creators="human-annotated",
        dialect=[],
        text_creation="created",
        bibtex_citation="""@inproceedings{agirre2012semeval,
          title={SemEval-2012 Task 6: A Pilot on Semantic Textual Similarity.* SEM 2012: The First Joint Conference on Lexical and Computational Semantics—},
          author={Agirre, Eneko and Cer, Daniel and Diab, Mona and Gonzalez-Agirre, Aitor},
          booktitle={Proceedings of the Sixth International Workshop on Semantic Evaluation (SemEval 2012), Montr{\'e}al, QC, Canada},
          pages={7--8},
          year={2012}
          }""",
        n_samples={"test": 6216},
        avg_character_length={"test": 64.7},
    )

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = super().metadata_dict
        metadata_dict["min_score"] = 0
        metadata_dict["max_score"] = 5
        return metadata_dict
