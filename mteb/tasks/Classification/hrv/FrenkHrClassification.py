from __future__ import annotations

from mteb.abstasks import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class FrenkHrClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="FrenkHrClassification",
        description="Croatian subset of the FRENK dataset",
        dataset={
            "path": "classla/FRENK-hate-hr",
            "revision": "e7fc9f3d8d6c5640a26679d8a50b1666b02cc41f",
        },
        reference="https://arxiv.org/abs/1906.02045",
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["hrv-Latn"],
        main_score="accuracy",
        date=("2021-05-28", "2021-05-28"),
        form=["written"],
        domains=["Social"],
        task_subtypes=["Sentiment/Hate speech"],
        license="Not specified",
        socioeconomic_status="low",
        annotations_creators="derived",
        dialect=[],
        text_creation="found",
        bibtex_citation="""@misc{ljubešić2019frenk,
        title={The FRENK Datasets of Socially Unacceptable Discourse in Slovene and English}, 
        author={Nikola Ljubešić and Darja Fišer and Tomaž Erjavec},
        year={2019},
        eprint={1906.02045},
        archivePrefix={arXiv},
        primaryClass={cs.CL},
        url={https://arxiv.org/abs/1906.02045}
        }""",
        n_samples={"test": 2120},
        avg_character_length={"test": 89.86},
    )
