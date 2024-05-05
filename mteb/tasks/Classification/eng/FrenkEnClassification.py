from __future__ import annotations

from mteb.abstasks import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class FrenkEnClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="FrenkEnClassification",
        description="English subset of the FRENK dataset",
        dataset={
            "path": "classla/FRENK-hate-en",
            "revision": "52483dba0ff23291271ee9249839865e3c3e7e50",
        },
        reference="https://arxiv.org/abs/1906.02045",
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
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
        n_samples={"test": 2300},
        avg_character_length={"test": 188.75},
    )
