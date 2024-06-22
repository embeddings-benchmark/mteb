from __future__ import annotations

from mteb.abstasks import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class FrenkSlClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="FrenkSlClassification",
        description="Slovenian subset of the FRENK dataset. Also available on HuggingFace dataset hub: English subset, Croatian subset.",
        dataset={
            "path": "classla/FRENK-hate-sl",
            "revision": "37c8b42c63d4eb75f549679158a85eb5bd984caa",
            "trust_remote_code": True,
        },
        reference="https://arxiv.org/pdf/1906.02045",
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["slv-Latn"],
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
        n_samples={"test": 2177},
        avg_character_length={"test": 136.61},
    )

    def dataset_transform(self):
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["test"]
        )
