from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class ESGClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="ESGClassification",
        dataset={
            "path": "FinanceMTEB/ESG",
            "revision": "521d56feabadda80b11d6adcc6b335d4c5ad8285",
        },
        description="A finance dataset performs sentence classification under the environmental, social, and corporate governance (ESG) framework.",
        reference="https://arxiv.org/abs/2309.13064",
        category="s2s",
        modalities=["text"],
        type="Classification",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2023-09-23", "2023-09-23"),
        domains=["Finance"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="derived",  # the annotations are a mix of derived, LM-generated and reviewed and expert-annotated. but derived is the predominant source.
        bibtex_citation="""@misc{yang2023investlmlargelanguagemodel,
              title={InvestLM: A Large Language Model for Investment using Financial Domain Instruction Tuning},
              author={Yi Yang and Yixuan Tang and Kar Yan Tam},
              year={2023},
              eprint={2309.13064},
              archivePrefix={arXiv},
              primaryClass={q-fin.GN},
              url={https://arxiv.org/abs/2309.13064},
        }""",
        descriptive_stats={
            "num_samples": {"test": 1000},
            "average_text_length": {"test": 170.817},
            "unique_labels": {"test": 4},
            "labels": {
                "test": {
                    "2": {"count": 497},
                    "0": {"count": 190},
                    "3": {"count": 276},
                    "1": {"count": 37},
                }
            },
        },
    )
