from __future__ import annotations

from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class HWUIntentClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="HWUIntentClassification",
        description="",
        dataset={
            "path": "DeepPavlov/hwu_intent_classification",
            "revision": "050d2712be8b6f069a4350139c9c2d3ed7ce4aaf",
        },
        reference="https://arxiv.org/abs/1903.05566",
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="f1",
        date=("2019-03-26", "2019-03-26"),
        domains=[],
        task_subtypes=["Intent classification"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@misc{liu2019benchmarkingnaturallanguageunderstanding,
          title={Benchmarking Natural Language Understanding Services for building Conversational Agents}, 
          author={Xingkun Liu and Arash Eshghi and Pawel Swietojanski and Verena Rieser},
          year={2019},
          eprint={1903.05566},
          archivePrefix={arXiv},
          primaryClass={cs.CL},
          url={https://arxiv.org/abs/1903.05566}, 
    }""",
    )
