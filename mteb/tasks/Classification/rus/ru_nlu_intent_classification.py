from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.MultilingualTask import MultilingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata


class RuNLUIntentClassification(AbsTaskClassification, MultilingualTask):
    metadata = TaskMetadata(
        name="RuNLUIntentClassification",
        dataset={
            "path": "mteb/RuNLUIntentClassification",
            "revision": "424d0f767aaa5c411e3a529eec04658e5726a39e",
        },
        description=(
            "Contains natural language data for human-robot interaction in home domain which we collected and"
            " annotated for evaluating NLU Services/platforms."
        ),
        reference="https://arxiv.org/abs/1903.05566",
        type="Classification",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs={
            "rus-eng": [
                "rus-Cyrl",
                "rus-Latn",
            ],
            "rus": [
                "rus-Cyrl",
            ],
        },
        main_score="accuracy",
        date=("2019-03-26", "2019-03-26"),
        domains=[],
        task_subtypes=["Intent classification"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@misc{liu2019benchmarkingnaturallanguageunderstanding,
  archiveprefix = {arXiv},
  author = {Xingkun Liu and Arash Eshghi and Pawel Swietojanski and Verena Rieser},
  eprint = {1903.05566},
  primaryclass = {cs.CL},
  title = {Benchmarking Natural Language Understanding Services for building Conversational Agents},
  url = {https://arxiv.org/abs/1903.05566},
  year = {2019},
}
""",
    )
