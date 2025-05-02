from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class SentiRuEval2016Classification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SentiRuEval2016",
        dataset={
            "path": "mteb/SentiRuEval2016",
            "revision": "8507eab0deef37f040a750afbcb4dba7a7de9c16",
        },
        description="Russian sentiment analysis evaluation SentiRuEval-2016 devoted to reputation monitoring of banks "
        "and telecom companies in Twitter. We describe the task, data, the procedure of data preparation, "
        "and participants’ results.",
        reference="https://github.com/mokoron/sentirueval",
        type="Classification",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["rus-Cyrl"],
        main_score="accuracy",
        date=("2015-01-01", "2016-01-01"),
        domains=[],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{loukachevitch2016sentirueval,
  author = {Loukachevitch, NV and Rubtsova, Yu V},
  booktitle = {Computational Linguistics and Intellectual Technologies},
  pages = {416--426},
  title = {SentiRuEval-2016: overcoming time gap and data sparsity in tweet sentiment analysis},
  year = {2016},
}
""",
    )
