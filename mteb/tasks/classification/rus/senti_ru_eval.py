from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


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
        superseded_by="SentiRuEval2016.v2",
    )


class SentiRuEval2016ClassificationV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SentiRuEval2016.v2",
        dataset={
            "path": "mteb/senti_ru_eval2016",
            "revision": "bfa4cbec1753ffed29a8244a4ec208cc9e6c09a0",
        },
        description="Russian sentiment analysis evaluation SentiRuEval-2016 devoted to reputation monitoring of banks and telecom companies in Twitter. We describe the task, data, the procedure of data preparation, and participants’ results. This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)",
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
        adapted_from=["SentiRuEval2016Classification"],
    )
