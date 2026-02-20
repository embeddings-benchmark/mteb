from mteb.abstasks.sts import AbsTaskSTS
from mteb.abstasks.task_metadata import TaskMetadata


class Farsick(AbsTaskSTS):
    metadata = TaskMetadata(
        name="Farsick",
        description="A Persian Semantic Textual Similarity And Natural Language Inference Dataset",
        reference="https://github.com/ZahraGhasemi-AI/FarSick",
        dataset={
            "path": "MCINext/farsick-sts",
            "revision": "f8b8d630f631c6c16b7bc3cb924bdf62a51bed06",
        },
        type="STS",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="cosine_spearman",
        date=("2024-09-01", "2024-12-31"),
        domains=[],
        task_subtypes=[],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=""" """,
    )
    min_score = 1


class SynPerSTS(AbsTaskSTS):
    metadata = TaskMetadata(
        name="SynPerSTS",
        description="Synthetic Persian Semantic Textual Similarity Dataset",
        reference="https://mcinext.com/",
        dataset={
            "path": "MCINext/synthetic-persian-sts",
            "revision": "914047db08928b5326d8b106583dc563b73d1ecf",
        },
        type="STS",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="cosine_spearman",
        date=("2024-09-01", "2024-12-31"),
        domains=["Web", "News", "Religious", "Blog"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="LM-generated",
        dialect=[],
        sample_creation="LM-generated and verified",
        bibtex_citation=""" """,
    )
    min_score = 1


class Query2Query(AbsTaskSTS):
    metadata = TaskMetadata(
        name="Query2Query",
        description="Query to Query Datasets.",
        reference="https://mcinext.com/",
        dataset={
            "path": "MCINext/query-to-query-sts",
            "revision": "52602079f9032fcf181775a310d79d2f197534e4",
        },
        type="STS",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="cosine_spearman",
        date=("2024-09-01", "2024-12-31"),
        domains=[],
        task_subtypes=[],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=""" """,
    )
    min_score = 0
    max_score = 2
