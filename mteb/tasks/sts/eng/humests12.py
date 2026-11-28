from mteb.abstasks import AbsTaskSTS
from mteb.abstasks.task_metadata import TaskMetadata
from mteb.tasks.sts.eng.sts12_sts import STS12_CITATION


class HUMESTS12(AbsTaskSTS):
    metadata = TaskMetadata(
        name="HUMESTS12",
        dataset={
            "path": "mteb/mteb-human-sts12-sts",
            "revision": "76cbf76792ec03cb1f76dc6ada05abcb23c82c0c",
        },
        description="Human evaluation subset of SemEval-2012 Task 6.",
        reference="https://www.aclweb.org/anthology/S12-1051.pdf",
        type="STS",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="cosine_spearman",
        date=("2005-01-01", "2012-12-31"),
        domains=["Encyclopaedic", "News", "Written"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation=STS12_CITATION,
        adapted_from=["STS12"],
    )
    min_score = 0
    max_score = 5
