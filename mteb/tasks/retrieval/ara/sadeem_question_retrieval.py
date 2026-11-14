from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class SadeemQuestionRetrieval(AbsTaskRetrieval):
    _EVAL_SPLIT = "test"

    metadata = TaskMetadata(
        name="SadeemQuestionRetrieval",
        dataset={
            "path": "mteb/SadeemQuestionRetrieval",
            "revision": "abe42d6129c3ab7d67df3c4a3325b8a57ab88f79",
        },
        reference="https://huggingface.co/datasets/sadeem-ai/sadeem-ar-eval-retrieval-questions",
        description="SadeemQuestion: A Benchmark Data Set for Community Question-Retrieval Research",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=[_EVAL_SPLIT],
        eval_langs=["ara-Arab"],
        main_score="ndcg_at_10",
        date=("2024-01-01", "2024-04-01"),
        domains=["Written", "Written"],
        task_subtypes=["Article retrieval"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="",
    )
