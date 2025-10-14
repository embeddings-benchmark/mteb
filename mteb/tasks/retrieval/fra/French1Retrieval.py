from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class French1Retrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="French1Retrieval",
        description="French general knowledge retrieval dataset with queries about celebrities, historical figures, and cultural topics.",
        reference="https://huggingface.co/datasets/mteb-private/French1Retrieval-sample",
        dataset={
            "path": "mteb-private/French1Retrieval",
            "revision": "c5c5a44f75dff57be44e5623e817239b050bf0f2",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fra-Latn"],
        main_score="ndcg_at_10",
        date=("2024-01-01", "2024-01-01"),
        domains=["Encyclopaedic", "Written"],
        task_subtypes=["Article retrieval"],
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="",
        is_public=False,
    )
