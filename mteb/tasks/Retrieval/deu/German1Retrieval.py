from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class German1Retrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="German1Retrieval",
        description="German dialogue retrieval dataset with business conversations and workplace communication scenarios.",
        reference="https://huggingface.co/datasets/mteb-private/German1Retrieval-sample",
        dataset={
            "path": "mteb-private/German1Retrieval",
            "revision": "d83dd140fc8fdf96372a6e9193cd4fd2d5cb72bd",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["deu-Latn"],
        main_score="ndcg_at_10",
        date=("2024-01-01", "2024-01-01"),
        domains=["Written", "Non-fiction"],
        task_subtypes=["Question answering"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="",
        is_public=False,
    )
