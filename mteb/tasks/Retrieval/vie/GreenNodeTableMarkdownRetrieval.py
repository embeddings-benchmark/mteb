from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

TEST_SAMPLES = 2048


class GreenNodeTableMarkdownRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="GreenNodeTableMarkdownRetrieval",
        description="GreenNodeTable documents",
        reference="https://huggingface.co/GreenNode",
        dataset={
            "path": "GreenNode/GreenNode-Table-Markdown-Retrieval-VN",
            "revision": "d86a4dad9fd7c70359f617d86984395ea89be1c5",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["vie-Latn"],
        main_score="ndcg_at_10",
        date=("2025-03-16", "2025-03-16"),
        domains=["Financial", "Encyclopaedic", "Non-fiction"],
        task_subtypes=["Article retrieval"],
        license="mit",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="",  # TODO: Add bibtex citation when the paper is published
    )
