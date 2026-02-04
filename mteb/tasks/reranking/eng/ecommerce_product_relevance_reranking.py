from __future__ import annotations

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class ERESSReranking(AbsTaskRetrieval):
    """ERESS: E-commerce Relevance Evaluation Suite for Reranking
    Dataset: https://huggingface.co/datasets/thebajajra/eress
    """

    metadata = TaskMetadata(
        name="ERESSReranking",
        description="""ERESS is a comprehensive e-commerce reranking dataset designed for holistic
    evaluation of reranking models. It includes diverse query intents including
    attribute-rich queries, navigational queries, gift/audience-specific queries,
    utility queries, and more.""",
        reference="https://huggingface.co/datasets/thebajajra/ERESSReranking",
        type="Reranking",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_5",
        dataset={
            "path": "thebajajra/ERESSReranking",
            "revision": "c095d55559370c1e8b0ae933ae26da111ae62966",
        },
        date=("1996-05-01", "2025-12-24"),
        domains=["Web", "E-commerce"],
        task_subtypes=["Product Reranking", "Query-Product Relevance"],
        license="apache-2.0",
        annotations_creators="LM-generated",  # LLM ensemble annotation
        dialect=[],
        sample_creation="found",  # Real-world queries
        prompt="Rerank products by relevance to the e-commerce query",
        bibtex_citation=r"""
@article{Bajaj2026RexRerankers,
  author = {Bajaj, Rahul and Garg, Anuj and Nupur, Jaya},
  journal = {Hugging Face Blog (Community Article)},
  month = jan,
  title = {{RexRerankers}: {SOTA} Rankers for Product Discovery and {AI} Assistants},
  url = {https://huggingface.co/blog/thebajajra/rexrerankers},
  urldate = {2026-01-24},
  year = {2026},
}
""",
    )
