from __future__ import annotations

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class ElasticKBRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="ElasticKBRetrieval",
        description=(
            "Retrieval benchmark built from the Elastic support knowledge base. "
            "Contains 9,754 documents and 420 queries (232 from real-world support "
            "chat sessions, 188 synthetic queries generated from KB articles). "
            "Relevance judgments are augmented labels produced by exhaustive "
            "all-pairs LLM annotation using strict comparison to original doc "
            "(grounding doc that lead to self-served ticket for real-world queries "
            "and generating doc for synthetic queries)."
        ),
        reference="https://huggingface.co/datasets/EmiliaElastic/elastic-kb-retrieval",
        dataset={
            "path": "EmiliaElastic/elastic-kb-retrieval",
            "revision": "3f60c92a79e3b09aa48688aa2b8437037a5720e8",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs={"en": ["eng-Latn"]},
        main_score="ndcg_at_10",
        date=["2024-01-01", "2025-04-01"],
        domains=["Written", "Engineering"],
        task_subtypes=["Question answering"],
        license="cc-by-4.0",
        annotations_creators="LM-generated",
        dialect=[],
        prompt={
            "query": "Given a support question, retrieve knowledge base articles that answer the question"
        },
        sample_creation="found and created",
        bibtex_citation="",
    )
