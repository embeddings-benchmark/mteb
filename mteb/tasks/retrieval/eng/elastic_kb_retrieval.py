from __future__ import annotations

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class ElasticKBRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="ElasticKBRetrieval",
        description=(
            "Retrieval benchmark built from the Elastic support knowledge base. "
            "Contains 9,754 documents (real documents from the Elastic support knowledge base) and 420 queries (232 from real-world support "
            "chat sessions, 188 synthetic queries generated from KB articles). "
            "Relevance judgments are augmented labels produced by exhaustive "
            "all-pairs LLM annotation using strict comparison to original doc "
            "(grounding doc that lead to self-served ticket for real-world queries "
            "and generating doc for synthetic queries)."
        ),
        reference="https://huggingface.co/blog/rteb",  # private set
        dataset={
            "path": "mteb-private/elastic-kb-retrieval",
            "revision": "21bdf1e024bf7c9f46720017559ce2f8c6116507",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["synthetic_test", "real_chat_test"],
        eval_langs={"en": ["eng-Latn"]},
        main_score="ndcg_at_10",
        is_public=False,
        date=("2015-01-01", "2026-04-01"),
        domains=["Written", "Engineering"],
        task_subtypes=["Question answering", "Conversational retrieval"],
        license="not specified",  # shared as an evaluation dataset, results can be shared, and the dataset is allowed to be sent to embedding APIs
        annotations_creators="LM-generated",
        dialect=[],
        prompt={
            "query": "Given a support question, retrieve knowledge base articles that answer the question"
        },
        sample_creation="multiple",  # see description
        bibtex_citation="",
        contributed_by="Jina by Elastic",
    )
