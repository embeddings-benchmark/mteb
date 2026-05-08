from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class LexRetrievalv1(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="LexRetrieval.v1",  # we suspect that there will be more
        description="Consists of simulated user-queries from the Danish encyclopaedia Lex.dk. The queries are synthetically generated dataset based "
        + "on existing user queries of the existing chatbot.",
        category="t2t",
        reference=None,
        main_score="ndcg_at_10",
        eval_langs=["dan-Latn"],
        eval_splits=["test"],
        modalities=["text"],
        type="Retrieval",
        domains=["Non-fiction", "Encyclopaedic"],
        license="not specified",
        date=("2009-01-28", "2025-08-12"),
        dataset={
            "path": "mteb-private/LexRetrieval",
            "revision": "b5b22dee3a30bb32ac579c14afe9eb152e8c70c7",
        },
        annotations_creators="LM-generated and reviewed",
        dialect=[],
        prompt="Given a question in Danish, retrieve the documents that can can help answer the question.",
        task_subtypes=[
            "Article retrieval",
            "Question answering",
            "Conversational retrieval",
        ],
        sample_creation="found",  # queries are synthetically generated based on existing user queries of the existing chatbot, but the documents are found, not generated
        bibtex_citation="",  # probably an upcoming paper
        adapted_from=None,
        is_public=False,
        contributed_by="Aarhus University & Lex.dk",
    )
