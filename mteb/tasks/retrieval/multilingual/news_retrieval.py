from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class GlobalNewsRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="GlobalNewsRetrieval",
        description="Multilingual news article retrieval with synthetic multihop queries.",
        dataset={
            "path": "mteb-private/global-news",
            "revision": "234dcd32922d32c78e08bfe506a2b90a5401cf7c",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs={
            "ar": ["ara-Arab"],
            "zh": ["zho-Hans"],
            "en": ["eng-Latn"],
            "fr": ["fra-Latn"],
            "ru": ["rus-Cyrl"],
            "es": ["spa-Latn"],
            "pt": ["por-Latn"],
            "sw": ["swa-Latn"],
            "hi": ["hin-Deva"],
            "ur": ["urd-Arab"],
        },
        main_score="ndcg_at_10",
        date=("2026-03-16", "2026-03-16"),
        domains=["News", "Written"],
        task_subtypes=["Question answering"],
        license="cc-by-4.0",
        annotations_creators="LM-generated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="",
        is_public=False,
        contributed_by="Jina by Elastic",
    )


class PublicNewsRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="PublicNewsRetrieval",
        description="Multilingual news article retrieval with synthetic multihop queries.",
        dataset={
            "path": "mteb-private/public-news",
            "revision": "82dad8c469646feae9631d69ba92bebfc0a586e9",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs={
            "pl": ["pol-Latn"],
            "pt": ["por-Latn"],
        },
        main_score="ndcg_at_10",
        date=("2026-03-16", "2026-03-16"),
        domains=["News", "Written"],
        task_subtypes=["Question answering"],
        license="cc-by-4.0",
        annotations_creators="LM-generated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="",
        is_public=False,
        contributed_by="Jina by Elastic",
    )
