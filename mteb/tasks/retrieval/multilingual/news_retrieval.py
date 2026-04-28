from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_LANGUAGES_GLOBAL = {
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
}

_LANGUAGES_PUBLIC = {
    "pl": ["pol-Latn"],
    "pt": ["por-Latn"],
}

_DESCRIPTION = "Multilingual news article retrieval with synthetic multihop queries."


class GlobalNewsRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="GlobalNewsRetrieval",
        description=_DESCRIPTION,
        dataset={
            "path": "jinaai/global-news",
            "revision": "b4e5e3d5fe31b5bb0cb9fcd1218a3fa7784bc0e4",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=_LANGUAGES_GLOBAL,
        main_score="ndcg_at_10",
        date=("2026-04-27", "2026-04-27"),
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
        description=_DESCRIPTION,
        dataset={
            "path": "jinaai/public-news",
            "revision": "bddbd5d01b9be88e16a570d13fa2245b37062ad8",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=_LANGUAGES_PUBLIC,
        main_score="ndcg_at_10",
        date=("2026-04-27", "2026-04-27"),
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
