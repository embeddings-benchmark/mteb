from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_EVAL_LANGS = {
    "bg": ["bul-Cyrl"],
    "bn": ["ben-Beng"],
    "cs": ["ces-Latn"],
    "da": ["dan-Latn"],
    "de": ["deu-Latn"],
    "en": ["eng-Latn"],
    "fa": ["fas-Arab"],
    "fi": ["fin-Latn"],
    "hi": ["hin-Deva"],
    "it": ["ita-Latn"],
    "nl": ["nld-Latn"],
    "pt": ["por-Latn"],
    "ro": ["ron-Latn"],
    "sr": ["srp-Cyrl"],
    "no": ["nor-Latn"],
    "sv": ["swe-Latn"],
}


class WikipediaRetrievalMultilingual(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="WikipediaRetrievalMultilingual",
        description="The dataset is derived from Cohere's wikipedia-2023-11 dataset and contains synthetically generated queries.",
        reference="https://huggingface.co/datasets/ellamind/wikipedia-2023-11-retrieval-multilingual-queries",
        dataset={
            "path": "mteb/WikipediaRetrievalMultilingual",
            "revision": "5f6c91d21f2f5b9afb663858d19848fbd223c775",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=_EVAL_LANGS,
        main_score="ndcg_at_10",
        date=("2023-11-01", "2024-05-15"),
        domains=["Encyclopaedic", "Written"],
        task_subtypes=["Question answering", "Article retrieval"],
        license="cc-by-sa-3.0",
        annotations_creators="LM-generated and reviewed",
        dialect=[],
        sample_creation="LM-generated and verified",
        bibtex_citation="",
    )
