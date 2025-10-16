from mteb.abstasks import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_LANGUAGES = {
    "en": ["eng-Latn"],
    "da": ["dan-Latn"],
    "no": ["nob-Latn"],
}


class HUMEWikipediaRerankingMultilingual(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="HUMEWikipediaRerankingMultilingual",
        description="Human evaluation subset of Wikipedia reranking dataset across multiple languages.",
        reference="https://github.com/ellamind/wikipedia-2023-11-reranking-multilingual",
        dataset={
            "path": "mteb/HUMEWikipediaRerankingMultilingual",
            "revision": "dd67517891b669ed96658b4dfea4741f0c10480a",
        },
        type="Reranking",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=_LANGUAGES,
        main_score="map_at_1000",
        date=("2023-01-01", "2023-12-31"),
        domains=["Encyclopaedic", "Written"],
        task_subtypes=[],
        license="cc-by-sa-3.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@misc{wikipedia_reranking_2023,
  author = {Ellamind},
  title = {Wikipedia 2023-11 Reranking Multilingual Dataset},
  url = {https://github.com/ellamind/wikipedia-2023-11-reranking-multilingual},
  year = {2023},
}
""",
        prompt="Given a query, rerank the Wikipedia passages by their relevance to the query",
        adapted_from=["WikipediaRerankingMultilingual"],
    )
