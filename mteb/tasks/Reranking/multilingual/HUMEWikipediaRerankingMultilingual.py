from __future__ import annotations

from mteb.abstasks.MultilingualTask import MultilingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskReranking import AbsTaskReranking

_LANGUAGES = {
    "en": ["eng-Latn"],
    "da": ["dan-Latn"],
    "no": ["nob-Latn"],
}


class HUMEWikipediaRerankingMultilingual(AbsTaskReranking, MultilingualTask):
    fast_loading = True
    metadata = TaskMetadata(
        name="HUMEWikipediaRerankingMultilingual",
        description="Human evaluation subset of Wikipedia reranking dataset across multiple languages.",
        reference="https://github.com/ellamind/wikipedia-2023-11-reranking-multilingual",
        dataset={
            "path": "mteb/mteb-human-wiki-reranking",
            "revision": "bdbce1ba2d0e58e88d1d13c54a555154adc5c165",
        },
        type="Reranking",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=_LANGUAGES,
        main_score="map",
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
