from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class SwednRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="SwednRetrieval",
        dataset={
            "path": "mteb/SwednRetrieval",
            "revision": "3920fa0e9ae78f6af3d8a20b33b39e1f9acdfdf9",
        },
        description="The SWE-DN corpus is based on 1,963,576 news articles from the Swedish newspaper Dagens Nyheter (DN) during the years 2000--2020. The articles are filtered to resemble the CNN/DailyMail dataset both regarding textual structure",
        reference="https://spraakbanken.gu.se/en/resources/swedn",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["swe-Latn"],
        main_score="ndcg_at_10",
        date=("2000-01-01", "2020-12-31"),
        domains=["News", "Non-fiction", "Written"],
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        task_subtypes=["Article retrieval"],
        sample_creation="found",
        bibtex_citation=r"""
@misc{monsenjulius_swedn_2024,
  doi = {10.23695/36V9-9017},
  url = {https://sprakbanken.se/resurser/swedn},
  author = {{Monsen, Julius} and {Jönsson, Arne}},
  keywords = {Language Technology (Computational Linguistics), Swedish},
  language = {sv},
  title = {SweDN 1.0},
  publisher = {Språkbanken},
  year = {2024},
}
""",
        prompt={
            "query": "Given a Swedish news headline retrieve summaries or news articles"
        },
    )
