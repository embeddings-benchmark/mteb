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
@inproceedings{monsen2021method,
  author = {Monsen, Julius and J{\"o}nsson, Arne},
  booktitle = {Proceedings of CLARIN Annual Conference},
  title = {A method for building non-english corpora for abstractive text summarization},
  year = {2021},
}
""",
        prompt={
            "query": "Given a Swedish news headline retrieve summaries or news articles"
        },
    )
