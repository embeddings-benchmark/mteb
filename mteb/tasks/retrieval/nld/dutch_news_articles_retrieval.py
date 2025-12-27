from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class DutchNewsArticlesRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="DutchNewsArticlesRetrieval",
        description="This dataset contains all the articles published by the NOS as of the 1st of January 2010. The "
        "data is obtained by scraping the NOS website. The NOS is one of the biggest (online) news "
        "organizations in the Netherlands.",
        reference="https://www.kaggle.com/datasets/maxscheijen/dutch-news-articles",
        dataset={
            "path": "clips/mteb-nl-news-articles-ret",
            "revision": "c8042a86f3eb0d1fcec79a4a44ebf1eafe635462",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["nld-Latn"],
        main_score="ndcg_at_10",
        date=("2009-11-01", "2010-01-01"),
        domains=["Written", "News"],
        task_subtypes=["Article retrieval"],
        license="cc-by-nc-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="",
        prompt={
            "query": "Gegeven een titel, haal het nieuwsartikel op dat het beste bij de titel past"
        },
    )
