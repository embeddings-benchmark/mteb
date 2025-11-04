from mteb.abstasks.clustering import AbsTaskClustering
from mteb.abstasks.task_metadata import TaskMetadata


class DutchNewsArticlesClusteringP2P(AbsTaskClustering):
    max_fraction_of_documents_to_embed = 1.0
    metadata = TaskMetadata(
        name="DutchNewsArticlesClusteringP2P",
        dataset={
            "path": "clips/mteb-nl-news-articles-cls",
            "revision": "0a7227d31f85c5676be92767f8df5405ea93de54",
        },
        description="This dataset contains all the articles published by the NOS as of the 1st of January 2010. The "
        "data is obtained by scraping the NOS website. The NOS is one of the biggest (online) news "
        "organizations in the Netherlands.",
        reference="https://www.kaggle.com/datasets/maxscheijen/dutch-news-articles",
        type="Clustering",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["nld-Latn"],
        main_score="v_measure",
        date=("2009-11-01", "2010-01-01"),
        domains=["Written", "News"],
        task_subtypes=[],
        license="cc-by-nc-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="",
        prompt={
            "query": "Identificeer de hoofdcategorie van nieuwsartikelen op basis van de titels en de inhoud"
        },
    )

    def dataset_transform(self):
        for split in self.dataset:
            self.dataset[split] = self.dataset[split].rename_columns(
                {"label": "labels", "text": "sentences"}
            )
