from mteb.abstasks.clustering import AbsTaskClustering
from mteb.abstasks.clustering_legacy import AbsTaskClusteringLegacy
from mteb.abstasks.task_metadata import TaskMetadata


class SwednClustering(AbsTaskClusteringLegacy):
    metadata = TaskMetadata(
        name="SwednClustering",
        dataset={
            "path": "mteb/SwednClustering",
            "revision": "45d2a99c3f1b6ee6189a6bb762ed74b7ef45dd9d",
        },
        description="The SWE-DN corpus is based on 1,963,576 news articles from the Swedish newspaper Dagens Nyheter (DN) during the years 2000--2020. The articles are filtered to resemble the CNN/DailyMail dataset both regarding textual structure. This dataset uses the category labels as clusters.",
        reference="https://spraakbanken.gu.se/en/resources/swedn",
        type="Clustering",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["swe-Latn"],
        main_score="v_measure",
        date=("2000-01-01", "2020-12-31"),  # best guess
        domains=["News", "Non-fiction", "Written"],
        license=None,
        annotations_creators="derived",
        dialect=[],
        task_subtypes=["Thematic clustering"],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{monsen2021method,
  author = {Monsen, Julius and J{\"o}nsson, Arne},
  booktitle = {Proceedings of CLARIN Annual Conference},
  title = {A method for building non-english corpora for abstractive text summarization},
  year = {2021},
}
""",
        superseded_by="SwednClusteringP2P",
    )


class SwednClusteringP2P(AbsTaskClustering):
    max_document_to_embed = 2048
    max_fraction_of_documents_to_embed = None

    metadata = TaskMetadata(
        name="SwednClusteringP2P",
        dataset={
            "path": "mteb/SwednClusteringP2P",
            "revision": "f8dbf10ec231cc25e9f63454d5cd2d90af95e5f8",
        },
        description="The SWE-DN corpus is based on 1,963,576 news articles from the Swedish newspaper Dagens Nyheter (DN) during the years 2000--2020. The articles are filtered to resemble the CNN/DailyMail dataset both regarding textual structure. This dataset uses the category labels as clusters.",
        reference="https://spraakbanken.gu.se/en/resources/swedn",
        type="Clustering",
        category="t2c",
        modalities=["text"],
        eval_splits=["summaries", "articles"],
        eval_langs=["swe-Latn"],
        main_score="v_measure",
        date=("2000-01-01", "2020-12-31"),  # best guess
        domains=["News", "Non-fiction", "Written"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        task_subtypes=["Thematic clustering"],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{monsen2021method,
  author = {Monsen, Julius and J{\"o}nsson, Arne},
  booktitle = {Proceedings of CLARIN Annual Conference},
  title = {A method for building non-english corpora for abstractive text summarization},
  year = {2021},
}
""",
        prompt="Identify news categories in Swedish passages",
    )


class SwednClusteringFastS2S(AbsTaskClustering):
    max_document_to_embed = 2048
    max_fraction_of_documents_to_embed = None

    metadata = TaskMetadata(
        name="SwednClusteringS2S",
        dataset={
            "path": "mteb/SwednClusteringS2S",
            "revision": "4dc1f92a8d5c4fe4be7995baa8009384f46d98d6",
        },
        description="The SWE-DN corpus is based on 1,963,576 news articles from the Swedish newspaper Dagens Nyheter (DN) during the years 2000--2020. The articles are filtered to resemble the CNN/DailyMail dataset both regarding textual structure. This dataset uses the category labels as clusters.",
        reference="https://spraakbanken.gu.se/en/resources/swedn",
        type="Clustering",
        category="t2c",
        modalities=["text"],
        eval_splits=["headlines"],
        eval_langs=["swe-Latn"],
        main_score="v_measure",
        date=("2000-01-01", "2020-12-31"),  # best guess
        domains=["News", "Non-fiction", "Written"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        task_subtypes=["Thematic clustering"],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{monsen2021method,
  author = {Monsen, Julius and J{\"o}nsson, Arne},
  booktitle = {Proceedings of CLARIN Annual Conference},
  title = {A method for building non-english corpora for abstractive text summarization},
  year = {2021},
}
""",
        prompt="Identify news categories in Swedish passages",
    )
