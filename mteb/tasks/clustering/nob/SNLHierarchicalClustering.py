from mteb.abstasks.clustering import AbsTaskClustering
from mteb.abstasks.task_metadata import TaskMetadata


def split_labels(record: dict) -> dict:
    record["labels"] = record["labels"].split(",")
    # First level is trivial
    record["labels"] = record["labels"][1:]
    return record


class SNLHierarchicalClusteringP2P(AbsTaskClustering):
    max_document_to_embed = 1300
    max_fraction_of_documents_to_embed = None

    metadata = TaskMetadata(
        name="SNLHierarchicalClusteringP2P",
        dataset={
            "path": "mteb/SNLHierarchicalClusteringP2P",
            "revision": "693a321c42fb13ffe76bb9043f8d2aaa8f0a9499",
        },
        description="Webscrabed articles from the Norwegian lexicon 'Det Store Norske Leksikon'. Uses articles categories as clusters.",
        reference="https://huggingface.co/datasets/mteb/SNLHierarchicalClusteringP2P",
        type="Clustering",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["nob-Latn"],
        main_score="v_measure",
        date=("2020-01-01", "2024-12-31"),  # best guess
        domains=["Encyclopaedic", "Non-fiction", "Written"],
        license="cc-by-nc-4.0",  # version assumed (not specified beforehand)
        annotations_creators="derived",
        dialect=[],
        task_subtypes=["Thematic clustering"],
        sample_creation="found",
        bibtex_citation=r"""
@mastersthesis{navjord2023beyond,
  author = {Navjord, J{\\o}rgen Johnsen and Korsvik, Jon-Mikkel Ryen},
  school = {Norwegian University of Life Sciences, {\\AA}s},
  title = {Beyond extractive: advancing abstractive automatic text summarization in Norwegian with transformers},
  year = {2023},
}
""",
        prompt="Identify categories in a Norwegian lexicon",
    )
    max_depth = 5


class SNLHierarchicalClusteringS2S(AbsTaskClustering):
    max_document_to_embed = 1300
    max_fraction_of_documents_to_embed = None

    metadata = TaskMetadata(
        name="SNLHierarchicalClusteringS2S",
        dataset={
            "path": "mteb/SNLHierarchicalClusteringS2S",
            "revision": "b505e4ce65f255228e49dd07b6f8148731c5dc64",
        },
        description="Webscrabed articles from the Norwegian lexicon 'Det Store Norske Leksikon'. Uses articles categories as clusters.",
        reference="https://huggingface.co/datasets/mteb/SNLHierarchicalClusteringS2S",
        type="Clustering",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["nob-Latn"],
        main_score="v_measure",
        date=("2020-01-01", "2024-12-31"),  # best guess
        domains=["Encyclopaedic", "Non-fiction", "Written"],
        license="cc-by-nc-4.0",  # version assumed (not specified beforehand)
        annotations_creators="derived",
        dialect=[],
        task_subtypes=["Thematic clustering"],
        sample_creation="found",
        bibtex_citation=r"""
@mastersthesis{navjord2023beyond,
  author = {Navjord, J{\\o}rgen Johnsen and Korsvik, Jon-Mikkel Ryen},
  school = {Norwegian University of Life Sciences, {\\AA}s},
  title = {Beyond extractive: advancing abstractive automatic text summarization in Norwegian with transformers},
  year = {2023},
}
""",
        prompt="Identify categories in a Norwegian lexicon",
    )
    max_depth = 5
