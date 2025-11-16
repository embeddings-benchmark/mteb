from mteb.abstasks.clustering import AbsTaskClustering
from mteb.abstasks.task_metadata import TaskMetadata


class OpenTenderClusteringS2S(AbsTaskClustering):
    max_fraction_of_documents_to_embed = 1.0
    metadata = TaskMetadata(
        name="OpenTenderClusteringS2S",
        dataset={
            "path": "clips/mteb-nl-opentender-clst-s2s-pr",
            "revision": "ad86cf1813d130e17dda0092d1c4f2c664e68d0c",
        },
        description="This dataset contains all the articles published by the NOS as of the 1st of January 2010. The "
        "data is obtained by scraping the NOS website. The NOS is one of the biggest (online) news "
        "organizations in the Netherlands.",
        reference="https://arxiv.org/abs/2509.12340",
        type="Clustering",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["nld-Latn"],
        main_score="v_measure",
        date=("2025-08-01", "2025-08-10"),
        domains=["Government", "Written"],
        task_subtypes=[],
        license="cc-by-nc-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@misc{banar2025mtebnle5nlembeddingbenchmark,
  archiveprefix = {arXiv},
  author = {Nikolay Banar and Ehsan Lotfi and Jens Van Nooten and Cristina Arhiliuc and Marija Kliocaite and Walter Daelemans},
  eprint = {2509.12340},
  primaryclass = {cs.CL},
  title = {MTEB-NL and E5-NL: Embedding Benchmark and Models for Dutch},
  url = {https://arxiv.org/abs/2509.12340},
  year = {2025},
}
""",
        prompt={
            "query": "Identificeer de hoofdcategorie van aanbestedingen op basis van de titels"
        },
    )
