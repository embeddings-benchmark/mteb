from mteb.abstasks.clustering import AbsTaskClustering
from mteb.abstasks.task_metadata import TaskMetadata


class OpenTenderClusteringP2P(AbsTaskClustering):
    max_fraction_of_documents_to_embed = 1.0
    metadata = TaskMetadata(
        name="OpenTenderClusteringP2P",
        dataset={
            "path": "clips/mteb-nl-opentender-cls-pr",
            "revision": "9af5657575a669dc18c7f897a67287ff7d1a0c65",
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
    )

    def dataset_transform(self):
        # reuse the dataset for classification
        for split in self.dataset:
            self.dataset[split] = self.dataset[split].map(
                lambda ex: {
                    "labels": ex["label"],
                    "sentences": ex["text"],
                }
            )
