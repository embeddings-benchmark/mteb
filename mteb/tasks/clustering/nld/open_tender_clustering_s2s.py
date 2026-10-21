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
        reference="https://aclanthology.org/2026.findings-acl.1236/",
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
@inproceedings{banar2025mtebnle5nlembeddingbenchmark,
  address = {San Diego, California, United States},
  author = {Banar, Nikolay and Lotfi, Ehsan and Van Nooten, Jens and Arhiliuc, Cristina and Kliocaite, Marija and Daelemans, Walter},
  booktitle = {Findings of the {A}ssociation for {C}omputational {L}inguistics: {ACL} 2026},
  doi = {10.18653/v1/2026.findings-acl.1236},
  editor = {Liakata, Maria and Moreira, Viviane P. and Zhang, Jiajun and Jurgens, David},
  isbn = {979-8-89176-395-1},
  month = {jul},
  pages = {24684--24709},
  publisher = {Association for Computational Linguistics},
  title = {{MTEB}-{NL} and {E}5-{NL}: Embedding Benchmark and Models for {D}utch},
  url = {https://aclanthology.org/2026.findings-acl.1236/},
  year = {2026},
}
""",
        prompt="Identificeer de hoofdcategorie van aanbestedingen op basis van de titels",
    )
