from mteb.abstasks.clustering_legacy import AbsTaskClusteringLegacy
from mteb.abstasks.task_metadata import TaskMetadata

_LANGUAGES = {
    "eng_Latn": ["eng-Latn"],
    "arb_Arab": ["ara-Arab"],
    "dan_Latn": ["dan-Latn"],
    "fra_Latn": ["fra-Latn"],
    "rus_Cyrl": ["rus-Cyrl"],
}


class HUMESIB200ClusteringS2S(AbsTaskClusteringLegacy):
    fast_loading = True
    metadata = TaskMetadata(
        name="HUMESIB200ClusteringS2S",
        description="Human evaluation subset of Clustering of news article headlines from SIB-200. Clustering of 10 sets, each with 8 categories and 10 texts per category.",
        reference="https://github.com/dadelani/sib-200",
        dataset={
            "path": "mteb/mteb-human-sib200-clustering",
            "revision": "d41717b1b94c0155f5ae7f84034e01af61be455e",
        },
        type="Clustering",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=_LANGUAGES,
        main_score="v_measure",
        date=("2020-01-01", "2022-12-31"),
        domains=["News", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{adelani-etal-2023-sib,
  address = {Toronto, Canada},
  author = {Adelani, David Ifeoluwa  and
Hedderich, Michael A.  and
Zhu, Dawei  and
van den Berg, Esther  and
Klakow, Dietrich},
  booktitle = {Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  doi = {10.18653/v1/2023.acl-long.660},
  month = jul,
  pages = {11784--11801},
  publisher = {Association for Computational Linguistics},
  title = {{SIB}-200: A Large-Scale News Classification Dataset for Over 200 Languages},
  url = {https://aclanthology.org/2023.acl-long.660},
  year = {2023},
}
""",
        prompt="Identify the news category that articles belong to based on their content",
        adapted_from=["SIB200ClusteringS2S"],
    )
