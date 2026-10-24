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
@inproceedings{adelani-etal-2024-sib,
  address = {St. Julian{'}s, Malta},
  author = {Adelani, David Ifeoluwa  and
Liu, Hannah  and
Shen, Xiaoyu  and
Vassilyev, Nikita  and
Alabi, Jesujoba O.  and
Mao, Yanke  and
Gao, Haonan  and
Lee, En-Shiun Annie},
  booktitle = {Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers)},
  doi = {10.18653/v1/2024.eacl-long.14},
  editor = {Graham, Yvette  and
Purver, Matthew},
  month = mar,
  pages = {226--245},
  publisher = {Association for Computational Linguistics},
  title = {{SIB}-200: A Simple, Inclusive, and Big Evaluation Dataset for Topic Classification in 200+ Languages and Dialects},
  url = {https://aclanthology.org/2024.eacl-long.14/},
  year = {2024},
}
""",
        prompt="Identify the news category that articles belong to based on their content",
        adapted_from=["SIB200ClusteringS2S"],
    )
