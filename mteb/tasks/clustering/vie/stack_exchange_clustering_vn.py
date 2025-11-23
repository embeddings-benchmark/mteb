from mteb.abstasks.clustering_legacy import AbsTaskClusteringLegacy
from mteb.abstasks.task_metadata import TaskMetadata


class StackExchangeClusteringVN(AbsTaskClusteringLegacy):
    metadata = TaskMetadata(
        name="StackExchangeClustering-VN",
        description="A translated dataset from Clustering of titles from 121 stackexchanges. Clustering of 25 sets, each with 10-50 classes, and each class with 100 - 1000 sentences. The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system: - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation. - Applies advanced embedding models to filter the translations. - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.",
        reference="https://arxiv.org/abs/2104.07081",
        dataset={
            "path": "GreenNode/stackexchange-clustering-vn",
            "revision": "cf01db048f2bf705741675b51613dc48e0bb122b",
        },
        type="Clustering",
        category="t2c",
        eval_splits=["test"],
        eval_langs=["vie-Latn"],
        main_score="v_measure",
        date=("2025-07-29", "2025-07-30"),
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="machine-translated and LM verified",
        domains=["Web", "Written"],
        task_subtypes=["Thematic clustering"],
        bibtex_citation=r"""
@misc{pham2025vnmtebvietnamesemassivetext,
  archiveprefix = {arXiv},
  author = {Loc Pham and Tung Luu and Thu Vo and Minh Nguyen and Viet Hoang},
  eprint = {2507.21500},
  primaryclass = {cs.CL},
  title = {VN-MTEB: Vietnamese Massive Text Embedding Benchmark},
  url = {https://arxiv.org/abs/2507.21500},
  year = {2025},
}
""",
        adapted_from=["StackExchangeClustering"],
    )
