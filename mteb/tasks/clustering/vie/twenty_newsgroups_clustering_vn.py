from mteb.abstasks import AbsTaskClusteringLegacy
from mteb.abstasks.task_metadata import TaskMetadata


class TwentyNewsgroupsClusteringVN(AbsTaskClusteringLegacy):
    metadata = TaskMetadata(
        name="TwentyNewsgroupsClustering-VN",
        description="A translated dataset from Clustering of the 20 Newsgroups dataset (subject only). The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system: - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation. - Applies advanced embedding models to filter the translations. - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.",
        reference="https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html",
        dataset={
            "path": "GreenNode/twentynewsgroups-clustering-vn",
            "revision": "770e1b9029cd85c79410bc6df1528a43fc2b9ad1",
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
        domains=["News", "Written"],
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
        adapted_from=["TwentyNewsgroupsClustering"],
    )
