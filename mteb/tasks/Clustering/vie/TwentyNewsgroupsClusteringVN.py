from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskClustering import AbsTaskClustering


class TwentyNewsgroupsClusteringVN(AbsTaskClustering):
    metadata = TaskMetadata(
        name="TwentyNewsgroupsClustering-VN",
        description="Clustering of the 20 Newsgroups dataset (subject only).",
        reference="https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html",
        dataset={
            "path": "GreenNode/twentynewsgroups-clustering-vn",
            "revision": "770e1b9029cd85c79410bc6df1528a43fc2b9ad1",
        },
        type="Clustering",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["vie-Latn"],
        main_score="v_measure",
        date=("2025-07-29", "2025-07-30"),
        form=None,
        domains=None,
        task_subtypes=None,
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="machine-translated",
        socioeconomic_status=None,
        text_creation=None,
        bibtex_citation="""
@misc{pham2025vnmtebvietnamesemassivetext,
    title={VN-MTEB: Vietnamese Massive Text Embedding Benchmark},
    author={Loc Pham and Tung Luu and Thu Vo and Minh Nguyen and Viet Hoang},
    year={2025},
    eprint={2507.21500},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/2507.21500}
}
""",
        adapted_from=["TwentyNewsgroupsClustering"],
    )
