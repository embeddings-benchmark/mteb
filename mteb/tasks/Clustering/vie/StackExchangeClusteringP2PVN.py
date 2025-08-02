from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from mteb.abstasks.AbsTaskClustering import AbsTaskClustering


class StackExchangeClusteringP2PVN(AbsTaskClustering):
    metadata = TaskMetadata(
        name="StackExchangeClusteringP2P-VN",
        description="Clustering of title+body from stackexchange. Clustering of 5 sets of 10k paragraphs and 5 sets of 5k paragraphs.",
        reference="https://arxiv.org/abs/2104.07081",
        dataset={
            "path": "GreenNode/stackexchange-clustering-p2p-vn",
            "revision": "8f154ee524a466850028531d21e1a62d958b8156",
        },
        type="Clustering",
        category="p2p",
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
        adapted_from=["StackExchangeClusteringP2P"],
    )
