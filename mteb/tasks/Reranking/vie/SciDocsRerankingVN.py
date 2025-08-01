from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskReranking import AbsTaskReranking


class SciDocsRerankingVN(AbsTaskReranking):
    metadata = TaskMetadata(
        name="SciDocsRR-VN",
        description="Ranking of related scientific papers based on their title.",
        reference="https://allenai.org/data/scidocs",
        dataset={
            "path": "GreenNode/scidocs-reranking-vn",
            "revision": "c9ab36ae6c75f754df6f1e043c09b5e0b5547cac",
        },
        type="Reranking",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["vie-Latn"],
        main_score="map",
        date=("2000-01-01", "2020-12-31"),  # best guess
        form=["written"],
        domains=["Academic", "Non-fiction"],
        task_subtypes=["Scientific Reranking"],
        license="cc-by-4.0",
        socioeconomic_status="high",
        annotations_creators=None,
        dialect=None,
        text_creation="found",
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
        n_samples={"test": 19599},
        avg_character_length={"test": 69.0},
    )
