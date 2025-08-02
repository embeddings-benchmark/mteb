from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskClassification, MultilingualTask


class AmazonReviewsVNClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="AmazonReviewsVNClassification",
        dataset={
            "path": "GreenNode/amazon-reviews-multi-vn",
            "revision": "27da94deb6d4f44af789a3d70750fa506b79f189"
        },
        description="A collection of Amazon reviews specifically designed to aid research in multilingual text classification.",
        reference="https://arxiv.org/abs/2010.02573",
        category="s2s",
        type="Classification",
        eval_splits=[ "test"],
        eval_langs=["vie-Latn"],
        main_score="accuracy",
        ddate=("2025-07-29", "2025-07-30"),
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
        adapted_from=["AmazonReviewsClassification"],
    )
