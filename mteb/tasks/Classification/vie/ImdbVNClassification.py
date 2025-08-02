from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskClassification


class ImdbVNClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="ImdbVNClassification",
        description="Large Movie Review Dataset",
        dataset={
            "path": "GreenNode/imdb-vn",
            "revision": "0dccb383ee26c90c99d03c8674cf40de642f099a"
        },
        reference="http://www.aclweb.org/anthology/P11-1015",
        type="Classification",
        category="p2p",
        eval_splits=["test"],
        eval_langs=["vie-Latn"],
        main_score="accuracy",
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
        adapted_from=["ImdbClassification"],
    )
