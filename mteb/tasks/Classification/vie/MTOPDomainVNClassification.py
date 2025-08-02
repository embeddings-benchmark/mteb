from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskClassification, MultilingualTask


class MTOPDomainVNClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="MTOPDomainVNClassification",
        dataset={
            "path": "GreenNode/mtop-domain-vn",
            "revision": "6e1ec8c54c018151c77472d94b1c0765230cf6ca"
        },
        description="MTOP: Multilingual Task-Oriented Semantic Parsing",
        reference="https://arxiv.org/pdf/2008.09335.pdf",
        category="s2s",
        type="Classification",
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
        adapted_from=["MTOPDomainClassification"],
    )
