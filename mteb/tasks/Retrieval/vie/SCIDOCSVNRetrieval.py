from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class SCIDOCSVN(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="SCIDOCS-VN",
        dataset={
            "path": "GreenNode/scidocs-vn",
            "revision": "724cddfa9d328a193f303a0a9b7789468ac79f26",
        },
        description=(
            "SciDocs, a new evaluation benchmark consisting of seven document-level tasks ranging from citation"
            " prediction, to document classification and recommendation."
        ),
        reference="https://allenai.org/data/scidocs",
        type="Retrieval",
        category="s2p",
        eval_splits=["test"],
        eval_langs=["vie-Latn"],
        main_score="ndcg_at_10",
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
        adapted_from=["SCIDOCS"],
    )
