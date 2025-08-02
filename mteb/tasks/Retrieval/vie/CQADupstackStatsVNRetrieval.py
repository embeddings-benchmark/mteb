from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class CQADupstackStatsVN(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CQADupstackStats-VN",
        description="CQADupStack: A Benchmark Data Set for Community Question-Answering Research",
        reference="http://nlp.cis.unimelb.edu.au/resources/cqadupstack/",
        dataset={
            "path": "GreenNode/cqadupstack-stats-vn",
            "revision": "6b8164f3af61f3bb7728724229ba36213fb46c25",
        },
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
        adapted_from=["CQADupstackStats"],
    )
