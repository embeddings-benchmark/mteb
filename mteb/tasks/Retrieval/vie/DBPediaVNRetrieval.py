from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class DBPediaVN(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="DBPedia-VN",
        description="DBpedia-Entity is a standard test collection for entity search over the DBpedia knowledge base",
        reference="https://github.com/iai-group/DBpedia-Entity/",
        dataset={
            "path": "GreenNode/dbpedia-vn",
            "revision": "c3e20179fbcee16217ef9461a14a54b7faca9b63",
        },
        type="Retrieval",
        category="s2p",
        # eval_splits=["dev", "test"],
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
        adapted_from=["DBPedia"],
    )
