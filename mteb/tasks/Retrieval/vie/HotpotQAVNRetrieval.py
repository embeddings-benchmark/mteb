from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class HotpotQAVN(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="HotpotQA-VN",
        dataset={
            "path": "GreenNode/hotpotqa-vn",
            "revision": "8a5220c7af5084f0d5d2afeb74f9c2b41b759ff0",
        },
        description=(
            "HotpotQA is a question answering dataset featuring natural, multi-hop questions, with strong"
            " supervision for supporting facts to enable more explainable question answering systems."
        ),
        reference="https://hotpotqa.github.io/",
        type="Retrieval",
        category="s2p",
        # eval_splits=["train", "dev", "test"],
        eval_splits=["test"],
        eval_langs=["vie-Latn"],
        main_score="ndcg_at_10",
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license=None,
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
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
        n_samples=None,
        avg_character_length=None,
    )
